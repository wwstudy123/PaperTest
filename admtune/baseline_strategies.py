"""有来源的数据库层基线复现；不是ADMTune消融的别名。不直接执行数据库或shell。"""
from collections import defaultdict
from copy import deepcopy
import math
from pathlib import Path

from auditor import AuditorSettings, TuningAuditor
from models import BenchmarkResult, LLMProposal
from validators.result_validator import ResultValidator
from .contracts import fingerprint

SOURCES = {
    "AgenticDB": {"repository": "xinyueyangiscas/AgenticDB",
                  "commit": "321cb7ca350bbf382e7a1696ccfff859ac53f315"},
    "AgentTune": {"repository": "intlyy/AgentTune",
                  "commit": "9b68872434665ef552b98999b05a785cb5d7aa4c"},
    "AgentTune-reproduction-reference": {"repository": "User-o2/AgentTune-Reproduction",
                  "commit": "10aafb5bf141e8ce62786ee883097fe399a9c433"},
}


class StrategyStopped(RuntimeError):
    pass


def rank_centroid(configs, names, top_k=2):
    """原版config_rank的competition ranks，非复现版的dense ranks。稳定处理并列。"""
    if not configs:
        return []
    means = {n: sum(c[n] for c in configs)/len(configs) for n in names}
    ranks = {}
    for name in names:
        distances = sorted(abs(c[name]-means[name]) for c in configs)
        rank = {}
        for index, value in enumerate(distances):
            rank.setdefault(value, index+1)
        ranks[name] = rank
    return sorted(configs, key=lambda c: sum(ranks[n][abs(c[n]-means[n])] for n in names))[:top_k]


def context_record(record):
    return {"trial_id": record["trial_id"], "config": record["config"],
            "objective_value": record["value"], "status": record["status"],
            "raw_tps": record["metadata"].get("raw_tps"),
            "p95_ms": record["metadata"].get("p95_latency_ms"),
            "measured_metrics": (record["metadata"].get("telemetry") or {}).get("delta"),
            "accepted": record["metadata"].get("accepted", False),
            "phase_id": record.get("phase_id"), "workload_context": record.get("context")}


def benchmark_result(record):
    return BenchmarkResult(score=record["value"] or 0, valid=bool(record["value"]),
        tps=record["metadata"].get("raw_tps"), p95_latency_ms=record["metadata"].get("p95_latency_ms"),
        primary_metric_name=record["metadata"].get("internal_score_metric", "tps_per_p95"),
        primary_metric_value=record["value"])


class AgenticDBStrategy:
    """上游单轮全局规划、执行反馈记忆、规则Auditor；只保留DB阶段。"""
    def __init__(self, space, llm, workload, event, *, objective_text=None):
        self.space, self.llm, self.workload, self.event = space, llm, workload, event
        self.auditor = TuningAuditor(AuditorSettings())
        self.rounds, self.pending_proposal = [], None
        self.best_record = None
        self.objective_text = objective_text or "最大化同次实测TPS/P95(ms)，不得当作纯TPS"
        self.earlier_phase_measurements = []
        self.physical_execution = "restored initial before/after each trial"
        self.current_workload_phase = None
        self.skill = (Path(__file__).resolve().parents[1]/"skills/agenticdb_skill.md").read_text()

    def propose(self, records, incumbent):
        for attempt in range(2):
            try:
                return self._propose(records, incumbent)
            except (ValueError, KeyError, TypeError) as exc:
                self.validation_feedback = str(exc)
                self.event("agenticdb_format_rejected", {"reason": str(exc), "attempt": attempt+1})
                if attempt:
                    raise

    def _propose(self, records, incumbent):
        best = self.best_record or records[0]
        audit_rounds = (self.rounds if self.current_workload_phase is None else [
            r for r in self.rounds if r["metadata"].get("workload_phase_id") == self.current_workload_phase])
        decision = self.auditor.audit(rounds=audit_rounds, current_phase="db",
                                      best_result=benchmark_result(best))
        self.event("agenticdb_auditor", decision.to_dict())
        if decision.stop:
            raise StrategyStopped(decision.reason)
        prompt = self.skill.replace("{{current_phase}}", "db").replace("{{task_brief}}", "目标："+self.objective_text)
        prompt = prompt.replace("{{benchmark_files}}", "按workload字段中的实际原生只读脚本和控制参数执行。")
        output = self.llm.complete("agenticdb_planner", {
            "objective": {"description": self.objective_text, "internal_score_direction": "maximize"},
            "workload": self.workload, "allowed_knob_space": self.space.describe(),
            "current_config": incumbent, "physical_execution": self.physical_execution,
            "history": self.rounds[-12:], "latest_observation": context_record(records[-1]),
            "earlier_phase_measurements": self.earlier_phase_measurements,
            "auditor": decision.to_dict(),
            "validation_feedback": getattr(self, "validation_feedback", None),
            "numeric_grid_rule": "普通数值必须等于lower+n*step（n为非负整数），特殊值仅取special。"
                                 "如lower=2000,step=500，2400不合法，2500合法。",
            "candidate_count": len(self.space.names),
            "global_plan_note": "目录少于8项，只能从该目录做全局组合并说明限制。",
        }, prompt, temperature=.2)
        if not isinstance(output.get("candidate_config"), dict) or "action_type" not in output:
            raise ValueError("Explicit action_type and candidate_config required")
        proposal = LLMProposal.from_dict(output)
        if proposal.action_type not in {"db_config", "repeat_benchmark"}:
            raise ValueError("AgenticDB DB-only rejects non-database actions")
        if proposal.action_type == "repeat_benchmark":
            if proposal.candidate_config:
                raise ValueError("repeat_benchmark cannot modify a configuration")
            cfg, kind = dict(incumbent), "native_repeat"
        else:
            cfg = self.space.merge(incumbent, proposal.candidate_config)
            if cfg == incumbent:
                raise ValueError("Unchanged db_config must explicitly request repeat_benchmark")
            kind = "candidate"
        self.pending_proposal = proposal.to_dict()
        self.validation_feedback = None
        return cfg, kind, {"rule": "agenticdb_global_planner", "proposal": self.pending_proposal}

    def observe(self, record):
        if record["metadata"]["kind"] in {"baseline", "verification"}:
            return
        if record["metadata"]["accepted"]:
            self.best_record = record
        self.rounds.append({"round_id": len(self.rounds)+1, "proposal": self.pending_proposal,
            "decision": "accepted" if record["metadata"]["accepted"] else "rollback",
            "result": benchmark_result(record).compact_dict(), "metadata": {
                "phase": "db", "workload_phase_id": self.current_workload_phase},
            "observation": context_record(record)})

    def accepts(self, record, incumbent):
        return ResultValidator().is_better(benchmark_result(record), benchmark_result(incumbent))[0]

    def snapshot(self):
        return {"rounds": deepcopy(self.rounds), "mechanisms": ["global_planning", "feedback_memory", "db_auditor"],
                "gp_used": False, "reflexion_used": False}


class AgentTuneStrategy:
    """论文7.1/7.2：匿名参数、range/step/special、R=3、k=2、top-5记忆。"""
    def __init__(self, space, llm, workload, event, *, objective_text=None):
        self.space, self.llm, self.workload, self.event = space, llm, workload, event
        self.aliases = {f"knob{i+1}": n for i, n in enumerate(space.names)}
        self.selected, self.ranges, self.beam, self.queue = [], {}, [], []
        self.layer_results, self.expected, self.layer = [], 0, 0
        self.objective_text = objective_text or "maximize same-measurement TPS/P95(ms), not TPS alone"
        self.initial_record = None

    def anonymous_feedback(self, record):
        # 匿名不能只改当前配置：实测历史和父节点反馈也不能泄露真实旋钮键名。
        return {**record, "config": {a: record["config"][self.aliases[a]] for a in self.selected}}

    def initialize(self):
        for attempt in range(2):
            try:
                return self._initialize()
            except (ValueError, TypeError, KeyError) as exc:
                if attempt:
                    raise
                self.format_feedback = str(exc)
                self.event("agenttune_format_retry", {"reason": str(exc), "max_retry": 1,
                    "selection_reused": bool(self.selected)})

    def _initialize(self):
        catalog = {a: {**self.space.knobs[n].__dict__, "name": a} for a, n in self.aliases.items()}
        for spec in catalog.values():
            spec.pop("name", None)
        selected = self.selected or self.llm.complete("agenttune_selector", {
            "catalog": catalog, "knob_num": min(4, len(catalog)), "workload": self.workload,
            "format_feedback": getattr(self, "format_feedback", None),
            "objective": self.objective_text},
            "你是AgentTune的KnobSelector。按负载选择指定数量的匿名旋钮。仅返回JSON对象"
            '{"selected":["knob1", "..."]}，不虚构目录外编号。', temperature=0)["selected"]
        if (not isinstance(selected, list) or len(selected) != min(4, len(catalog))
                or len(set(selected)) != len(selected) or set(selected)-set(catalog)):
            raise ValueError("AgentTune selector returned invalid aliases")
        self.selected = selected
        ranges = self.llm.complete("agenttune_pruner", {
            "catalog": {a: catalog[a] for a in selected}, "workload": self.workload,
            "format_feedback": getattr(self, "format_feedback", None)},
            "你是AgentTune的RangePruner，给所选旋钮提供硬边界内的优先范围、步长和特殊值。"
            "整数/浮点的min_value/max_value/step必须是数值且与目录硬步长对齐，不能为null；"
            "special_values只取目录已有值（可在普通范围外）。仅返回JSON："
            '{"ranges":{"knob1":{"min_value":1,"max_value":10,"step":1,"special_values":[]}}}。',
            temperature=0)["ranges"]
        if set(ranges) != set(selected):
            raise ValueError("AgentTune ranges must cover selected aliases exactly")
        for alias, r in ranges.items():
            k = self.space.knobs[self.aliases[alias]]
            if k.kind not in {"integer", "float"}:
                raise ValueError("This pinned experiment supports numeric knob catalog only")
            if set(r) != {"min_value", "max_value", "step", "special_values"}:
                raise ValueError("Invalid AgentTune range schema")
            lo, hi, step = r["min_value"], r["max_value"], r["step"]
            if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
                   for v in (lo, hi, step)):
                raise ValueError("Finite numeric ranges required")
            k.validate(lo); k.validate(hi)
            if (not k.lower <= lo <= hi <= k.upper or step < k.step
                    or abs(step/k.step-round(step/k.step)) > 1e-7):
                raise ValueError("AgentTune range or step violates hard bounds")
            if any(not any(type(v) is type(s) and v == s for s in k.special) for v in r["special_values"]):
                raise ValueError("Invented special value")
        self.selected, self.ranges = selected, ranges
        self.event("agenttune_initialization", {"aliases": self.aliases, "selected": selected, "ranges": ranges})

    def propose(self, records, incumbent):
        if not self.selected or not self.ranges:
            self.initialize()
            self.beam = [context_record(self.initial_record or records[0])]
        if not self.queue:
            if self.expected:
                if len(self.layer_results) != self.expected:
                    raise ValueError("Do not expand incomplete measured beam")
                self.beam = list(self.layer_results)
            memory = sorted([context_record(r) for r in records if r["value"] is not None],
                            key=lambda r: -r["objective_value"])[:5]
            pool, seen = [], set()
            self.layer += 1
            for parent_index, parent in enumerate(self.beam):
                for child in range(3):
                    try:
                        response = self.llm.complete("agenttune_recommender", {
                            "workload": self.workload, "ranges": self.ranges,
                            "knobs": {a: self.space.knobs[self.aliases[a]].description for a in self.selected},
                            "current_configuration": {a: parent["config"][self.aliases[a]] for a in self.selected},
                            "parent_feedback": self.anonymous_feedback(parent),
                            "top_performing_memory_window": [self.anonymous_feedback(r) for r in memory],
                            "objective": self.objective_text},
                            "你是AgentTune的ConfigurationRecommender。根据当前父配置、执行反馈和记忆窗口"
                            "独立采样一个改进配置。必须完整给出所有ranges中的匿名旋钮，不给目录外键；"
                            "遵循普通范围/步长或特殊值。不虚构已测性能。仅返回JSON："
                            '{"configuration":{"knob1":1,"knob2":2}}。',
                            temperature=1, top_p=.98)
                        delta = response["configuration"]
                        if not isinstance(delta, dict) or set(delta) != set(self.selected):
                            raise ValueError("Missing or unknown selected knob")
                        cfg = self.space.merge(parent["config"], {self.aliases[a]: v for a, v in delta.items()})
                        for a, value in delta.items():
                            r = self.ranges[a]
                            if value not in r["special_values"]:
                                if not r["min_value"] <= value <= r["max_value"]:
                                    raise ValueError("Outside pruned range")
                                lattice = (value-r["min_value"])/r["step"]
                                if abs(lattice-round(lattice)) > 1e-7:
                                    raise ValueError("Outside pruned lattice")
                        key = fingerprint(cfg)
                        if cfg == parent["config"] or key in seen:
                            continue
                        seen.add(key); pool.append(cfg)
                    except (ValueError, TypeError, KeyError) as exc:
                        self.event("agenttune_child_rejected", {"layer": self.layer,
                            "parent_index": parent_index, "child": child, "reason": str(exc)})
            chosen = rank_centroid(pool, [self.aliases[a] for a in self.selected], 2)
            self.event("agenttune_centroid_selection", {
                "layer": self.layer, "beam_parents": len(self.beam), "node_count": 3, "top_k": 2,
                "valid_pool": pool, "selected": chosen, "unmeasured_predictions": False})
            if not chosen:
                raise StrategyStopped("AgentTune has no valid child after original R=3 expansion")
            self.queue, self.layer_results, self.expected = list(chosen), [], len(chosen)
        return self.queue.pop(0), "candidate", {"rule": "agenttune_beam_centroid", "layer": self.layer}

    def observe(self, record):
        if record["metadata"]["kind"] == "candidate":
            self.layer_results.append(context_record(record))

    def accepts(self, record, incumbent):
        # AgentTune返回搜索树中实测最优，不强加ADMTune的1%晋升门槛。
        return record["value"] is not None and record["value"] > incumbent["value"]

    def snapshot(self):
        return {"aliases": self.aliases, "selected": self.selected, "ranges": self.ranges,
                "layer": self.layer, "beam": self.beam, "queue": self.queue,
                "layer_results": self.layer_results, "expected": self.expected,
                "beam_width": 2, "node_count": 3, "history_num": 5,
                "gp_used": False, "reflexion_used": False}
