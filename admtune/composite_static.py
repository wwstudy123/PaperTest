"""TPS/P95真实小预算闭环；候选拒绝后有限刷新或提前停止，绝不重复填充预算。"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import time

from dotenv import load_dotenv
import yaml

from config import load_app_config
from llm_client import LLMClient
from .agents import AgenticDBBackend
from .budget import BudgetExhausted, Limits
from .calibration import hardware_context, process_sample, status_sample
from .continuous import ContinuousRun, DurableLedger, Journal, atomic_json, code_hash
from .contracts import Environment, Objective, fingerprint
from .dynamic import instance_lock
from .evidence_wire import EvidenceWireBackend
from .live import load_space
from .paper_pilot import PaperExecutor
from .provenance import source_provenance
from .replay_report import write_csv
from .security import redact
from .static_diagnostic import (CONCURRENCY_VARIABLES, DiagnosticConnector, cgroup_sample,
                                joint_utility, restore_concurrency_context)
from .timed_static import TimedRun

OBJECTIVE = "tps_per_p95:maximize"
POLICY = {"protocol_version": 1, "objective": OBJECTIVE, "label_precision_decimals": 6,
          "bootstrap": 6, "batch_size": 8, "beta": 2, "tolerated_drop": .1, "min_gain": .01,
          "max_refresh_total": 2, "refresh_per_decision": 1, "incumbent_repeat": False,
          "final_reflection_call_reserve": 1, "final_reflection_token_reserve": 10000,
          "final_reflection_seconds_reserve": 30}


def runtime_policy(gp_policy="legacy", initial_noise_repeats=0):
    if gp_policy not in {"legacy", "repeat-aware"} or isinstance(initial_noise_repeats, bool) or initial_noise_repeats not in {0, 3}:
        raise ValueError("Unknown GP policy or initial noise calibration schedule")
    policy = deepcopy(POLICY)
    if gp_policy == "legacy" and initial_noise_repeats == 0:
        return policy
    from .repeat_aware_gp import POLICY as REPEAT_POLICY
    return {**policy, "protocol_version": 2, "gp_policy": gp_policy,
            "gp_training": deepcopy(REPEAT_POLICY) if gp_policy == "repeat-aware" else "legacy_every_10",
            "initial_noise_repeats": initial_noise_repeats,
            "calibration_promotes_incumbent": False,
            "baseline_anchor": "first_initial_measurement_not_maximum_repeat"}


def pilot_limits(initial_noise_repeats=0):
    runtime_policy("legacy", initial_noise_repeats)
    return (Limits(trials=10, wall_seconds=1500, llm_calls=6, llm_tokens=24000,
                   recovery_reserve_seconds=60) if initial_noise_repeats else
            Limits(trials=8, wall_seconds=1200, llm_calls=8, llm_tokens=32000,
                   recovery_reserve_seconds=60))


class SearchStopped(RuntimeError):
    pass


class CompositeExecutor:
    """底层日志始终是TPS；只有返回的优化标签变为同次观测的规范化精度比值。"""
    def __init__(self, raw):
        self.raw, self.concurrency_initial, self.last = raw, None, None

    def __getattr__(self, name):
        return getattr(self.raw, name)

    @property
    def run_dir(self):
        return self.raw.run_dir

    @run_dir.setter
    def run_dir(self, value):
        self.raw.run_dir = value

    def preflight(self):
        result = self.raw.preflight()
        if "innodb_thread_concurrency" in self.raw.space.names:
            self.concurrency_initial = {k: int(v) for k, v in
                self.raw.db.show_variables(list(CONCURRENCY_VARIABLES)).items()}
            restore_concurrency_context(self.raw.db, self.concurrency_initial)
        return {**result, "optimization_objective": OBJECTIVE,
                "label_precision_decimals": 6, "concurrency_initial": self.concurrency_initial}

    def measure(self, config, context):
        initial = getattr(self.raw, "initial", None)
        if initial is not None and self.raw.readback() != initial.config:
            raise ValueError("Each trial must start from the restored common initial configuration")
        if self.concurrency_initial:
            restore_concurrency_context(self.raw.db, self.concurrency_initial)
        connector = getattr(self.raw, "connector", None)
        if connector is not None:
            connector.boundaries = None
            connector.command_evidence = []
        raw = self.raw.measure(config, context)
        metadata = {**raw.metadata, "raw_tps": raw.value, "optimization_objective": OBJECTIVE,
                    "label_precision_decimals": 6, "score_transform": "same_trial_tps_div_p95_ms"}
        for key in ("raw_tps", "p95_latency_ms"):
            value = metadata.get(key)
            if isinstance(value, (float, int)) and not math.isfinite(value):
                metadata[key+"_invalid_repr"] = repr(value)
                metadata[key] = None
        if connector is not None:
            metadata["telemetry"] = getattr(connector, "boundaries", None)
            metadata["command_evidence"] = getattr(connector, "command_evidence", [])
        value = None
        if raw.status == "measured" and raw.complete and raw.effective:
            try:
                exact = joint_utility(raw.value, raw.metadata.get("p95_latency_ms"))
                value = round(exact, 6)
                if value <= 0:
                    raise ValueError("Positive utility not representable at declared label precision")
                metadata["joint_value_unrounded"] = exact
            except ValueError as exc:
                return replace(raw, value=None, status="invalid_objective", constraints_ok=False,
                               metadata={**metadata, "reason": str(exc)})
        if value is not None:
            cg = (metadata.get("telemetry") or {}).get("before", {}).get("cgroup", {})
            quota = cg.get("quota_us")
            period = cg.get("period_us")
            self.last = {"raw_tps": raw.value, "p95_latency_ms": raw.metadata["p95_latency_ms"],
                         "objective_value": value, "objective_key": OBJECTIVE,
                         "constraints_ok": raw.constraints_ok,
                         "cpu_quota_cores": quota/period if quota and period and quota > 0 else None,
                         "telemetry_delta": (metadata.get("telemetry") or {}).get("delta")}
        return replace(raw, value=value, metadata=metadata)

    def restore_initial(self):
        self.raw.restore_initial()
        if self.concurrency_initial:
            restore_concurrency_context(self.raw.db, self.concurrency_initial)

    def diagnose(self):
        return {**self.raw.diagnose(), "objective_key": OBJECTIVE,
                "objective_description": "Maximize same-measurement TPS/P95(ms), NOT TPS alone; "
                                         "a throughput gain with large tail-latency regression can be worse.",
                "label_precision_decimals": 6, "latest_measured_observation": self.last}


class CompositeLedger(DurableLedger):
    def __init__(self, journal, limits, final_trial_seconds):
        super().__init__(journal, limits=limits)
        self.final_trial_seconds = final_trial_seconds
        self.finalizing = self.final_reflection = False

    def remaining_seconds(self):
        reserve = 0 if self.finalizing else self.final_trial_seconds + POLICY["final_reflection_seconds_reserve"]
        return max(0, super().remaining_seconds()-reserve)

    def reserve_llm(self, max_tokens):
        if not self.final_reflection and self.llm_calls >= self.limits.llm_calls-1:
            raise BudgetExhausted("One API call reserved for final incremental reflection")
        if not self.final_reflection and self.llm_tokens+max_tokens > (
                self.limits.llm_tokens-POLICY["final_reflection_token_reserve"]):
            raise BudgetExhausted("Tokens reserved for final incremental reflection")
        super().reserve_llm(max_tokens)


class CompositeRun(TimedRun):
    objective_key = OBJECTIVE

    def __init__(self, *args, gp_policy="legacy", initial_noise_repeats=0, **kwargs):
        runtime_policy(gp_policy, initial_noise_repeats)
        super().__init__(*args, **kwargs)
        old = self.state.get("objective_key")
        if old not in {None, OBJECTIVE} or any(
                t.objective_key != OBJECTIVE or t.environment_key != self.environment.key
                for t in self.store.trials()):
            raise ValueError("Do not reuse a TPS/fixed-rate/other-environment journal")
        self.state["objective_key"] = OBJECTIVE
        self.state.setdefault("refresh_count", 0)
        self.state.setdefault("retired_by_model", {})
        bound = self.journal.count > 0 or "gp_policy" in self.state or self.state.get("gp") is not None
        if bound and (self.state.get("gp_policy", "legacy") != gp_policy
                or self.state.get("initial_noise_repeats", 0) != initial_noise_repeats):
            raise ValueError("Do not change GP or calibration policy in an existing journal")
        self.gp_policy, self.initial_noise_repeats = gp_policy, initial_noise_repeats
        self.state.update(gp_policy=gp_policy, initial_noise_repeats=initial_noise_repeats)
        self.repeat_gp = None
        if gp_policy == "repeat-aware":
            from .repeat_aware_gp import RepeatAwareGP
            saved = self.state.get("repeat_gp")
            if self.state.get("gp") and not saved:
                raise ValueError("Repeat-aware GP requires matching repeat provenance")
            self.repeat_gp = (RepeatAwareGP.from_snapshot(self.space, saved) if saved
                              else RepeatAwareGP(self.space))
            self.gp = self.repeat_gp.model

    def fit(self):
        if self.gp_policy == "legacy":
            return super().fit()
        history = [t for t in self.store.trials() if t.trainable and t.constraints_ok]
        if any(t.objective_key != OBJECTIVE or t.environment_key != self.environment.key for t in history):
            raise ValueError("Refuse mixed objective/environment GP labels")
        if len(history) < 6 or not self.repeat_gp.update(history):
            return
        self.gp = self.repeat_gp.model
        self.state["repeat_gp"] = self.repeat_gp.snapshot()
        self.save_gp_snapshot(self.gp.snapshot([t.trial_id for t in history]))

    def step(self):
        record = super().step()
        if record.metadata["kind"] == "noise_calibration" and not (
                record.trainable and record.constraints_ok):
            raise RuntimeError("Invalid predeclared calibration observation; stop without budget filling")
        return record

    def checkpoint_reflection_allowed(self):
        # 10号最终复测不能提前消费新增证据或占用为阶段结束反思保留的调用。
        return not self.finalizing

    def experiences(self):
        records = [t for t in self.store.trials() if t.trainable and t.constraints_ok][-3:]
        return [{"type": "measurement", "trial_id": t.trial_id, "config": t.config,
                 "value": t.value, "objective_key": OBJECTIVE,
                 "raw_tps": t.metadata.get("raw_tps"), "p95_latency_ms": t.metadata.get("p95_latency_ms")}
                for t in records] + [r for r in super().experiences() if r["type"] == "reflection"]

    def replenish(self, seq):
        if self.state["pool"]:
            return
        # 不继承旧协议的seq+8冷却：空池应有限刷新，而不是盲测旧配置。
        self.state["next_proposal_at"] = seq
        try:
            super().replenish(seq)  # 仍保留一次有收费记账的格式纠错，不放宽校验。
        finally:
            self.journal.put("state", self.save_state())
        retired = set(self.state["retired_by_model"].get(str(self.gp.version), []))
        self.state["pool"] = [c for c in self.state["pool"] if fingerprint(c) not in retired]
        self.journal.put("state", self.save_state())

    def retire_pool(self, seq):
        pool = list(self.state["pool"])
        key = str(self.gp.version)
        retired = set(self.state["retired_by_model"].get(key, []))
        retired.update(fingerprint(c) for c in pool)
        self.state["retired_by_model"][key] = sorted(retired)
        self.state["pool"] = []
        self.team.feedback = {
            "reason": "No admissible novel candidate; propose new configurations, not the incumbent.",
            "objective_key": OBJECTIVE, "predictions_are_not_measurements": True,
            "model_version": self.gp.version, "rejected_unmeasured_configs": pool,
            "latest_measured_feedback": self.team.feedback}
        self.store.event("pool_retired", {"next_seq": seq, "model_version": self.gp.version,
                                         "config_ids": sorted(retired)})
        self.journal.put("state", self.save_state())

    def choose(self, seq):
        if not self.finalizing and 2 <= seq <= self.initial_noise_repeats+1:
            if self.state["baseline"] is None:
                raise ValueError("Valid initial baseline required before noise calibration")
            return "noise_calibration", self.state["initial"], {
                "rule": "predeclared_initial_noise_calibration", "ordinal": seq-1}
        decision = super().choose(seq)
        if decision[0] != "incumbent_repeat":
            return decision
        self.retire_pool(seq)
        if self.state["refresh_count"] >= POLICY["max_refresh_total"]:
            raise SearchStopped("Global bounded candidate refresh exhausted")
        self.state["refresh_count"] += 1
        self.store.event("candidate_refresh", {"next_seq": seq, "round": self.state["refresh_count"]})
        self.journal.put("state", self.save_state())
        decision = super().choose(seq)
        if decision[0] == "incumbent_repeat":
            self.retire_pool(seq)
            raise SearchStopped("No admissible novel candidate after one bounded refresh")
        return decision

    def export(self):
        # 不调用旧export：Trial.value在这里是综合分数，不能冒充TPS列。
        self.journal.audit()
        records = self.store.trials()
        best, rows = None, []
        for t in records:
            valid = t.trainable and t.constraints_ok
            kind = t.metadata["kind"]
            if valid and kind in {"baseline", "startup", "candidate", "audit"} and (
                    best is None or t.value > best.value):
                best = t
            rows.append({"trial": t.metadata["seq"], "kind": kind, "valid": valid, "status": t.status,
                "objective_value": t.value, "joint_value_unrounded": t.metadata.get("joint_value_unrounded"),
                "tps": t.metadata.get("raw_tps"), "p95_ms": t.metadata.get("p95_latency_ms"),
                "best_found_objective": None if best is None else best.value,
                "best_config_tps": None if best is None else best.metadata.get("raw_tps"),
                "best_config_p95_ms": None if best is None else best.metadata.get("p95_latency_ms"),
                "accepted": t.metadata["accepted"],
                "active_elapsed_seconds": t.metadata["active_elapsed_seconds"],
                "benchmark_log": t.metadata.get("benchmark_log"),
                "config_json": json.dumps(t.config, sort_keys=True)})
        atomic_json(self.output/"progress.json", {
            "objective": OBJECTIVE, "completed_attempts": len(records), "next_trial": self.journal.count+1,
            "incumbent": self.state["incumbent"], "best_value": self.state["best_value"],
            "refresh_count": self.state["refresh_count"], "budget": self.ledger.snapshot(),
            "counts_by_kind": {k: sum(t.metadata["kind"] == k for t in records)
                               for k in sorted({t.metadata["kind"] for t in records})}})
        write_csv(self.output/"curve.csv", rows, list(rows[0]) if rows else ["trial"])
        (self.output/"trials.jsonl").write_text("".join(json.dumps(t.to_dict(), allow_nan=False)+"\n" for t in records))
        (self.output/"events.jsonl").write_text("".join(json.dumps(
            {"seq": n, "kind": kind, "payload": json.loads(payload)})+"\n"
            for n, kind, payload in self.journal.db.execute("SELECT seq,kind,payload FROM events ORDER BY seq")))
        atomic_json(self.output/"reflections.json", self.store.reflections())

    def run_pilot(self):
        stop, final_value = "trial_cap", None
        while self.journal.count < self.ledger.limits.trials-1:
            try:
                self.ledger.ensure_time(self.executor.estimated_trial_seconds)
                self.step()
            except (SearchStopped, BudgetExhausted) as exc:
                stop = str(exc)
                self.store.event("search_stopped", {"reason": stop, "next_seq": self.journal.count+1})
                break
        self.finalizing = self.ledger.finalizing = True
        try:
            self.ledger.ensure_time(self.executor.estimated_trial_seconds)
            if self.state["baseline"] is not None and self.journal.count < self.ledger.limits.trials:
                final = self.step()
                if final.trainable and final.constraints_ok:
                    final_value = final.value
        except BudgetExhausted as exc:
            self.store.event("final_verification_not_admitted", {"reason": str(exc)})
        self.fit()
        reflected, reflection_error = False, None
        self.ledger.final_reflection = True
        try:
            reflected = self.reflector.trigger(["stage_end"], force=True)
        except (ValueError, RuntimeError) as exc:
            reflection_error = redact(str(exc))
            self.store.event("final_reflection_deferred", {"reason": reflection_error})
        self.journal.put("state", self.save_state())
        self.ledger.persist()
        self.export()
        records = self.store.trials()
        gp_candidates = sum(t.metadata["kind"] == "candidate" for t in records)
        gp_rejections = self.journal.db.execute(
            "SELECT COUNT(*) FROM events WHERE kind='prescreen_refused'").fetchone()[0]
        return {"status": "completed" if final_value is not None else "incomplete",
                "objective": OBJECTIVE, "label_precision_decimals": 6,
                "gp_policy": self.gp_policy, "initial_noise_repeats": self.initial_noise_repeats,
                "repeat_noise_audit": self.repeat_gp.audit if self.repeat_gp else None,
                "stop_reason": stop, "completed_trials": self.journal.count,
                "baseline_value": self.state["baseline"], "search_value": self.state["best_value"],
                "final_verification_value": final_value, "incumbent": self.state["incumbent"],
                "gp_training_count": len((self.state.get("gp") or {}).get("training_ids", [])),
                "prospective_predictions": sum(t.prediction is not None for t in records),
                "gp_selected_candidates": gp_candidates, "gp_rejection_decisions": gp_rejections,
                "refresh_count": self.state["refresh_count"], "final_reflection_committed": reflected,
                "reflection_error": reflection_error,
                "closed_loop_verified": bool(final_value and reflected and self.gp.fitted
                                             and any(t.prediction for t in records)
                                             and (gp_candidates or gp_rejections)),
                "budget": self.ledger.snapshot()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("target", "benchmark", "knobs", "output", "restore-audit"):
        parser.add_argument("--"+key, type=Path, required=True)
    parser.add_argument("--dedicated-instance", action="store_true")
    parser.add_argument("--allow-paid-llm", action="store_true")
    parser.add_argument("--gp-policy", choices=["legacy", "repeat-aware"], default="legacy")
    parser.add_argument("--initial-noise-repeats", type=int, choices=[0, 3], default=0)
    a = parser.parse_args()
    if not a.dedicated_instance or not a.allow_paid_llm:
        parser.error("需要专用实例和付费LLM显式授权")
    root = Path(__file__).resolve().parents[1]
    load_dotenv(override=True)
    if not all(os.getenv(k) for k in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL")):
        parser.error("需要显式模型、端点和密钥；禁止替身降级")
    app = load_app_config(project_root=root, target_path=a.target, benchmark_path=a.benchmark,
                          knobs_path=a.knobs, dry_run_override=False)
    w = app.benchmark.workload
    if (app.target.dbms != "mysql" or app.target.mysql.host not in {"localhost", "127.0.0.1"}
            or (w.duration, w.warmup_time, w.threads, w.rand_type, w.rand_seed) != (60, 15, 32, "uniform", 42)
            or w.script_path != "/usr/share/sysbench/oltp_read_only.lua"):
        parser.error("需要冻结的本地原生只读32线程uniform 15+60协议")
    lock = instance_lock(app.target.mysql.config_path)
    a.output.mkdir(parents=True, mode=0o700, exist_ok=False)
    space = load_space(a.knobs, memory_bytes=app.target.hardware_memory_bytes)
    connector = DiagnosticConnector(app.target.ssh, config_path=app.target.mysql.config_path,
                                    dbms="mysql", dry_run=False)
    raw = PaperExecutor(app, space, Objective(), a.output/"runtime", allow_mutations=True,
                        local=True, connector=connector)
    executor = CompositeExecutor(raw)
    journal = ledger = client = None
    result = {"status": "failed", "objective": OBJECTIVE}
    limits = pilot_limits(a.initial_noise_repeats)
    policy = runtime_policy(a.gp_policy, a.initial_noise_repeats)
    try:
        if connector.run("pgrep -x sysbench", check=False).ok:
            raise ValueError("已有Sysbench进程，拒绝并行测量或配置修改")
        preflight = executor.preflight()
        audit = json.loads(a.restore_audit.read_text())
        if not audit.get("completed") or audit["config_sha256"] != preflight["config_sha256"]:
            raise ValueError("本轮需要完成且匹配的冷恢复审计")
        pid = int(Path(app.target.mysql.pid_file).read_text())
        if Path(f"/proc/{pid}/cgroup").read_text() != Path("/proc/self/cgroup").read_text():
            raise ValueError("MySQL与采样进程cgroup不一致")
        def capture():
            return {"at": time.monotonic(), "cgroup": cgroup_sample(),
                    "process": process_sample(pid), "mysql": status_sample(raw.db),
                    "concurrency_variables": raw.db.show_variables(list(CONCURRENCY_VARIABLES))}
        connector.capture = capture
        extra = yaml.safe_load(a.target.read_text())["admtune"]
        environment = Environment("mysql", preflight["db_version"], socket.gethostname(), extra["data_id"],
            preflight["readiness"]["live_schema_hash"], fingerprint({
                "benchmark": asdict(app.benchmark), "space": space.describe(),
                "initial": preflight["current_config"], "policy": policy, "limits": asdict(limits)}))
        atomic_json(a.output/"manifest.json", {
            "method": "ADMTune-composite-pilot", "simulated": False, "objective": OBJECTIVE,
            "policy": policy, "limits": asdict(limits), "benchmark": asdict(app.benchmark),
            "space": space.describe(), "preflight": preflight, "restore": audit,
            "environment": asdict(environment), "hardware": hardware_context(),
            "code_sha256": code_hash(root), "model": os.getenv("OPENAI_MODEL"),
            "imported_historical_observations": 0, "created_at_utc": datetime.now(timezone.utc).isoformat(),
            **source_provenance()})
        backup = a.output/"initial.my.cnf"
        backup.write_text(raw.initial.file_content); backup.chmod(0o600)
        journal = Journal(a.output/"records.sqlite3")
        ledger = CompositeLedger(journal, limits, executor.estimated_trial_seconds)
        client = LLMClient.from_env(dry_run=False); client.allow_fallback = False
        backend = EvidenceWireBackend(AgenticDBBackend(client, ledger, trace_dir=a.output/"llm_calls"),
                                      audit_dir=a.output/"evidence")
        result = CompositeRun(journal, executor, space, environment, a.output, backend, ledger,
                              gp_policy=a.gp_policy, initial_noise_repeats=a.initial_noise_repeats).run_pilot()
    except Exception as exc:
        result.update(status="failed", error=redact(str(exc)))
    finally:
        try:
            executor.restore_initial()
            result["initial_configuration_restored"] = raw.initial is not None
        except Exception as exc:
            result.update(status="recovery_failed", recovery_error=redact(str(exc)))
        if client:
            atomic_json(a.output/"llm_usage.json", client.usage_summary())
        if ledger:
            ledger.persist()
            result["budget"] = ledger.snapshot()
            if result["budget"]["elapsed_seconds"] > limits.wall_seconds:
                result["status"] = "wall_budget_exceeded"
        if journal:
            journal.db.execute("PRAGMA wal_checkpoint(FULL)")
            journal.store.close()
        executor.close(); lock.close()
        result["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        atomic_json(a.output/"summary.json", result)
        atomic_json(a.output/"checksums.json", {
            str(f.relative_to(a.output)): hashlib.sha256(f.read_bytes()).hexdigest()
            for f in a.output.rglob("*") if f.is_file() and f.suffix in {".json", ".jsonl", ".csv", ".log"}
            and f.name != "checksums.json"})
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("status") == "completed" and result.get("closed_loop_verified") else 1


if __name__ == "__main__":
    raise SystemExit(main())
