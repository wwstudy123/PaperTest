"""固定来源、统一执行协议的真实基线入口；无启发式替身、无论文曲线导入。"""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import shutil
import time

from dotenv import load_dotenv

from config import load_app_config
from llm_client import LLMClient
from .baseline_strategies import AgenticDBStrategy, AgentTuneStrategy, SOURCES, StrategyStopped
from .budget import BudgetExhausted, Limits
from .calibration import hardware_context, process_sample, status_sample
from .composite_static import CompositeExecutor, CompositeLedger, CompositeRun, OBJECTIVE
from .continuous import DurableLedger, Journal, atomic_json, code_hash
from .contracts import Environment, Trial, Objective, fingerprint
from .execution import Measurement
from .dynamic import instance_lock
from .live import load_space
from .paper_pilot import PaperExecutor
from .provenance import source_provenance
from .replay_report import write_csv
from .security import redact
from .static_diagnostic import CONCURRENCY_VARIABLES, DiagnosticConnector, cgroup_sample

POLICY = {"version": "formal_db_baselines_v1", "objective": OBJECTIVE, "trials": 8,
    "wall_seconds": 1500, "llm_calls": 20, "llm_tokens": 64000, "default_measurements": 3,
    "workload_seed": 42, "workload": "native_read_only_uniform_32_15plus60",
    "reset": "full_logical_checksum_and_cold_restart_no_buffer_pool_load",
    "historical_observations_imported": 0, "label_decimals": 6,
    "agenttune": {"top_k": 2, "node_count": 3, "history_num": 5,
                  "selector_temperature": 0, "pruner_temperature": 0,
                  "recommender_temperature": 1, "top_p": .98},
    "agenticdb": {"db_only": True, "acceptance_min_gain": .01},
    "admtune": {"gp_policy": "repeat-aware", "initial_noise_repeats": 0},
    "provider_seed": None}


class FinalLedger(DurableLedger):
    finalizing = False
    def __init__(self, journal, limits, final_seconds):
        super().__init__(journal, limits=limits)
        self.final_seconds = final_seconds
    def remaining_seconds(self):
        return max(0., super().remaining_seconds()-(0 if self.finalizing else self.final_seconds))


class BaselineLLM:
    """每个AgentTune子节点都真实独立调用，不使用同prompt缓存或隐式重试。"""
    def __init__(self, client, ledger, output):
        if client.dry_run or not client.api_key:
            raise ValueError("Real explicit LLM required")
        self.client, self.ledger, self.output = client, ledger, output
        client.response_retries = 0
        client.allow_fallback = False
    def complete(self, role, payload, system, *, temperature=.2, top_p=None):
        text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
        reserved = len((system+text).encode())+1500+64
        self.ledger.reserve_llm(reserved)
        trace = {"role": role, "payload": payload, "system_prompt": system,
                 "model": self.client.model, "sampling": {"temperature": temperature, "top_p": top_p},
                 "reservation": reserved, "cache_disabled": True}
        started = time.monotonic()
        try:
            result = self.client._call_openai_compatible(text, operation="baseline_"+role,
                system_prompt=system, max_output_tokens=1500, temperature=temperature, top_p=top_p,
                timeout_s=max(1, min(self.client.timeout_s, int(self.ledger.remaining_seconds()-60))))
            trace["response"] = result
            if not isinstance(result, dict):
                raise ValueError("Object response required")
            return result
        except Exception as exc:
            trace["error"] = redact(str(exc))
            raise
        finally:
            usage = self.client.consume_last_usage()
            trace.update(usage=usage, elapsed_seconds=time.monotonic()-started)
            self.output.mkdir(parents=True, exist_ok=True, mode=0o700)
            atomic_json(self.output/f"{self.ledger.llm_calls:03d}-{role}.json",
                        json.loads(redact(json.dumps(trace, ensure_ascii=False)).replace(
                            self.client.api_key, "<redacted>")))
            self.ledger.settle_llm(reserved, usage.get("total_tokens") if usage else None)


def workload_features(app, preflight):
    # 与原生只读Lua一致的代表性SQL，不冒充捕获到的请求日志。
    import sqlglot
    from sqlglot import exp
    queries = ["SELECT c FROM sbtest1 WHERE id=123"]*10 + [
        "SELECT c FROM sbtest1 WHERE id BETWEEN 123 AND 222",
        "SELECT SUM(k) FROM sbtest1 WHERE id BETWEEN 123 AND 222",
        "SELECT c FROM sbtest1 WHERE id BETWEEN 123 AND 222 ORDER BY c",
        "SELECT DISTINCT c FROM sbtest1 WHERE id BETWEEN 123 AND 222 ORDER BY c"]
    trees = [sqlglot.parse_one(q, read="mysql") for q in queries]
    return {"workload": asdict(app.benchmark.workload),
            "feature_source": "rule-based representative instantiation, not captured query trace",
            "selects_per_transaction": 14, "read_ratio": 1., "write_ratio": 0.,
            "average_tables_per_select": sum(len(list(t.find_all(exp.Table))) for t in trees)/14,
            "aggregate_nodes_per_select": sum(len(list(t.find_all(exp.AggFunc))) for t in trees)/14,
            "order_by_select_fraction": sum(t.find(exp.Order) is not None for t in trees)/14,
            "data_access": {"distribution": "uniform", "tables": 4, "rows_per_table": 2000000,
                            "range_size": 100, "table_selection": "uniform"},
            "schema_hash": preflight["readiness"]["live_schema_hash"],
            "db_version": preflight["db_version"], "hardware": hardware_context(),
            "representative_queries": queries[-5:]}


class BaselineRun:
    def __init__(self, method, journal, executor, space, env, output, ledger, strategy=None):
        self.method, self.j, self.executor, self.space = method, journal, executor, space
        self.env, self.output, self.ledger, self.strategy = env, output, ledger, strategy
        self.state = {"method": method, "incumbent": executor.readback(), "baseline": None, "best_value": None}
        self.best_record = None
        self.invalid_streak = 0
        if journal.count:
            self.state = journal.get("state")
            if self.state["method"] != method:
                raise ValueError("Cannot resume another method")
            for t in journal.store.trials():
                if t.metadata["kind"] == "baseline" or t.metadata.get("accepted"):
                    self.best_record = t.to_dict()
            if strategy and method == "AgenticDB":
                strategy.rounds = self.state.get("strategy", {}).get("rounds", [])
                strategy.best_record = self.best_record

    def step(self, config, kind, decision):
        if self.j.count >= self.ledger.limits.trials:
            raise BudgetExhausted("Real measurement cap reached")
        self.ledger.ensure_time(self.executor.estimated_trial_seconds)
        config = self.space.merge(config, {})
        submitted = self.j.reserve({"kind": kind, "config": config, "prediction": None,
                                    "decision": decision}, self.state)
        self.ledger.trials = self.j.count; self.ledger.persist()
        folder = self.output/"trials"/f"{self.j.count:06d}"
        folder.mkdir(parents=True, mode=0o700, exist_ok=False)
        self.executor.run_dir = folder
        offset = len(self.executor.resize_checks)
        try:
            m = self.executor.measure(config, [0.])
        except Exception as exc:
            m = Measurement(None, "execution_failed", False, False, False, {"reason": redact(str(exc))})
        finally:
            self.executor.restore_initial()
        meta = {**m.metadata, "seq": self.j.count, "kind": kind, "accepted": False,
                "accepted_new_configuration": False,
                "resize_checks": self.executor.resize_checks[offset:],
                "active_elapsed_seconds": self.ledger.snapshot()["elapsed_seconds"]}
        if meta.get("benchmark_log"):
            meta["benchmark_log"] = str(folder.relative_to(self.output)/meta["benchmark_log"])
        record = Trial(submitted["trial_id"], "formal-static", self.env.key, OBJECTIVE, config, [0.],
                       m.status, m.value, m.complete, m.effective, m.constraints_ok, None, meta)
        data = record.to_dict()
        if record.trainable and record.constraints_ok:
            if kind == "baseline":
                self.state.update(baseline=record.value, best_value=record.value, incumbent=config)
                self.best_record = data
            elif kind != "verification" and self.strategy and self.strategy.accepts(data, self.best_record):
                data["metadata"].update(accepted=True,
                    accepted_new_configuration=config != self.state["incumbent"])
                self.state.update(best_value=record.value, incumbent=config)
                self.best_record = data
        self.invalid_streak = 0 if record.trainable and record.constraints_ok else self.invalid_streak+1
        if self.strategy:
            self.strategy.observe(data)
            self.state["strategy"] = self.strategy.snapshot()
        self.j.finish(submitted, Trial(**data), self.state)
        self.ledger.persist()
        self.export()
        if kind == "baseline" and not (record.trainable and record.constraints_ok):
            raise RuntimeError("Invalid baseline")
        return data

    def export(self):
        self.j.audit()
        records = [r.to_dict() for r in self.j.store.trials()]
        (self.output/"trials.jsonl").write_text("".join(json.dumps(r, allow_nan=False)+"\n" for r in records))
        atomic_json(self.output/"progress.json", {**self.state, "completed_trials": len(records),
                                                "budget": self.ledger.snapshot()})
        write_csv(self.output/"curve.csv", [
            {"trial": r["metadata"]["seq"], "kind": r["metadata"]["kind"],
             "status": r["status"], "objective_value": r["value"],
             "tps": r["metadata"].get("raw_tps"), "p95_ms": r["metadata"].get("p95_latency_ms"),
             "accepted": r["metadata"]["accepted"],
             "accepted_new_configuration": r["metadata"]["accepted_new_configuration"],
             "active_elapsed_seconds": r["metadata"]["active_elapsed_seconds"],
             "config_json": json.dumps(r["config"], sort_keys=True)} for r in records],
            ["trial","kind","status","objective_value","tps","p95_ms","accepted",
             "accepted_new_configuration","active_elapsed_seconds","config_json"])
        (self.output/"events.jsonl").write_text("".join(json.dumps(
            {"seq": n, "kind": k, "payload": json.loads(v)})+"\n"
            for n,k,v in self.j.db.execute("SELECT seq,kind,payload FROM events ORDER BY seq")))

    def run(self):
        stop, error = "trial_cap", None
        if self.j.count == 0:
            self.step(self.state["incumbent"], "baseline", {"rule": "common_baseline"})
        try:
            if self.method == "Default":
                self.step(self.state["incumbent"], "default_repeat", {"rule": "three_default_measurements"})
            else:
                while self.j.count < self.ledger.limits.trials-1:
                    self.ledger.ensure_time(self.executor.estimated_trial_seconds)
                    config, kind, decision = self.strategy.propose(
                        [t.to_dict() for t in self.j.store.trials()], self.state["incumbent"])
                    self.step(config, kind, decision)
                    if self.invalid_streak >= 3:
                        raise StrategyStopped("Three consecutive invalid measurements")
        except (StrategyStopped, BudgetExhausted) as exc:
            stop = str(exc)
        except (ValueError, KeyError, TypeError) as exc:
            stop, error = "strategy_validation_failed", str(exc)
        self.ledger.finalizing = True
        final = self.step(self.state["incumbent"], "verification", {"rule": "independent_final"})
        self.export()
        valid_candidates = sum(t.trainable and t.constraints_ok and t.metadata["kind"] == "candidate"
                               for t in self.j.store.trials())
        final_valid = final["status"] == "measured" and final["complete"] and final["effective"] and final["constraints_ok"]
        return {"status": "completed" if final_valid and error is None and (self.method == "Default" or valid_candidates > 0) else "failed",
                "method": self.method, "objective": OBJECTIVE, "completed_trials": self.j.count,
                "valid_candidate_measurements": valid_candidates,
                "baseline_value": self.state["baseline"], "search_value": self.state["best_value"],
                "final_verification_value": final["value"], "incumbent": self.state["incumbent"],
                "stop_reason": stop, "error": error,
                "baseline_ready": final_valid and error is None and (self.method == "Default" or valid_candidates > 0),
                "budget": self.ledger.snapshot()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["Default", "AgenticDB", "AgentTune", "ADMTune"], required=True)
    for name in ("target", "benchmark", "knobs", "output", "reset-audit"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument("--dedicated-instance", action="store_true")
    parser.add_argument("--allow-paid-llm", action="store_true")
    parser.add_argument("--resume", action="store_true", help="续接失败基线；保留所有已消耗测量/API/时间")
    a = parser.parse_args()
    if not a.dedicated_instance or (a.method != "Default" and not a.allow_paid_llm):
        parser.error("Explicit dedicated instance and paid-LLM authorization required")
    if a.repeat < 0:
        parser.error("Nonnegative independent repeat ID required")
    if not shutil.which("mysql") or not shutil.which("sysbench"):
        parser.error("mysql/sysbench must be on PATH before starting real measurements or paid calls")
    root = Path(__file__).resolve().parents[1]
    load_dotenv(root/".env", override=True)
    app = load_app_config(project_root=root, target_path=a.target, benchmark_path=a.benchmark,
                          knobs_path=a.knobs, dry_run_override=False)
    w = app.benchmark.workload
    if (w.threads, w.duration, w.warmup_time, w.rand_type, w.rand_seed, w.script_path) != (
            32, 60, 15, "uniform", 42, "/usr/share/sysbench/oltp_read_only.lua"):
        parser.error("Frozen native read-only workload required")
    if app.target.dbms != "mysql" or app.target.mysql.host not in {"localhost","127.0.0.1"}:
        parser.error("Local dedicated MySQL required")
    lock = instance_lock(app.target.mysql.config_path)
    if a.resume:
        old = json.loads((a.output/"summary.json").read_text())
        if a.method != "AgenticDB" or old["status"] != "failed" or old["method"] != a.method:
            parser.error("Only failed AgenticDB validation run can be continued")
        atomic_json(a.output/"summary-before-continuation.json", old)
        if (a.output/"llm_usage.json").exists():
            atomic_json(a.output/"llm_usage-before-continuation.json",
                        json.loads((a.output/"llm_usage.json").read_text()))
    else:
        a.output.mkdir(mode=0o700, parents=True, exist_ok=False)
    space = load_space(a.knobs, memory_bytes=app.target.hardware_memory_bytes)
    connector = DiagnosticConnector(app.target.ssh, config_path=app.target.mysql.config_path, dbms="mysql", dry_run=False)
    raw = PaperExecutor(app, space, Objective(),
                        a.output/"runtime", allow_mutations=True, local=True, connector=connector)
    executor = CompositeExecutor(raw)
    j = ledger = client = None
    result = {"status": "failed", "method": a.method}
    limits = Limits(trials=8, wall_seconds=1500, llm_calls=20, llm_tokens=64000, recovery_reserve_seconds=60)
    try:
        if connector.run("pgrep -x sysbench", check=False).ok:
            raise ValueError("Another benchmark active")
        pre = executor.preflight()
        reset = json.loads(a.reset_audit.read_text())
        if not reset.get("completed") or reset["config_sha256"] != pre["config_sha256"]:
            raise ValueError("Completed common reset audit required")
        pid = int(Path(app.target.mysql.pid_file).read_text())
        assert Path(f"/proc/{pid}/cgroup").read_text() == Path("/proc/self/cgroup").read_text()
        connector.capture = lambda: {"at": time.monotonic(), "cgroup": cgroup_sample(),
            "process": process_sample(pid), "mysql": status_sample(raw.db),
            "concurrency_variables": raw.db.show_variables(list(CONCURRENCY_VARIABLES))}
        env = Environment("mysql", pre["db_version"], socket.gethostname(), "formal-workset-v2",
            pre["readiness"]["live_schema_hash"], fingerprint({
                "common_policy": POLICY, "method": a.method, "repeat": a.repeat,
                "space": space.describe(), "benchmark": asdict(app.benchmark),
                "initial": pre["current_config"], "logical_checksums": reset["logical_checksums"]}))
        manifest_path = a.output/("continuation-manifest.json" if a.resume else "manifest.json")
        atomic_json(manifest_path, {
            "method": a.method, "display_method": "AgenticDB (DB-only)" if a.method=="AgenticDB" else a.method,
            "objective": OBJECTIVE, "comparison_protocol": POLICY, "limits": asdict(limits),
            "repeat": a.repeat, "provider_seed_controlled": False, "simulated": False,
            "sources": SOURCES, "benchmark": asdict(app.benchmark), "space": space.describe(),
            "preflight": pre, "reset": reset, "environment": asdict(env), "hardware": hardware_context(),
            "model": os.getenv("OPENAI_MODEL"), "code_sha256": code_hash(root),
            "imported_historical_observations": 0, **source_provenance()})
        j = Journal(a.output/"records.sqlite3")
        if a.method == "ADMTune":
            from .agents import AgenticDBBackend
            from .evidence_wire import EvidenceWireBackend
            ledger = CompositeLedger(j, limits, executor.estimated_trial_seconds)
            client = LLMClient.from_env(dry_run=False); client.allow_fallback = False
            backend = EvidenceWireBackend(AgenticDBBackend(client, ledger, trace_dir=a.output/"llm_calls"),
                                          audit_dir=a.output/"evidence")
            result = CompositeRun(j, executor, space, env, a.output, backend, ledger,
                                  gp_policy="repeat-aware").run_pilot()
            result["method"] = "ADMTune"
        else:
            ledger = FinalLedger(j, limits, executor.estimated_trial_seconds)
            strategy = None
            if a.method != "Default":
                client = LLMClient.from_env(dry_run=False)
                llm = BaselineLLM(client, ledger, a.output/"llm_calls")
                cls = AgenticDBStrategy if a.method=="AgenticDB" else AgentTuneStrategy
                strategy = cls(space, llm, workload_features(app, pre), j.store.event)
                if a.resume:
                    strategy.validation_feedback = old.get("error")
            result = BaselineRun(a.method, j, executor, space, env, a.output, ledger, strategy).run()
    except Exception as exc:
        result.update(status="failed", error=redact(str(exc)))
    finally:
        try:
            executor.restore_initial(); result["initial_configuration_restored"] = raw.initial is not None
        except Exception as exc:
            result.update(status="recovery_failed", recovery_error=redact(str(exc)))
        if client:
            atomic_json(a.output/"llm_usage.json", client.usage_summary())
        if ledger:
            ledger.persist(); result["budget"] = ledger.snapshot()
            if result["budget"]["elapsed_seconds"] > limits.wall_seconds:
                result.update(status="wall_budget_exceeded")
        if j:
            j.db.execute("PRAGMA wal_checkpoint(FULL)"); j.store.close()
        executor.close(); lock.close()
        result["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        atomic_json(a.output/"summary.json", result)
        atomic_json(a.output/"checksums.json", {
            str(p.relative_to(a.output)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in a.output.rglob("*") if p.is_file() and p.suffix in {".json",".jsonl",".csv",".log"}
            and p.name != "checksums.json"})
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"]=="completed" and result.get("initial_configuration_restored") else 1


if __name__ == "__main__":
    raise SystemExit(main())
