"""统一 Sysbench 的 Default/ADMTune 小预算实测；不是完整基线矩阵。"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import time

from dotenv import load_dotenv
import yaml

from config import load_app_config
from llm_client import LLMClient
from .agents import AgenticDBBackend
from .budget import BudgetExhausted, Ledger, Limits
from .contracts import Environment, Objective, fingerprint
from .controller import Policy
from .dynamic import BatchedController, PooledTeam, instance_lock
from .evidence_wire import EvidenceWireBackend
from .live import load_space
from .memory import Store
from .provenance import source_provenance
from .real_execution import AgenticDBExecutor
from .reflection import Reflector
from .replay_report import write_csv
from .security import redact
from .workload import PhaseDetector, Window, WorkloadEncoder
from tools.mysql_resize import wait_buffer_pool_ready


class PaperExecutor(AgenticDBExecutor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.origin = time.monotonic()
        self.timeline = []
        self.resize_checks = []

    def _verify(self, config, invariants):
        super()._verify(config, invariants)
        self.resize_checks.append(wait_buffer_pool_ready(self.db, config["innodb_buffer_pool_size"]))

    def measure(self, config, context):
        started = time.monotonic() - self.origin
        result = super().measure(config, context)
        result.metadata.update(started_seconds=started, ended_seconds=time.monotonic() - self.origin)
        self.timeline.append({**asdict(result), "config": dict(config)})
        path = self.run_dir / "measurements.jsonl"
        with path.open("a") as f:
            f.write(json.dumps(self.timeline[-1], allow_nan=False) + "\n")
        path.chmod(0o600)
        return result


def tune(executor, space, environment, output, *, seed, trials):
    ledger = Ledger(Limits(trials=trials, wall_seconds=1800, llm_calls=4,
                           llm_tokens=20000, recovery_reserve_seconds=180))
    client = LLMClient.from_env(dry_run=False)
    client.allow_fallback = False
    backend = EvidenceWireBackend(AgenticDBBackend(client, ledger, trace_dir=output / "llm_calls"),
                                  audit_dir=output / "evidence_aliases")
    store = Store(output / "records.sqlite3")
    encoder = WorkloadEncoder([Window(0, 1, [[1, 0]])])
    detector = PhaseDetector(encoder, change_threshold=3, stable_threshold=1)
    detector.state, detector.phase = "stable", 1
    policy = Policy(bootstrap=4, max_batches=1, max_replans=0, final_repeats=1)
    controller = BatchedController(
        space=space, environment=environment, objective=Objective(), detector=detector,
        executor=executor, team=PooledTeam(space, backend, batch_size=6),
        reflector=Reflector(store, backend), store=store, ledger=ledger,
        policy=policy, seed=seed, snapshot_dir=output / "gp_snapshots",
    )
    result = {"complete": False}
    try:
        controller.begin_phase("static-1", [0.0])
        while controller.active:
            controller.step()
        result = controller.finalize()
        result["complete"] = bool(result["independent_verification_values"])
    except BudgetExhausted as exc:
        store.event("budget_stop", {"reason": str(exc)})
        result = controller.finalize()
        result.update(complete=False, budget_stop=str(exc))
    except Exception as exc:
        result.update(complete=False, error=redact(str(exc)))
    finally:
        records = [r.to_dict() for r in store.trials()]
        (output / "trials.jsonl").write_text("".join(json.dumps(r, allow_nan=False) + "\n" for r in records))
        usage = {**client.usage_summary(), "budget": ledger.snapshot()}
        (output / "llm_usage.json").write_text(json.dumps(usage, ensure_ascii=False, indent=2))
        store.close()
    return result, records


def curve_rows(method, records, measurements, *, seed):
    """P95 随 TPS 最优配置联动；不拼接来自不同配置的独立极值。"""
    rows, best = [], None
    by_log = {m["metadata"].get("benchmark_log"): m for m in measurements if m["metadata"].get("benchmark_log")}
    for step, record in enumerate(records, 1):
        metadata = record["metadata"]
        observation = by_log.get(metadata.get("benchmark_log"))
        kind = metadata.get("kind", "default")
        valid = bool(record["complete"] and record["effective"] and record["constraints_ok"]
                     and record["value"] is not None and record["status"] == "measured")
        if valid and kind != "verification" and (best is None or record["value"] > best["value"]):
            best = record
        rows.append({
            "method": method, "seed": seed, "trial": step, "kind": kind, "valid": valid,
            "tps": record["value"], "p95_ms": metadata.get("p95_latency_ms"),
            "ended_seconds": metadata.get("ended_seconds",
                                          None if observation is None else observation["metadata"]["ended_seconds"]),
            "best_tps": None if best is None else best["value"],
            "best_config_p95_ms": None if best is None else best["metadata"].get("p95_latency_ms"),
            "benchmark_log": metadata.get("benchmark_log"),
            "config_json": json.dumps(record["config"], sort_keys=True),
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["Default", "ADMTune"], required=True)
    for name in ("target", "benchmark", "knobs", "output", "restore-audit"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument("--default-measurements", type=int, default=3)
    parser.add_argument("--dedicated-instance", action="store_true")
    parser.add_argument("--allow-paid-llm", action="store_true")
    args = parser.parse_args()
    if not args.dedicated_instance or (args.method == "ADMTune" and not args.allow_paid_llm):
        parser.error("Explicit instance and paid-LLM authorization required")
    if not 6 <= args.trials <= 8 or not 1 <= args.default_measurements <= 3:
        parser.error("Pilot caps: 6..8 tuning trials and 1..3 Default measurements")
    load_dotenv(args.env_file, override=True)
    if args.method == "ADMTune" and not all(os.getenv(k) for k in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL")):
        parser.error("Explicit model/key/endpoint required")
    audit = json.loads(args.restore_audit.read_text())
    if not audit.get("completed"):
        parser.error("A completed cold-restore audit is required")
    app = load_app_config(project_root=Path(__file__).resolve().parent.parent, target_path=args.target,
                          benchmark_path=args.benchmark, knobs_path=args.knobs, dry_run_override=False)
    if app.target.dbms != "mysql" or app.target.mysql.host not in {"127.0.0.1", "localhost"}:
        parser.error("Dedicated local MySQL required")
    if (app.benchmark.workload.duration < 60 or app.benchmark.workload.warmup_time < 15
            or app.benchmark.workload.workload_script != "oltp_read_only.lua"):
        parser.error("Frozen pilot requires read-only Sysbench, >=15s warmup and >=60s measurement")
    lock = instance_lock(app.target.mysql.config_path)
    args.output.mkdir(mode=0o700, parents=True, exist_ok=False)
    space = load_space(args.knobs, memory_bytes=app.target.hardware_memory_bytes)
    executor = PaperExecutor(app, space, Objective(), args.output, allow_mutations=True, local=True)
    result, records = {"complete": False}, []
    try:
        preflight = executor.preflight()
        if preflight["config_sha256"] != audit["config_sha256"]:
            raise ValueError("Initial configuration differs from restored configuration")
        extra = yaml.safe_load(args.target.read_text())["admtune"]
        environment = Environment("mysql", preflight["db_version"], socket.gethostname(), extra["data_id"],
                                  preflight["readiness"]["live_schema_hash"],
                                  fingerprint({"benchmark": asdict(app.benchmark), "space": space.describe(),
                                               "initial": preflight["current_config"]}))
        manifest = {
            "method": args.method, "seed": args.seed, "simulated": False, "pilot": True,
            "full_baseline_matrix": False, "objective": "tps:maximize",
            "benchmark": asdict(app.benchmark), "preflight": preflight,
            "environment": asdict(environment), "space": space.describe(),
            "restore_audit": audit, "restore_audit_sha256": hashlib.sha256(args.restore_audit.read_bytes()).hexdigest(),
            "data_snapshot_restored_by_harness": True, "created_at_utc": datetime.now(timezone.utc).isoformat(),
            **source_provenance(),
        }
        (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
        if args.method == "Default":
            config = dict(preflight["current_config"])
            for _ in range(args.default_measurements):
                measured = executor.measure(config, [0.0])
                records.append({**asdict(measured), "config": config})
            result = {"complete": all(r["status"] == "measured" and r["complete"] and r["constraints_ok"] for r in records),
                      "llm_calls": 0, "configuration_search": False}
            (args.output / "trials.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
        else:
            result, records = tune(executor, space, environment, args.output, seed=args.seed, trials=args.trials)
    except Exception as exc:
        result.update(complete=False, error=redact(str(exc)))
    finally:
        try:
            executor.restore_initial()
            result["initial_configuration_restored"] = executor.initial is not None
            if executor.initial is not None:
                executor._verify(executor.initial.config, executor.initial.invariants)
        except Exception as exc:
            result.update(complete=False, recovery_error=redact(str(exc)))
        rows = curve_rows(args.method, records, executor.timeline, seed=args.seed)
        write_csv(args.output / "curve.csv", rows, list(rows[0]) if rows else ["method"])
        (args.output / "resize_checks.json").write_text(json.dumps(executor.resize_checks, indent=2))
        result.update(method=args.method, pilot=True, full_baseline_matrix=False,
                      finished_at_utc=datetime.now(timezone.utc).isoformat(),
                      warning="单种子静态小预算试验；不能作为六方法正式结论。")
        (args.output / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
        checksums = {str(p.relative_to(args.output)): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in args.output.rglob("*") if p.is_file()
                     and p.suffix in {".json", ".jsonl", ".csv", ".log"} and p.name != "checksums.json"}
        (args.output / "checksums.json").write_text(json.dumps(checksums, indent=2))
        executor.close()
        lock.close()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["complete"] and result.get("initial_configuration_restored") else 1


if __name__ == "__main__":
    raise SystemExit(main())
