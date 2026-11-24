"""零 LLM：冻结热点/均匀分布与缓冲池 B-C-B 交错校准，不计入正式搜索。"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

from dotenv import load_dotenv

from config import load_app_config
from .contracts import Objective
from .dynamic import instance_lock
from .live import load_space
from .paper_pilot import PaperExecutor
from .provenance import source_provenance
from .replay_report import write_csv
from .security import redact


def plan():
    return [(dist, candidate) for dist, block in
            (("special", (False, True, False)), ("uniform", (False, True, False, True, False)))
            for candidate in block]


def evaluate(records):
    rows = [r for r in records if r["distribution"] == "uniform"]
    result = {"passed": False, "chosen_distribution": None, "min_gain": .05,
              "max_reference_drift": .10, "drift_multiplier": 2, "llm_calls": 0}
    if len(rows) != 5 or not all(r["valid"] for r in rows):
        return {**result, "reason": "Incomplete or invalid uniform B-C-B-C-B block"}
    b = [rows[i]["tps"] for i in (0, 2, 4)]
    drift = max(abs(b[i + 1] / b[i] - 1) for i in (0, 1))
    gains = [rows[i]["tps"] / ((rows[i-1]["tps"] + rows[i+1]["tps"]) / 2) - 1 for i in (1, 3)]
    passed = drift <= .10 and all(g > max(.05, 2 * drift) for g in gains)
    return {**result, "passed": passed, "chosen_distribution": "uniform" if passed else None,
            "paired_gains": gains, "reference_drift": drift,
            "reason": "Predeclared signal gate passed" if passed else "Signal insufficient; do not start paid 50-trial run"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("target", "benchmark", "knobs", "output", "restore-audit"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--dedicated-instance", action="store_true")
    a = p.parse_args()
    if not a.dedicated_instance:
        p.error("Dedicated instance required")
    load_dotenv(override=True)
    app = load_app_config(project_root=Path(__file__).resolve().parent.parent,
                          target_path=a.target, benchmark_path=a.benchmark, knobs_path=a.knobs, dry_run_override=False)
    if app.benchmark.workload.script_path != "/usr/share/sysbench/oltp_read_only.lua":
        p.error("Read-only native Sysbench required")
    if app.benchmark.workload.duration < 60 or app.benchmark.workload.warmup_time < 15:
        p.error("Frozen 15+60 minimum timing required")
    audit = json.loads(a.restore_audit.read_text())
    if not audit.get("completed"):
        p.error("Completed cold restore required")
    lock = instance_lock(app.target.mysql.config_path)
    a.output.mkdir(mode=0o700, parents=True, exist_ok=False)
    space = load_space(a.knobs, memory_bytes=app.target.hardware_memory_bytes)
    executor = PaperExecutor(app, space, Objective(), a.output, allow_mutations=True, local=True)
    records, result = [], {"completed": False}
    try:
        preflight = executor.preflight()
        if preflight["config_sha256"] != audit["config_sha256"]:
            raise ValueError("Restored configuration mismatch")
        initial = dict(preflight["current_config"])
        if initial["innodb_buffer_pool_size"] != 1024**3:
            raise ValueError("Predeclared 1-GiB baseline required")
        candidate = space.merge(initial, {"innodb_buffer_pool_size": 2 * 1024**3})
        manifest = {"mode": "zero_llm_discrimination", "plan": plan(), "benchmark": asdict(app.benchmark),
                    "preflight": preflight, "restore": audit, "gate": evaluate([]), **source_provenance()}
        (a.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
        started = time.monotonic()
        for i, (distribution, is_candidate) in enumerate(plan(), 1):
            if time.monotonic() - started > 1500:
                raise TimeoutError("Calibration wall budget reached")
            app.benchmark.workload.rand_type = distribution
            config = candidate if is_candidate else initial
            m = executor.measure(config, [0.0])
            r = {"trial": i, "distribution": distribution, "candidate": is_candidate,
                 "config": config, "valid": m.status == "measured" and m.complete and m.effective and m.constraints_ok,
                 "tps": m.value, "p95_ms": m.metadata.get("p95_latency_ms"), "measurement": asdict(m)}
            records.append(r)
            with (a.output / "trials.jsonl").open("a") as f:
                f.write(json.dumps(r, allow_nan=False) + "\n")
            (a.output / "progress.json").write_text(json.dumps({"completed_trials": i, "last": r}, indent=2))
            if not r["valid"]:
                raise ValueError("Invalid calibration observation; no paid run")
        result = {"completed": True, "gate": evaluate(records), "llm_calls": 0}
    except Exception as exc:
        result["error"] = redact(str(exc))
    finally:
        try:
            executor.restore_initial()
            result["initial_configuration_restored"] = executor.initial is not None
        except Exception as exc:
            result.update(completed=False, recovery_error=redact(str(exc)))
        rows = [{k: r[k] for k in ("trial", "distribution", "candidate", "valid", "tps", "p95_ms")} for r in records]
        write_csv(a.output / "metrics.csv", rows, list(rows[0]) if rows else ["trial"])
        (a.output / "summary.json").write_text(json.dumps(result, indent=2))
        (a.output / "checksums.json").write_text(json.dumps({
            f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in a.output.iterdir()
            if f.is_file() and f.suffix in {".json", ".jsonl", ".csv", ".log"} and f.name != "checksums.json"}, indent=2))
        executor.close()
        lock.close()
    print(json.dumps(result, indent=2))
    return 0 if result.get("completed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
