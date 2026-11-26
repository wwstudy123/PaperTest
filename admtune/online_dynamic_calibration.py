"""只读动态实验的六次零LLM校准；不把校准性能标签导入后续搜索。"""
import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import time

from dotenv import load_dotenv
from config import load_app_config
from .calibration import execute_trial
from .continuous import atomic_json
from .contracts import Objective
from .dynamic import instance_lock
from .live import load_space
from .provenance import source_provenance
from .real_execution import AgenticDBExecutor
from .dynamic_runtime import GenerationalRequests
from .replay import ReplayPlan, replay
from .replay_report import read_jsonl
from .contracts import fingerprint
from .security import redact


def plan(space, initial):
    result = []
    for block, profile, delta, seed in [
        ("scan", "range-scan", {"innodb_buffer_pool_size": 2*1024**3}, 710),
        ("sort", "range-sort", {"sort_buffer_size": 8*1024**2}, 711)]:
        for index in range(3):
            result.append({"name": f"{block}-{index}", "block": block,
                "kind": "candidate" if index == 1 else "reference", "query_profile": profile,
                "rate": 10, "read_percent": 100, "seed": seed, "range_rows": 32768,
                "top_k": 8192 if block == "sort" else 0,
                "config": space.merge(initial, delta) if index == 1 else dict(initial)})
    return result


def evaluate(records):
    metrics, sse, df = {}, 0., 0
    valid = len(records) == 6 and all(r["valid"] for r in records)
    if not valid:
        return {"passed": False, "reason": "Incomplete six real measurements", "llm_calls": 0}
    for block in ("scan", "sort"):
        rows = [r for r in records if r["block"] == block]
        left, candidate, right = [r["value_p95_ms"] for r in rows]
        metrics[block] = {"reference_left_ms": left, "candidate_ms": candidate,
                         "reference_right_ms": right, "gain": 1-candidate/((left+right)/2),
                         "reference_drift": abs(right/left-1)}
        sse += (math.log(left)-math.log(right))**2/2
        df += 1
    estimated = math.sqrt(sse/df)
    return {"passed": all(m["gain"] > .05 and m["reference_drift"] <= .20 for m in metrics.values()),
            "metrics": metrics, "gate": {"min_gain": .05, "max_reference_drift": .20},
            "repeat_degrees_of_freedom": df, "estimated_log_std": estimated,
            "fixed_noise_log_std": max(.02, estimated),
            "noise_note": "Global pilot floor: max(predeclared 2% log-scale minimum, paired-reference estimate). "
                          "Two degrees of freedom only, not established general calibration.",
            "llm_calls": 0}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("target", "benchmark", "knobs", "output"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--dedicated-instance", action="store_true")
    p.add_argument("--warm-recheck-source", type=Path,
                   help="独立暖态补测：180秒固定预热，只新增3次扫描；保留来源排序3次和冷态失败记录")
    a = p.parse_args()
    if not a.dedicated_instance:
        p.error("Dedicated database required")
    root = Path(__file__).resolve().parents[1]; load_dotenv(root/".env", override=True)
    app = load_app_config(project_root=root, target_path=a.target, benchmark_path=a.benchmark,
                          knobs_path=a.knobs, dry_run_override=False)
    lock = instance_lock(app.target.mysql.config_path)
    a.output.mkdir(mode=0o700, parents=True, exist_ok=False)
    space = load_space(a.knobs, memory_bytes=app.target.hardware_memory_bytes)
    ex = AgenticDBExecutor(app, space, Objective(), a.output, allow_mutations=True, local=True)
    records, summary = [], {"completed": False, "llm_calls": 0}
    try:
        pre = ex.preflight()
        specs = plan(space, pre["current_config"])
        reused = []
        if a.warm_recheck_source:
            old = read_jsonl(a.warm_recheck_source/"trials.jsonl")
            reused = [r for r in old if r["block"] == "sort"]
            if len(reused) != 3 or any(not r["valid"] for r in reused):
                raise ValueError("Three real valid sort measurements required for warm-only revision")
            specs = [s for s in specs if s["block"] == "scan"]
        atomic_json(a.output/"manifest.json", {"mode": "online_dynamic_zero_llm_calibration",
            "plan": specs, "warmup_seconds": 15, "measurement_seconds": 60,
            "gate": {"min_gain": .05, "max_reference_drift": .20}, "noise_minimum": .02,
            "preflight": pre, "simulated": False,
            "warm_recheck_source": str(a.warm_recheck_source) if a.warm_recheck_source else None,
            "reused_sort_records_fingerprint": fingerprint(reused) if reused else None,
            "new_measurements_planned": len(specs), **source_provenance()})
        started = time.monotonic()
        if a.warm_recheck_source:
            wp = ReplayPlan(scenario="steady", phase_seconds=60, table_size=app.benchmark.workload.table_size,
                tables=app.benchmark.workload.tables, read_percent=100, query_profile="range-mixed",
                range_rows=32768, top_k=8192, seed=712)
            client = GenerationalRequests(asdict(ex.settings), pre["current_config"])
            warm = a.output/"warmup"; warm.mkdir()
            try:
                warmed = replay(wp, client, warm, simulated=False)
                atomic_json(warm/"summary.json", warmed)
                if not warmed["complete"]:
                    raise RuntimeError("Common neutral 180-second warmup incomplete")
            finally:
                client.close()
        for spec in specs:
            if time.monotonic()-started > 900:
                raise TimeoutError("No new trial after 900 seconds")
            r = execute_trial(ex, asdict(ex.settings), spec, spec["config"], a.output,
                              app.benchmark.workload.tables, app.benchmark.workload.table_size)
            records.append(r)
            with (a.output/"trials.jsonl").open("a") as f:
                f.write(json.dumps(r, allow_nan=False)+"\n")
            atomic_json(a.output/"progress.json", {"completed_trials": len(records), "last_p95_ms": r["value_p95_ms"]})
            if not r["valid"]:
                raise RuntimeError("Invalid calibration stream")
        for r in reused:
            r = {**r, "reused_from": str(a.warm_recheck_source)}
            records.append(r)
            with (a.output/"trials.jsonl").open("a") as f:
                f.write(json.dumps(r, allow_nan=False)+"\n")
        summary.update(completed=True, result=evaluate(records),
                       new_measurements=len(specs), reused_measurements=len(reused))
    except Exception as exc:
        summary["error"] = redact(str(exc))
    finally:
        try:
            ex.restore_initial(); summary["initial_configuration_restored"] = ex.initial is not None
        except Exception as exc:
            summary.update(completed=False, recovery_error=redact(str(exc)))
        ex.close(); lock.close(); atomic_json(a.output/"summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["completed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
