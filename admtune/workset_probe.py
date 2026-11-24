"""隔离大工作集上的范围/排序参数区分度筛查；只读数据、零 LLM。"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time

from dotenv import load_dotenv
import yaml

from config import load_app_config
from .calibration import execute_trial, hardware_context, write_json
from .contracts import Objective, fingerprint
from .dynamic_runtime import GenerationalRequests
from .live import load_space
from .provenance import source_provenance
from .real_execution import AgenticDBExecutor
from .replay import ReplayPlan, planned_request
from .replay_report import read_jsonl
from .security import redact
from tools.mysql_resize import wait_buffer_pool_ready
from tools.prepare_workset_v2 import file_hash, DB


def probe_plan(space, initial):
    if initial["innodb_buffer_pool_size"] != 1024**3 or initial["sort_buffer_size"] != 262144:
        raise ValueError("Frozen probes require 1 GiB pool and 256 KiB sort baseline")
    specs = []
    for block, profile, delta, seed in (
        ("scan", "range-scan", {"innodb_buffer_pool_size": 2 * 1024**3}, 610),
        ("sort", "range-sort", {"sort_buffer_size": 8 * 1024**2}, 611),
    ):
        candidate = space.merge(initial, delta)
        for i in range(5):
            specs.append({
                "name": f"{block}-{i}", "block": block, "query_profile": profile,
                "kind": "probe" if i % 2 else "reference", "rate": 10, "read_percent": 100,
                "seed": seed, "range_rows": 32768, "top_k": 8192 if block == "sort" else 0,
                "config": candidate if i % 2 else dict(initial),
            })
    return specs


def check_preparation(preparation, settings, workload):
    if (not preparation.get("completed") or not preparation.get("snapshot_restore_verified")
            or int(settings["port"]) != 3308 or settings["database"] != DB
            or preparation.get("port") != 3308 or preparation.get("database") != settings["database"]
            or workload.tables != 4 or workload.table_size != 2_000_000):
        raise ValueError("Validated isolated 4x2000000 workset and restore rehearsal required")


def validate_analyze(rows):
    expected = {f"{DB}.sbtest{i}" for i in range(1, 5)}
    if (len(rows) != 4 or {r[0] for r in rows} != expected
            or any(len(r) != 4 or r[1:] != ["analyze", "status", "OK"] for r in rows)):
        raise ValueError("All four dedicated tables must finish statistics refresh")


def run_probe(actuator, app, settings, output, records, preparation):
    preflight = actuator.preflight()
    # Statistics immediately after a bulk load/physical restore can be stale.
    # Refresh once, before all trials; never selectively between treatments.
    analyze = actuator.db.query_rows("ANALYZE TABLE sbtest1,sbtest2,sbtest3,sbtest4;")
    validate_analyze(analyze)
    table_stats = actuator.db.query_rows(
        "SELECT TABLE_NAME,TABLE_ROWS,DATA_LENGTH,INDEX_LENGTH FROM information_schema.tables "
        "WHERE TABLE_SCHEMA=DATABASE() ORDER BY TABLE_NAME;"
    )
    initial = actuator.initial.config
    conditions = actuator.db.show_variables(["innodb_flush_method", "innodb_buffer_pool_instances"])
    if str(conditions["innodb_flush_method"]).upper() != "O_DIRECT" or int(conditions["innodb_buffer_pool_instances"]) != 1:
        raise ValueError("Frozen direct-I/O/one-pool instance conditions changed")
    plan = probe_plan(actuator.space, initial)
    workload = app.benchmark.workload
    client = GenerationalRequests(settings, initial)
    sanity = []
    try:
        client.preflight(ReplayPlan(tables=workload.tables, table_size=workload.table_size))
        samples = []
        with client.connect().cursor() as cursor:
            for table in range(1, 5):
                cursor.execute(f"SELECT id,k,c,pad FROM sbtest{table} WHERE id IN (1,1000000,2000000) ORDER BY id")
                samples.append(cursor.fetchall())
        if fingerprint(samples) != preparation["logical_state"]["samples_sha256"]:
            raise ValueError("Prepared/restored dataset sample identity changed")
        for spec in (plan[0], plan[5]):
            query_plan = ReplayPlan(
                scenario="steady", read_percent=100, query_profile=spec["query_profile"],
                range_rows=spec["range_rows"], top_k=spec["top_k"],
                tables=workload.tables, table_size=workload.table_size,
            )
            request = planned_request(query_plan, 0, 0, "preflight", "read", 1, 1)
            with client.connect().cursor() as cursor:
                cursor.execute("EXPLAIN FORMAT=JSON " + request.sql, request.parameters)
                explain = json.loads(cursor.fetchone()[0])
            started = time.monotonic()
            result = client(request)  # one bounded structural check, not a performance label
            sanity.append({"profile": spec["query_profile"], "explain": explain,
                           "elapsed_seconds": time.monotonic() - started, "result": result})
    finally:
        client.close()
    manifest = {
        "mode": "workset_parameter_screening", "simulated": False, "llm_calls": 0,
        "objective": "p95_latency_ms:minimize", "actuator_benchmark_not_executed": True,
        "preflight": preflight, "space": actuator.space.describe(),
        "memory_bytes": actuator.space.memory_bytes, "fixed_resources": actuator.space.fixed_resources,
        "hardware": hardware_context(), "fixed_instance_conditions": conditions,
        "pre_measurement_analyze": analyze, "table_statistics_estimates_after_analyze": table_stats,
        "preflight_query_checks_not_performance_labels": sanity,
        "preparation_source_revision": preparation["git_revision"],
        "snapshot_manifest_sha256": preparation["snapshot_manifest_sha256"],
        "snapshot_restore_verified_before_run": True, "reset_snapshot_between_trials": False,
        "read_only_queries": True, "warmup_seconds": 15, "measurement_seconds": 60,
        "plan": plan, "max_trials": 10, "max_wall_before_trial_seconds": 1200,
        "is_thesis_evidence": False, **source_provenance(),
    }
    write_json(output / "manifest.json", manifest)
    origin = time.monotonic()
    expected_results = {}
    for spec in plan:
        if len(records) >= 10 or time.monotonic() - origin > 1200:
            raise TimeoutError("Workset screening budget reached")
        write_json(output / "progress.tmp", {"current_trial": spec["name"], "completed_trials": len(records),
                                           "llm_calls": 0, "elapsed_seconds": time.monotonic() - origin})
        (output / "progress.tmp").replace(output / "progress.json")
        record = execute_trial(actuator, settings, spec, spec["config"], output, workload.tables, workload.table_size)
        records.append(record)
        with (output / "trials.jsonl").open("a") as f:
            f.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        if not record["valid"] or not record["constraints_ok"]:
            raise ValueError("Invalid/SLA-violating range probe; do not use it as a performance label")
        block = spec["block"]
        signature = record["query_result_signature"]
        if block in expected_results and signature != expected_results[block]:
            raise ValueError("Identical read-only schedules returned different query results")
        expected_results[block] = signature
    return {"completed": True, "identical_read_only_results_per_block": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("target", "benchmark", "knobs", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--dedicated-instance", action="store_true")
    args = parser.parse_args()
    if not args.dedicated_instance:
        parser.error("Explicit dedicated-instance authorization required")
    load_dotenv(args.env_file, override=True)
    root = Path(__file__).resolve().parent.parent
    app = load_app_config(project_root=root, target_path=args.target.resolve(),
                          benchmark_path=args.benchmark.resolve(), knobs_path=args.knobs.resolve(),
                          dry_run_override=False)
    settings = yaml.safe_load(args.target.read_text())["mysql"]
    if settings["host"] not in {"127.0.0.1", "localhost"}:
        parser.error("Workset probes are local-only")
    config_path = Path(settings["config_path"])
    if not config_path.is_absolute():
        parser.error("Absolute dedicated config path required")
    preparation = json.loads((config_path.parent / "preparation.json").read_text())
    check_preparation(preparation, settings, app.benchmark.workload)
    space = load_space(args.knobs, memory_bytes=app.target.hardware_memory_bytes)
    if any(k.restart_required for k in space.knobs.values()):
        parser.error("No restart knobs allowed during screening")
    fd = os.open(str(config_path) + ".admtune.lock", os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0), 0o600)
    lock = os.fdopen(fd, "w")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        lock.close()
        parser.error("Instance already locked")
    records, actuator = [], None
    result = {"completed": False, "llm_calls": 0, "is_thesis_evidence": False}
    try:
        args.output.mkdir(parents=True, mode=0o700, exist_ok=False)
        actuator = AgenticDBExecutor(app, space, Objective(), args.output, allow_mutations=True, local=True)
        result.update(run_probe(actuator, app, settings, args.output, records, preparation))
    except Exception as exc:
        result.update(completed=False, error=redact(str(exc)))
    finally:
        if actuator is not None and actuator.initial is not None:
            try:
                wait_buffer_pool_ready(actuator.db, actuator.readback()["innodb_buffer_pool_size"])
                actuator.restore_initial()
                result["final_buffer_pool_ready"] = wait_buffer_pool_ready(
                    actuator.db, actuator.initial.config["innodb_buffer_pool_size"])
                actual = actuator.connector.read_file(str(config_path), sudo=True)
                result["final_config_sha256"] = hashlib.sha256(actual.encode()).hexdigest()
                result["initial_configuration_restored"] = actual == actuator.initial.file_content
            except Exception as exc:
                result.update(initial_configuration_restored=False, recovery_error=redact(str(exc)))
        if actuator is not None:
            actuator.close()
        lock.close()
    result.update(trials=len(records), valid_trials=sum(r["valid"] for r in records),
                  request_counts={k: sum(r["stream"]["counts"][k] for r in records)
                                  for k in ("offered", "submitted", "succeeded", "failed", "dropped")})
    if actuator is not None:
        write_json(args.output / "summary.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["completed"] and result.get("initial_configuration_restored") else 1


if __name__ == "__main__":
    raise SystemExit(main())
