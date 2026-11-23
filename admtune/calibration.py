"""专用 MySQL 零 LLM 校准：重复、增压及交错基准的参数筛查；非正式对比。"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import yaml
from dotenv import load_dotenv

from config import load_app_config
from .contracts import Objective, fingerprint
from .dynamic_runtime import GenerationalRequests
from .live import load_space
from .provenance import source_provenance
from .real_execution import AgenticDBExecutor
from .replay import ReplayPlan, replay
from .replay_report import read_jsonl, export_replay
from .security import redact
from tools.apply_db_config import apply_db_config
from tools.mysql_resize import wait_buffer_pool_ready
from tools.rollback import create_config_backup


def baseline_plan():
    """Six repeats, then six ascending pressure probes; all predeclared."""
    return [
        {"name": f"repeat-{read}-{rep}", "kind": "repeat", "rate": 10,
         "read_percent": read, "seed": 42 + rep}
        for rep in range(3) for read in (90, 10)
    ] + [
        {"name": f"pressure-{read}-{rate}", "kind": "pressure", "rate": rate,
         "read_percent": read, "seed": 42}
        for rate in (100, 500, 1000) for read in (90, 10)
    ]


def sensitivity_plan(space, initial, rate, parameters=None):
    """Four single-knob probes, each bracketed by a reference, one block only."""
    deltas = [
        {"innodb_buffer_pool_size": 2 * 1024**3},
        {"innodb_io_capacity": 1600},
        {"innodb_io_capacity_max": 6000},
        {"sort_buffer_size": 8 * 1024**2},
    ]
    names = {next(iter(d)) for d in deltas}
    if parameters is not None:
        if not parameters or len(parameters) != len(set(parameters)) or set(parameters) - names:
            raise ValueError("Explicit unique known sensitivity parameters required")
        deltas = [d for d in deltas if next(iter(d)) in parameters]
    configs = [space.merge(initial, delta) for delta in deltas]
    if any(sum(c[n] != initial[n] for n in initial) != 1 for c in configs):
        raise ValueError("Each predeclared probe must change exactly one parameter")
    random.Random(20261124).shuffle(configs)
    specs = []
    for i in range(2 * len(configs) + 1):
        config = dict(initial) if i % 2 == 0 else configs[i // 2]
        specs.append({"name": f"sensitivity-{i}", "kind": "reference" if i % 2 == 0 else "probe",
                      "rate": rate, "read_percent": 10, "seed": 600, "config": config})
    return specs


def completed_sensitivity_parameters(records):
    completed = set()
    for left, candidate, right in zip(records, records[1:], records[2:]):
        if [r["kind"] for r in (left, candidate, right)] != ["reference", "probe", "reference"]:
            continue
        if not all(r["valid"] and r["value_p95_ms"] is not None for r in (left, candidate, right)):
            continue
        if left["config"] != right["config"] or any(
            len({r[k] for r in (left, candidate, right)}) != 1 for k in ("rate", "read_percent", "seed")
        ):
            raise ValueError("Mismatched sensitivity reference block")
        changed = [k for k in left["config"] if left["config"][k] != candidate["config"][k]]
        if len(changed) != 1:
            raise ValueError("Sensitivity block did not isolate one knob")
        completed.add(changed[0])
    return completed


def supplement_plan(parent, preflight, hardware):
    manifest = json.loads((parent / "manifest.json").read_text())
    summary = json.loads((parent / "summary.json").read_text())
    records = read_jsonl(parent / "trials.jsonl")
    if (manifest.get("mode") != "zero_llm_calibration" or manifest.get("git_dirty") is not False
            or summary.get("completed") or not summary.get("initial_configuration_restored")
            or summary.get("error") != "Calibration budget reached; no new trial"
            or summary.get("llm_calls") != 0):
        raise ValueError("Supplement only a restored original run stopped by its wall budget")
    if (manifest["preflight"]["space_key"] != preflight["space_key"]
            or manifest["preflight"]["current_config"] != preflight["current_config"]
            or summary["final_config_sha256"] != preflight["config_sha256"]
            or manifest["preflight"]["readiness"]["live_schema_hash"] != preflight["readiness"]["live_schema_hash"]
            or manifest["hardware_readonly"] != hardware):
        raise ValueError("Parent and supplement environments are not compatible")
    done = completed_sensitivity_parameters(records)
    names = {"innodb_buffer_pool_size", "innodb_io_capacity", "innodb_io_capacity_max", "sort_buffer_size"}
    missing = sorted(names - done)
    if not missing:
        raise ValueError("No sensitivity parameters remain; do not repeat paid or free work")
    audit = {"parent_run": str(parent), "completed_parameters": sorted(done),
             "remaining_parameters": missing, "parent_new_measurements": len(records),
             "parent_sha256": {name: hashlib.sha256((parent / name).read_bytes()).hexdigest()
                               for name in ("manifest.json", "summary.json", "trials.jsonl")}}
    return missing, choose_rate(records, 10), audit


def measurement_stats(requests, plan, config, warmup=15, measurement=60):
    cohort = [r for r in requests if warmup <= r["scheduled"] < warmup + measurement]
    expected = plan.base_rate * measurement
    good = [r for r in cohort if r.get("ok") and r.get("rows") == 1]
    effective = len(good) == expected and all(
        r.get("config_generation") == 0 and not r.get("config_transition", True)
        and r.get("session_config_hash") == fingerprint(config) for r in good
    )
    complete = (len(cohort) == expected and len({r["seq"] for r in cohort}) == expected
                and all(r["emit_lag_seconds"] <= plan.max_emit_lag_seconds for r in cohort))
    def percentile(field):
        return float(np.percentile(field, 95)) if field else None
    observed = percentile([1000 * (r["completed"] - r["emitted"]) for r in good])
    return {
        "measurement_start": warmup, "measurement_end": warmup + measurement,
        "expected_requests": expected, "completed_cohort_requests": len(cohort),
        "complete_cohort": complete, "effective": effective,
        "observed_p95_ms": observed,
        "value_p95_ms": observed if complete and effective else None,
        "queue_p95_ms": percentile([1000 * (r["started"] - r["emitted"]) for r in good]),
        "service_p95_ms": percentile([1000 * (r["completed"] - r["started"]) for r in good]),
        "constraints_ok": observed is not None and observed <= 1000,
    }


def choose_rate(records, read_percent):
    eligible = [r["rate"] for r in records
                if r["read_percent"] == read_percent and r["valid"] and r["constraints_ok"]
                and r["queue_p95_ms"] <= r["service_p95_ms"]]
    if not eligible:
        raise ValueError("No valid reference rate without dominant client queueing")
    return max(eligible)


def process_sample(pid):
    # Only process-local accounting; host-wide disk counters would include neighbours.
    text = (Path("/proc") / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()
    io = {}
    for line in (Path("/proc") / str(pid) / "io").read_text().splitlines():
        name, value = line.split(":")
        io[name] = int(value)
    return {"at": time.monotonic(), "cpu_seconds": (int(text[11]) + int(text[12])) / os.sysconf("SC_CLK_TCK"),
            "start_ticks": int(text[19]), "read_bytes": io["read_bytes"], "write_bytes": io["write_bytes"]}


def status_sample(db):
    names = ["Innodb_buffer_pool_reads", "Innodb_buffer_pool_read_requests",
             "Innodb_data_reads", "Innodb_data_writes", "Innodb_os_log_written",
             "Sort_rows", "Created_tmp_disk_tables"]
    return {k: int(v) for k, v in db.show_status(names).items() if v is not None}


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    path.chmod(0o600)


def hardware_context():
    paths = [
        "/sys/fs/cgroup/cpu.max", "/sys/fs/cgroup/memory.max",
        "/sys/fs/cgroup/cpu/cpu.cfs_quota_us", "/sys/fs/cgroup/cpu/cpu.cfs_period_us",
        "/sys/fs/cgroup/cpu,cpuacct/cpu.cfs_quota_us",
        "/sys/fs/cgroup/cpu,cpuacct/cpu.cfs_period_us",
        "/sys/fs/cgroup/memory/memory.limit_in_bytes",
    ]
    limits = {p: Path(p).read_text().strip() for p in paths if Path(p).is_file()}
    model = next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                  if line.startswith("model name")), None)
    return {"cpu_model": model, "visible_cpu_count": os.cpu_count(), "cgroup_limits": limits}


def prepare(actuator, config):
    before = actuator.snapshot()
    first = wait_buffer_pool_ready(actuator.db, before.config["innodb_buffer_pool_size"])
    actuator.counter += 1
    create_config_backup(actuator.connector, config_path=actuator.settings.config_path,
                         round_id=actuator.counter, run_dir=actuator.run_dir)
    delta = {k: v for k, v in config.items() if before.config[k] != v}
    if delta:
        apply_db_config(actuator.connector, actuator.db, config_path=actuator.settings.config_path,
                        candidate_config=delta, knob_specs=actuator.specs)
    actuator._verify(config, before.invariants)
    last = wait_buffer_pool_ready(actuator.db, config["innodb_buffer_pool_size"])
    return [first, last]


def execute_trial(actuator, settings, spec, config, output, tables, table_size):
    plan = ReplayPlan(scenario="steady", phase_seconds=28, base_rate=spec["rate"],
                      high_rate=spec["rate"] + 10, read_percent=spec["read_percent"],
                      tables=tables, table_size=table_size, seed=spec["seed"])
    folder = output / spec["name"]
    folder.mkdir(mode=0o700, exist_ok=False)
    resize_checks = prepare(actuator, config)  # no live request workers during a change
    pid = int(Path(actuator.settings.pid_file).read_text().strip())
    before_status = status_sample(actuator.db)
    before_process = process_sample(pid)
    client = GenerationalRequests(settings, config)
    try:
        stream = replay(plan, client, folder, simulated=False)
    finally:
        # replay exits its thread pool before unwinding; close sessions between
        # trials rather than accumulating idle sessions on the server.
        client.close()
    after_process = process_sample(pid)
    after_status = status_sample(actuator.db)
    if after_process["start_ticks"] != before_process["start_ticks"]:
        raise ValueError("MySQL process changed during trial")
    actuator._verify(config, actuator.initial.invariants)
    metrics = measurement_stats(read_jsonl(folder / "requests.jsonl"), plan, config)
    if not stream["complete"]:
        metrics["value_p95_ms"] = None
    elapsed = after_process["at"] - before_process["at"]
    delta = {k: after_status[k] - before_status[k] for k in before_status}
    if any(v < 0 for v in delta.values()):
        raise ValueError("Database counters reset during trial")
    record = {
        **spec, **metrics, "config": config, "directory": spec["name"],
        "valid": stream["complete"] and metrics["complete_cohort"] and metrics["effective"],
        "plan": asdict(plan), "stream": stream,
        "buffer_pool_ready_checks": resize_checks,
        "process_metrics_full_stream": {
            "wall_seconds": elapsed,
            "cpu_core_equivalents": (after_process["cpu_seconds"] - before_process["cpu_seconds"]) / elapsed,
            "read_bytes": after_process["read_bytes"] - before_process["read_bytes"],
            "write_bytes": after_process["write_bytes"] - before_process["write_bytes"],
        },
        "database_counters_full_stream": delta,
        "llm_calls": 0, "simulated": False,
    }
    write_json(folder / "summary.json", {**stream, "mode": "zero_llm_calibration", "calibration": record})
    export_replay(folder)
    return record


def run_calibration(actuator, settings, app, output, complete_from=None):
    records, skipped = [], []
    started = time.monotonic()
    preflight = actuator.preflight()
    initial = dict(actuator.initial.config)
    hardware = hardware_context()
    parameters, explicit_rate, parent_audit = None, None, None
    if complete_from is not None:
        parameters, explicit_rate, parent_audit = supplement_plan(complete_from.resolve(), preflight, hardware)
    max_trials = 21 if parameters is None else 2 * len(parameters) + 1
    # Abort before measurements if this particular screening protocol is not applicable.
    sensitivity_plan(actuator.space, initial, 10, parameters)
    workload = app.benchmark.workload
    client = GenerationalRequests(settings, initial)
    try:
        checked = client.preflight(ReplayPlan(tables=workload.tables, table_size=workload.table_size))
    finally:
        client.close()
    tables = actuator.db.query_rows(
        "SELECT TABLE_NAME,TABLE_ROWS,DATA_LENGTH,INDEX_LENGTH "
        "FROM information_schema.tables WHERE TABLE_SCHEMA=DATABASE() ORDER BY TABLE_NAME;"
    )
    variables = actuator.db.show_variables(["innodb_flush_method", "innodb_buffer_pool_size",
                                           "innodb_page_size", "innodb_buffer_pool_instances"])
    manifest = {
        "mode": "zero_llm_calibration", "preflight": preflight, "data_preflight": checked,
        "space": actuator.space.describe(), "memory_bytes": actuator.space.memory_bytes,
        "fixed_resources": actuator.space.fixed_resources,
        "table_statistics_estimates": tables, "database_variables": variables,
        "hardware_readonly": hardware,
        "warmup_seconds": 15, "measurement_seconds": 60, "stream_seconds": 84,
        "max_trials": max_trials, "max_wall_before_new_trial_seconds": 2100,
        "stage": "full" if parameters is None else "supplement_missing_sensitivity",
        "parent_evidence": parent_audit,
        "baseline_plan": baseline_plan() if parameters is None else [],
        "sensitivity_parameters": parameters, "sensitivity_order_seed": 20261124,
        "request_semantics": "Point SELECT/UPDATE, autocommit, no added Sysbench",
        "data_snapshot_restored": False, "llm_calls": 0, **source_provenance(),
    }
    write_json(output / "manifest.json", manifest)
    def measure(spec, config):
        if len(records) >= max_trials or time.monotonic() - started > 2100:
            raise TimeoutError("Calibration budget reached; no new trial")
        temp = output / "progress.tmp"
        write_json(temp, {"current_trial": spec, "completed_trials": len(records), "llm_calls": 0,
                          "elapsed_seconds": time.monotonic() - started})
        temp.replace(output / "progress.json")
        record = execute_trial(actuator, settings, spec, config, output, workload.tables, workload.table_size)
        records.append(record)
        with (output / "trials.jsonl").open("a") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        return record
    blocked = set()
    for spec in baseline_plan() if parameters is None else []:
        if spec["read_percent"] in blocked:
            skipped.append(spec)
            continue
        record = measure(spec, initial)
        if not record["valid"] or not record["constraints_ok"]:
            if spec["kind"] == "repeat":
                raise ValueError("Low-rate repeated baseline invalid; stop calibration")
            blocked.add(spec["read_percent"])  # do not escalate an already overloaded mix
    rate = choose_rate(records, 10) if explicit_rate is None else explicit_rate
    for spec in sensitivity_plan(actuator.space, initial, rate, parameters):
        record = measure(spec, spec["config"])
        if not record["valid"] or not record["constraints_ok"]:
            raise ValueError("Sensitivity measurement invalid; restore and stop")
    return {"completed": True, "selected_write_heavy_rate": rate, "trials": len(records),
            "stage": manifest["stage"], "parent_evidence": parent_audit,
            "skipped_pressure_probes": skipped, "invalid_trials": sum(not r["valid"] for r in records)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("target", "benchmark", "knobs", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--dedicated-instance", action="store_true")
    parser.add_argument("--complete-from", type=Path,
                        help="只补齐已恢复且因原时间预算停止的校准，另存新目录")
    args = parser.parse_args()
    if not args.dedicated_instance:
        parser.error("Requires dedicated instance; writes data and temporarily changes DB knobs")
    load_dotenv(args.env_file, override=True)
    app = load_app_config(project_root=Path(__file__).resolve().parent.parent,
                          target_path=args.target.resolve(), benchmark_path=args.benchmark.resolve(),
                          knobs_path=args.knobs.resolve(), dry_run_override=False)
    settings = yaml.safe_load(args.target.read_text())["mysql"]
    if settings["host"] not in {"127.0.0.1", "localhost"}:
        parser.error("Calibration is local-only")
    space = load_space(args.knobs, memory_bytes=app.target.hardware_memory_bytes)
    if any(k.restart_required for k in space.knobs.values()):
        parser.error("No restart-required knobs allowed")
    config_path = app.target.mysql.config_path
    if not Path(config_path).is_absolute():
        parser.error("An absolute dedicated config path is required")
    fd = os.open(config_path + ".admtune.lock", os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0), 0o600)
    lock = os.fdopen(fd, "w")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        lock.close()
        parser.error("Another mutator holds the instance lock")
    actuator = None
    result = {"completed": False, "llm_calls": 0, "is_thesis_evidence": False,
              "data_snapshot_restored": False}
    try:
        args.output.mkdir(mode=0o700, parents=True, exist_ok=False)
        actuator = AgenticDBExecutor(app, space, Objective(), args.output, allow_mutations=True, local=True)
        result.update(run_calibration(actuator, settings, app, args.output, complete_from=args.complete_from))
    except Exception as exc:
        result.update(completed=False, error=redact(str(exc)))
    finally:
        if actuator is not None and actuator.initial is not None:
            try:
                wait_buffer_pool_ready(actuator.db, actuator.readback()["innodb_buffer_pool_size"])
                actuator.restore_initial()
                result["final_buffer_pool_ready"] = wait_buffer_pool_ready(
                    actuator.db, actuator.initial.config["innodb_buffer_pool_size"])
                restored = actuator.connector.read_file(config_path, sudo=True)
                result["final_config_sha256"] = hashlib.sha256(restored.encode()).hexdigest()
                result["initial_configuration_restored"] = restored == actuator.initial.file_content
            except Exception as exc:
                result.update(initial_configuration_restored=False, recovery_error=redact(str(exc)))
        if actuator is not None:
            actuator.close()
        lock.close()
    if actuator is not None and args.output.exists():
        write_json(args.output / "summary.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["completed"] and result.get("initial_configuration_restored") else 1


if __name__ == "__main__":
    raise SystemExit(main())
