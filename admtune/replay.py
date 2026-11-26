"""固定外生到达计划的真实 MySQL 回放；仅验收监测，不执行参数调优。"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import socket
import threading
import time

from dotenv import load_dotenv
import yaml

from .contracts import fingerprint
from .provenance import source_provenance
from .security import redact
from .workload import PhaseDetector, TemplateCatalog, Window, WorkloadEncoder


@dataclass(frozen=True)
class ReplayPlan:
    scenario: str = "composition"
    phase_seconds: int = 24
    window_seconds: int = 4
    slices: int = 2
    base_rate: int = 10
    high_rate: int = 30
    tables: int = 4
    table_size: int = 100000
    workers: int = 4
    queue_size: int = 32
    seed: int = 42
    max_emit_lag_seconds: float = 0.5
    read_percent: int = 50
    query_profile: str = "point"
    range_rows: int = 0
    top_k: int = 0
    steady_scan_percent: int = 50

    def __post_init__(self):
        if self.scenario not in {"composition", "intensity", "steady", "transient"}:
            raise ValueError("Unknown replay scenario")
        for name in ["phase_seconds", "window_seconds", "slices", "base_rate", "high_rate", "tables", "table_size", "workers", "queue_size"]:
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"Positive integer required: {name}")
        if self.phase_seconds % self.window_seconds or self.phase_seconds < 6 * self.window_seconds:
            raise ValueError("Each phase needs at least six complete windows")
        if self.window_seconds % self.slices or self.base_rate % 10 or self.high_rate % 10:
            raise ValueError("Integral slices and rates divisible by ten required")
        if self.high_rate <= self.base_rate:
            raise ValueError("High rate must exceed base rate")
        if type(self.read_percent) is not int or self.read_percent not in range(0, 101, 10):
            raise ValueError("Read percentage must be an integer multiple of ten")
        mixed = self.query_profile == "range-mixed"
        if (type(self.steady_scan_percent) is not int or self.steady_scan_percent not in range(0, 101, 10)
                or (self.steady_scan_percent != 50 and not (mixed and self.scenario == "steady"))):
            raise ValueError("Explicit scan mixture is only for read-only steady range-mixed streams")
        if self.scenario != "steady" and self.read_percent != 50 and not mixed:
            raise ValueError("Explicit read percentage is supported only by steady replay")
        if self.query_profile not in {"point", "range-scan", "range-sort", "range-mixed"}:
            raise ValueError("Unknown query profile")
        if type(self.range_rows) is not int or type(self.top_k) is not int:
            raise ValueError("Integer range and limit settings required")
        if self.query_profile == "point":
            if self.range_rows != 0 or self.top_k != 0:
                raise ValueError("Point replay must not silently use range settings")
        else:
            allowed = {"composition", "steady"} if mixed else {"steady"}
            if self.scenario not in allowed or self.read_percent != 100:
                raise ValueError("Range streams require an explicit read-only supported scenario")
            if type(self.range_rows) is not int or not 1 <= self.range_rows <= self.table_size:
                raise ValueError("Range must fit the declared contiguous dataset")
            if (type(self.top_k) is not int or
                    (self.query_profile == "range-scan" and self.top_k != 0) or
                    (self.query_profile in {"range-sort", "range-mixed"} and not 0 < self.top_k < self.range_rows)):
                raise ValueError("Explicit valid top-K required for sorting probes")
        if not math.isfinite(self.max_emit_lag_seconds) or self.max_emit_lag_seconds <= 0:
            raise ValueError("Finite positive emission lag bound required")
        if self.duration * self.high_rate > 100000:
            raise ValueError("Replay exceeds smoke request cap")

    @property
    def duration(self):
        return 3 * self.phase_seconds

    @property
    def expected_phase_count(self):
        return 3 if self.scenario in {"composition", "intensity"} else 1


@dataclass(frozen=True)
class Request:
    seq: int
    scheduled: float
    phase: str
    operation: str
    table: int
    row: int

    @property
    def sql(self):
        if self.operation == "read":
            return f"SELECT c FROM sbtest{self.table} WHERE id=%s"
        return f"UPDATE sbtest{self.table} SET k=k+1 WHERE id=%s"

    @property
    def template_sql(self):
        return self.sql.replace("%s", "1")

    @property
    def parameters(self):
        return (self.row,)


@dataclass(frozen=True)
class RangeRequest(Request):
    range_rows: int
    top_k: int = 0

    def __post_init__(self):
        if (self.operation != "read" or type(self.range_rows) is not int or self.range_rows < 1
                or type(self.top_k) is not int or not 0 <= self.top_k < self.range_rows):
            raise ValueError("Invalid read-only range request")

    @property
    def sql(self):
        if not self.top_k:
            return (f"SELECT COUNT(*),SUM(k),SUM(OCTET_LENGTH(c)) FROM sbtest{self.table} "
                    "WHERE id BETWEEN %s AND %s")
        # SUM(k) depends on which rows survive LIMIT: ORDER BY cannot simply be
        # discarded as it could be for a derived table consumed only by COUNT.
        return (f"SELECT COUNT(*),SUM(k) FROM (SELECT k FROM sbtest{self.table} "
                f"WHERE id BETWEEN %s AND %s ORDER BY c,id LIMIT {self.top_k}) AS chosen")

    @property
    def parameters(self):
        return (self.row, self.row + self.range_rows - 1)

    def validate_result(self, rows):
        count = self.top_k or self.range_rows
        columns = 2 if self.top_k else 3
        if (len(rows) != 1 or len(rows[0]) != columns or rows[0][0] != count
                or any(value is None for value in rows[0])):
            raise ValueError("Incomplete range aggregate or unexpected result shape")


def planned_request(plan, seq, scheduled, phase, operation, table, row):
    if operation == "read" and plan.query_profile != "point":
        return RangeRequest(seq, scheduled, phase, operation, table, row, plan.range_rows, plan.top_k)
    return Request(seq, scheduled, phase, operation, table, row)


def declared_requests(plan):
    """预声明模板全集；混合只读负载共享扫描/排序通道，不读取未来计划。"""
    if plan.query_profile == "range-mixed":
        return [RangeRequest(0, 0, "", "read", table, 1, plan.range_rows, limit)
                for limit in (0, plan.top_k) for table in range(1, plan.tables + 1)]
    return [planned_request(plan, 0, 0, "", operation, table, 1)
            for operation in ("read", "write") for table in range(1, plan.tables + 1)]


def query_semantics(plan):
    if plan.query_profile == "point":
        return "Primary-key point SELECT c / UPDATE k=k+1, autocommit; no joins or sorts."
    return (f"Read-only range aggregates, R={plan.range_rows}; "
            f"profile={plan.query_profile}, top-K={plan.top_k}; sorting uses ORDER BY c,id. "
            "One autocommit SQL per request; QPS is not Sysbench transaction TPS.")


def make_schedule(plan: ReplayPlan) -> list[Request]:
    rng = random.Random(plan.seed)
    requests = []
    for second in range(plan.duration):
        phase_index = second // plan.phase_seconds
        if plan.query_profile == "range-mixed":
            scans = (plan.base_rate * (1 if phase_index == 1 else 9) // 10
                     if plan.scenario == "composition" else plan.base_rate * plan.steady_scan_percent // 100)
            limits = [0] * scans + [plan.top_k] * (plan.base_rate - scans)
            rng.shuffle(limits)
            for i, limit in enumerate(limits):
                label = "ABA"[phase_index] if plan.expected_phase_count == 3 else "A"
                requests.append(RangeRequest(
                    len(requests), second + i / plan.base_rate, label, "read",
                    rng.randint(1, plan.tables), rng.randint(1, plan.table_size - plan.range_rows + 1),
                    plan.range_rows, limit))
            continue
        transient = plan.scenario == "transient" and second in {plan.phase_seconds, 2 * plan.phase_seconds}
        high = (plan.scenario == "intensity" and phase_index == 1) or transient
        rate = plan.high_rate if high else plan.base_rate
        reads = (rate * (1 if phase_index == 1 else 9) // 10 if plan.scenario == "composition"
                 else rate * plan.read_percent // 100 if plan.scenario == "steady" else rate // 2)
        operations = ["read"] * reads + ["write"] * (rate - reads)
        rng.shuffle(operations)
        for i, operation in enumerate(operations):
            label = "ABA"[phase_index] if plan.expected_phase_count == 3 else ("B" if transient else "A")
            maximum = (plan.table_size - plan.range_rows + 1
                       if operation == "read" and plan.query_profile != "point" else plan.table_size)
            requests.append(planned_request(plan, len(requests), second + i / rate, label, operation,
                                            rng.randint(1, plan.tables), rng.randint(1, maximum)))
    return requests


def monitor(plan: ReplayPlan):
    # 模板全集来自预声明操作，不能扫描未来测试阶段来发现模板。
    universe = declared_requests(plan)
    catalog = TemplateCatalog([r.template_sql for r in universe], min_samples=1)
    channels = catalog.cluster_count + 1
    # 独立的预声明中性校准窗口；不使用测试阶段观测拟合归一化器或阈值。
    counts = [[0] * channels for _ in range(plan.slices)]
    slice_seconds = plan.window_seconds / plan.slices
    for request in (r for r in universe if r.table == 1):
        ch = catalog.channel(request.template_sql)
        for row in counts:
            row[ch] += plan.base_rate * slice_seconds / 2
    calibration = Window(0, plan.window_seconds, counts)
    encoder = WorkloadEncoder([calibration])
    detector = PhaseDetector(encoder, change_threshold=3, stable_threshold=1,
                             confirm_windows=2, stable_windows=2)
    return catalog, detector, {"encoder": encoder.metadata(), "calibration": asdict(calibration),
                               "template_channels": catalog.mapping,
                               "unknown_channel": catalog.unknown_channel,
                               "change_threshold": 3, "stable_threshold": 1,
                               "confirm_windows": 2, "stable_windows": 2,
                               "note": "预声明联调阈值，非正式校准结果；不读取阶段标签"}


def request_channels(plan, catalog):
    """Encode the predeclared query universe once, never per scheduled request."""
    return {
        query: catalog.channel(query)
        for query in (r.template_sql for r in declared_requests(plan))
    }


def replay(plan: ReplayPlan, execute, output: Path, *, simulated: bool = True, guard=None, observer=None) -> dict:
    """execute 只由有界工作线程调用；调度不等待查询完成，过载明确丢弃。"""
    schedule = make_schedule(plan)
    catalog, detector, metadata = monitor(plan)
    channel_by_query = request_channels(plan, catalog)
    (output / "schedule.json").write_text(json.dumps([asdict(r) for r in schedule]))
    (output / "monitor.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    replay_started_at = datetime.now(timezone.utc).isoformat()
    started = time.monotonic()
    if observer is not None:
        observer.on_start(started)
    pending = {}
    counts = [[0] * (catalog.cluster_count + 1) for _ in range(plan.slices)]
    window_index = 0
    totals = dict(offered=0, submitted=0, dropped=0, succeeded=0, failed=0)
    max_lag = 0.0
    ready = []
    event_counts = {}
    def work(request):
        begin = time.monotonic() - started
        try:
            result = execute(request)
            details = result if isinstance(result, dict) else {"rows": result}
            return {"seq": request.seq, "started": begin, "completed": time.monotonic() - started,
                    "ok": True, **details}
        except Exception as exc:
            return {"seq": request.seq, "started": begin, "completed": time.monotonic() - started,
                    "ok": False, "error_type": type(exc).__name__}
    with (output / "offered.jsonl").open("w") as offered, (output / "requests.jsonl").open("w") as results, (output / "windows.jsonl").open("w") as windows:
        def collect():
            for future in list(pending):
                if future.done():
                    record = future.result()
                    totals["succeeded" if record["ok"] else "failed"] += 1
                    full = {**pending.pop(future), **record}
                    results.write(json.dumps(full) + "\n")
                    if observer is not None:
                        observer.on_result(full)
        def finish_window():
            nonlocal counts, window_index
            start = window_index * plan.window_seconds
            w = Window(start, start + plan.window_seconds, counts)
            detection = detector.observe(w)
            if detection.event:
                event_counts[detection.event] = event_counts.get(detection.event, 0) + 1
            if detection.event == "phase_ready":
                ready.append({"end": w.end, "phase_id": detection.phase_id})
            windows.write(json.dumps({"window": asdict(w), "detection": asdict(detection),
                                      "observed_at": time.monotonic() - started}) + "\n")
            if guard is not None:
                # 此回调只更新短小的观测状态；不得在调度线程中执行调参或压测。
                guard.observe(w, detection)
            if observer is not None:
                observer.on_window(w, detection)
            windows.flush()
            offered.flush()
            results.flush()
            window_index += 1
            counts = [[0] * (catalog.cluster_count + 1) for _ in range(plan.slices)]
        with ThreadPoolExecutor(max_workers=plan.workers) as pool:
            for request in schedule:
                if observer is not None:
                    observer.check_health()
                while request.scheduled >= (window_index + 1) * plan.window_seconds:
                    deadline = started + (window_index + 1) * plan.window_seconds
                    time.sleep(max(0, deadline - time.monotonic()))
                    finish_window()
                time.sleep(max(0, started + request.scheduled - time.monotonic()))
                collect()
                emitted = time.monotonic() - started
                lag = max(0, emitted - request.scheduled)
                max_lag = max(max_lag, lag)
                ch = channel_by_query[request.template_sql]
                offset = request.scheduled - window_index * plan.window_seconds
                counts[min(plan.slices - 1, int(offset * plan.slices / plan.window_seconds))][ch] += 1
                record = {**asdict(request), "emitted": emitted, "emit_lag_seconds": lag, "channel": ch}
                totals["offered"] += 1
                if len(pending) >= plan.workers + plan.queue_size:
                    totals["dropped"] += 1
                    offered.write(json.dumps({**record, "submitted": False}) + "\n")
                    if observer is not None:
                        observer.on_offered({**record, "submitted": False})
                else:
                    totals["submitted"] += 1
                    if observer is not None:
                        observer.on_offered({**record, "submitted": True})
                    pending[pool.submit(work, request)] = record
                    offered.write(json.dumps({**record, "submitted": True}) + "\n")
            time.sleep(max(0, started + plan.duration - time.monotonic()))
            finish_window()
        collect()
    result = {"simulated": simulated, "mode": "monitor_only_no_tuning", "plan": asdict(plan),
              "schedule_hash": fingerprint([asdict(r) for r in schedule]), "counts": totals,
              "phase_ready": ready, "max_emit_lag_seconds": max_lag,
              "complete": totals["offered"] == len(schedule) and totals["succeeded"] == len(schedule)
                          and max_lag <= plan.max_emit_lag_seconds,
              "three_phases_detected": len(ready) == 3,
              "expected_phase_count": plan.expected_phase_count,
              "phase_count_matches_plan": len(ready) == plan.expected_phase_count,
              "detector_event_counts": event_counts,
              "detector_matches_plan": len(ready) == plan.expected_phase_count
                                       and event_counts.get("change_confirmed", 0) == plan.expected_phase_count - 1
                                       and detector.state == "stable",
              "wall_seconds": time.monotonic() - started,
              "replay_started_at_utc": replay_started_at,
              "llm_calls": 0, "configuration_mutations": 0,
              "data_snapshot_restored": False}
    if guard is not None:
        result["mode"] = "phase_guard_acceptance_no_tuning"
        result["phase_guard_passed"] = guard.finish(output, result["complete"])["passed"]
    return result


class MySQLRequests:
    def __init__(self, settings):
        import pymysql
        self.driver, self.settings = pymysql, settings
        self.local = threading.local()
        self.connections = []
        self.lock = threading.Lock()

    def connect(self):
        s = self.settings
        conn = self.driver.connect(host=s["host"], port=int(s["port"]), user=s["mysql_user"],
                                   password=os.environ[s["mysql_password_env"]], database=s["database"],
                                   autocommit=True, connect_timeout=5, read_timeout=5, write_timeout=5)
        with self.lock:
            self.connections.append(conn)
        return conn

    def preflight(self, plan):
        conn = self.connect()
        with conn.cursor() as cursor:
            cursor.execute("SELECT VERSION(),@@port")
            version, port = cursor.fetchone()
            if str(version).split("-")[0] != "8.0.36" or int(port) != int(self.settings["port"]):
                raise ValueError("Expected dedicated MySQL 8.0.36 instance")
            for table in range(1, plan.tables + 1):
                cursor.execute(f"SELECT COUNT(*),MIN(id),MAX(id) FROM sbtest{table}")
                if tuple(cursor.fetchone()) != (plan.table_size, 1, plan.table_size):
                    raise ValueError("Replay requires complete contiguous sbtest IDs")
                cursor.execute(f"SELECT c,k FROM sbtest{table} WHERE id=1")
                if cursor.fetchone() is None:
                    raise ValueError("Expected Sysbench columns and rows")
        return {"mysql_version": version, "port": port, "tables_checked": plan.tables}

    def snapshot(self):
        names = ["innodb_buffer_pool_size", "innodb_io_capacity", "innodb_io_capacity_max",
                 "sort_buffer_size", "innodb_flush_log_at_trx_commit", "sync_binlog",
                 "innodb_doublewrite", "log_bin", "transaction_isolation", "performance_schema"]
        with self.connect().cursor() as cursor:
            cursor.execute("SELECT " + ",".join("@@" + name for name in names))
            values = dict(zip(names, map(str, cursor.fetchone())))
        path = Path(self.settings["config_path"])
        return {"variables": values, "config_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    def __call__(self, request):
        if not hasattr(self.local, "connection"):
            self.local.connection = self.connect()
        with self.local.connection.cursor() as cursor:
            cursor.execute(request.sql, request.parameters)
            if request.operation == "read":
                result_rows = cursor.fetchall()
                rows = len(result_rows)
                if isinstance(request, RangeRequest):
                    request.validate_result(result_rows)
                    return {"rows": 1, "aggregate_count": int(result_rows[0][0]), "result_fingerprint": fingerprint(
                        [[str(v) for v in row] for row in result_rows])}
            else:
                rows = cursor.rowcount
            if rows != 1:
                raise ValueError("Request did not touch exactly one expected row")
            return rows

    def close(self):
        for conn in self.connections:
            if conn.open:
                conn.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--scenario", choices=["composition", "intensity", "steady", "transient"], required=True)
    parser.add_argument("--phase-seconds", type=int, default=24)
    parser.add_argument("--query-profile", choices=["point", "range-mixed"], default="point")
    parser.add_argument("--table-size", type=int, default=100000)
    parser.add_argument("--range-rows", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--guard-acceptance", action="store_true",
                        help="保持 15+60 秒观测，验收阶段预算及跨阶段拒绝；不调参")
    parser.add_argument("--dedicated-instance", action="store_true")
    args = parser.parse_args()
    if not args.dedicated_instance:
        parser.error("Requires a dedicated experimental database; UPDATE changes data")
    try:
        plan = ReplayPlan(scenario=args.scenario, phase_seconds=args.phase_seconds,
                          query_profile=args.query_profile, table_size=args.table_size,
                          read_percent=100 if args.query_profile == "range-mixed" else 50,
                          range_rows=args.range_rows, top_k=args.top_k)
        if args.guard_acceptance:
            from .phase_safety import GuardAcceptance
            guard = GuardAcceptance(plan)
        else:
            guard = None
    except ValueError as exc:
        parser.error(str(exc))
    load_dotenv(args.env_file, override=True)
    target = yaml.safe_load(args.target.read_text())
    settings = target["mysql"]
    if settings.get("host", "127.0.0.1") not in {"127.0.0.1", "localhost"}:
        parser.error("Replay currently requires local MySQL")
    args.output.mkdir(parents=True, exist_ok=False)
    client = None
    result = {"simulated": False, "complete": False}
    created_at = datetime.now(timezone.utc).isoformat()
    initial = None
    try:
        client = MySQLRequests(settings)
        preflight = client.preflight(plan)
        initial = client.snapshot()
        manifest = {"plan": asdict(plan), "preflight": preflight, **source_provenance(),
                    "execution_hostname": socket.gethostname(),
                    "created_at_utc": created_at, "initial_database_state": initial,
                    "environment_ids": target.get("admtune", {}),
                    "declared_hardware": target.get("target", {}).get("hardware", {}),
                    "request_semantics": query_semantics(plan),
                    "phase_guard_acceptance": args.guard_acceptance,
                    "phase_timing": guard.timing.report() if guard else None,
                    "warning": "Monitor-only low-rate replay, not Sysbench TPS or dynamic tuning"}
        (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
        result = replay(plan, client, args.output, simulated=False, guard=guard)
    except Exception as exc:
        result["error_type"] = type(exc).__name__
        result["error"] = redact(str(exc))
    finally:
        if client is not None:
            try:
                final = client.snapshot()
                result["final_database_state"] = final
                result["configuration_unchanged"] = initial is not None and initial == final
                result["complete"] = result["complete"] and result["configuration_unchanged"]
            except Exception as exc:
                result["audit_error_type"] = type(exc).__name__
                result["audit_error"] = redact(str(exc))
                result["complete"] = False
            try:
                client.close()
            except Exception as exc:
                result["close_error_type"] = type(exc).__name__
                result["complete"] = False
        result["created_at_utc"] = created_at
        result["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        (args.output / "summary.json").write_text(json.dumps(result, indent=2))
        if all((args.output / name).exists() for name in ["offered.jsonl", "requests.jsonl", "windows.jsonl"]):
            try:
                from .replay_report import export_replay
                export_replay(args.output)
            except Exception as exc:
                result["export_error_type"] = type(exc).__name__
                result["complete"] = False
                (args.output / "summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return 0 if (result.get("complete") and result.get("detector_matches_plan")
                 and (not args.guard_acceptance or result.get("phase_guard_passed"))) else 1


if __name__ == "__main__":
    raise SystemExit(main())
