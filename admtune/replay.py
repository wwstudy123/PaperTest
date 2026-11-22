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

    def __post_init__(self):
        if self.scenario not in {"composition", "intensity"}:
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
        if not math.isfinite(self.max_emit_lag_seconds) or self.max_emit_lag_seconds <= 0:
            raise ValueError("Finite positive emission lag bound required")
        if self.duration * self.high_rate > 100000:
            raise ValueError("Replay exceeds smoke request cap")

    @property
    def duration(self):
        return 3 * self.phase_seconds


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


def make_schedule(plan: ReplayPlan) -> list[Request]:
    rng = random.Random(plan.seed)
    requests = []
    for second in range(plan.duration):
        phase_index = second // plan.phase_seconds
        rate = plan.high_rate if plan.scenario == "intensity" and phase_index == 1 else plan.base_rate
        reads = rate * (1 if phase_index == 1 else 9) // 10 if plan.scenario == "composition" else rate // 2
        operations = ["read"] * reads + ["write"] * (rate - reads)
        rng.shuffle(operations)
        for i, operation in enumerate(operations):
            requests.append(Request(len(requests), second + i / rate, "ABA"[phase_index], operation,
                                    rng.randint(1, plan.tables), rng.randint(1, plan.table_size)))
    return requests


def monitor(plan: ReplayPlan):
    # 模板全集来自预声明操作，不能扫描未来测试阶段来发现模板。
    catalog = TemplateCatalog([
        Request(0, 0, "", operation, table, 1).template_sql
        for operation in ["read", "write"] for table in range(1, plan.tables + 1)
    ], min_samples=1)
    channels = catalog.cluster_count + 1
    # 独立的预声明中性校准窗口；不使用测试阶段观测拟合归一化器或阈值。
    counts = [[0] * channels for _ in range(plan.slices)]
    slice_seconds = plan.window_seconds / plan.slices
    for operation in ["read", "write"]:
        ch = catalog.channel(Request(0, 0, "", operation, 1, 1).template_sql)
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


def replay(plan: ReplayPlan, execute, output: Path, *, simulated: bool = True) -> dict:
    """execute 只由有界工作线程调用；调度不等待查询完成，过载明确丢弃。"""
    schedule = make_schedule(plan)
    catalog, detector, metadata = monitor(plan)
    channel_by_query = {r.template_sql: catalog.channel(r.template_sql) for r in schedule}
    (output / "schedule.json").write_text(json.dumps([asdict(r) for r in schedule]))
    (output / "monitor.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    replay_started_at = datetime.now(timezone.utc).isoformat()
    started = time.monotonic()
    pending = {}
    counts = [[0] * (catalog.cluster_count + 1) for _ in range(plan.slices)]
    window_index = 0
    totals = dict(offered=0, submitted=0, dropped=0, succeeded=0, failed=0)
    max_lag = 0.0
    ready = []
    def work(request):
        begin = time.monotonic() - started
        try:
            rows = execute(request)
            return {"seq": request.seq, "started": begin, "completed": time.monotonic() - started,
                    "ok": True, "rows": rows}
        except Exception as exc:
            return {"seq": request.seq, "started": begin, "completed": time.monotonic() - started,
                    "ok": False, "error_type": type(exc).__name__}
    with (output / "offered.jsonl").open("w") as offered, (output / "requests.jsonl").open("w") as results, (output / "windows.jsonl").open("w") as windows:
        def collect():
            for future in list(pending):
                if future.done():
                    record = future.result()
                    totals["succeeded" if record["ok"] else "failed"] += 1
                    results.write(json.dumps({**pending.pop(future), **record}) + "\n")
        def finish_window():
            nonlocal counts, window_index
            start = window_index * plan.window_seconds
            w = Window(start, start + plan.window_seconds, counts)
            detection = detector.observe(w)
            if detection.event == "phase_ready":
                ready.append({"end": w.end, "phase_id": detection.phase_id})
            windows.write(json.dumps({"window": asdict(w), "detection": asdict(detection),
                                      "observed_at": time.monotonic() - started}) + "\n")
            windows.flush()
            offered.flush()
            results.flush()
            window_index += 1
            counts = [[0] * (catalog.cluster_count + 1) for _ in range(plan.slices)]
        with ThreadPoolExecutor(max_workers=plan.workers) as pool:
            for request in schedule:
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
                else:
                    totals["submitted"] += 1
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
              "wall_seconds": time.monotonic() - started,
              "replay_started_at_utc": replay_started_at,
              "llm_calls": 0, "configuration_mutations": 0,
              "data_snapshot_restored": False}
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
            cursor.execute(request.sql, (request.row,))
            if request.operation == "read":
                rows = len(cursor.fetchall())
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
    parser.add_argument("--scenario", choices=["composition", "intensity"], required=True)
    parser.add_argument("--dedicated-instance", action="store_true")
    args = parser.parse_args()
    if not args.dedicated_instance:
        parser.error("Requires a dedicated experimental database; UPDATE changes data")
    load_dotenv(args.env_file, override=True)
    target = yaml.safe_load(args.target.read_text())
    settings = target["mysql"]
    if settings.get("host", "127.0.0.1") not in {"127.0.0.1", "localhost"}:
        parser.error("Replay currently requires local MySQL")
    plan = ReplayPlan(scenario=args.scenario)
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
                    "created_at_utc": created_at, "initial_database_state": initial,
                    "environment_ids": target.get("admtune", {}),
                    "declared_hardware": target.get("target", {}).get("hardware", {}),
                    "request_semantics": "单条 SELECT/UPDATE，autocommit；QPS 不是 Sysbench 事务 TPS",
                    "warning": "Monitor-only low-rate replay, not Sysbench TPS or dynamic tuning"}
        (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
        result = replay(plan, client, args.output, simulated=False)
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
    return 0 if result.get("complete") and result.get("three_phases_detected") else 1


if __name__ == "__main__":
    raise SystemExit(main())
