"""零LLM静态诊断：三配置、三位置平衡区组，采集正式测量边界计数器。"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shlex
import statistics
import time

from dotenv import load_dotenv

from config import load_app_config
from connectors.local_connector import LocalConnector
from .calibration import hardware_context, process_sample, status_sample, write_json
from .contracts import Objective
from .dynamic import instance_lock
from .live import load_space
from .paper_pilot import PaperExecutor
from .provenance import source_provenance
from .replay_report import write_csv
from .security import redact


NAMES = ("Default", "BufferOnly", "ADMTuneConfig")


def plan():
    # 各配置在每个区组、每个位置各出现一次；不是随机排列。
    return [{"seq": 3*b+i+1, "block": b+1, "position": i+1, "name": NAMES[(b+i) % 3]}
            for b in range(3) for i in range(3)]


def configurations(initial, chosen, space):
    if initial != {"innodb_buffer_pool_size": 1024**3, "innodb_io_capacity": 200,
                   "innodb_io_capacity_max": 2000, "sort_buffer_size": 262144}:
        raise ValueError("诊断只接受冻结的共同初始配置")
    full = space.merge(chosen, {})
    if full != {"innodb_buffer_pool_size": 4*1024**3, "innodb_io_capacity": 1200,
                "innodb_io_capacity_max": 4500, "sort_buffer_size": 262144}:
        raise ValueError("ADMTune配置必须来自已核验的100分钟运行")
    return dict(zip(NAMES, [dict(initial), space.merge(initial, {
        "innodb_buffer_pool_size": 4*1024**3}), full]))


def cgroup_sample():
    for base in (Path("/sys/fs/cgroup/cpu,cpuacct"), Path("/sys/fs/cgroup/cpu")):
        if (base/"cpu.stat").is_file():
            fields = dict(line.split() for line in (base/"cpu.stat").read_text().splitlines())
            usage = base/"cpuacct.usage"
            return {"version": 1, "path": str(base),
                    "nr_periods": int(fields["nr_periods"]),
                    "nr_throttled": int(fields["nr_throttled"]),
                    "throttled_seconds": int(fields["throttled_time"])/1e9,
                    "usage_seconds": int(usage.read_text())/1e9 if usage.is_file() else None,
                    "quota_us": int((base/"cpu.cfs_quota_us").read_text()),
                    "period_us": int((base/"cpu.cfs_period_us").read_text())}
    base = Path("/sys/fs/cgroup")
    if (base/"cpu.stat").is_file() and (base/"cpu.max").is_file():
        fields = dict(line.split() for line in (base/"cpu.stat").read_text().splitlines())
        quota, period = (base/"cpu.max").read_text().split()
        return {"version": 2, "path": str(base), "nr_periods": int(fields["nr_periods"]),
                "nr_throttled": int(fields["nr_throttled"]),
                "throttled_seconds": int(fields["throttled_usec"])/1e6,
                "usage_seconds": int(fields["usage_usec"])/1e6,
                "quota_us": None if quota == "max" else int(quota), "period_us": int(period)}
    raise ValueError("找不到本容器CPU配额与节流计数器；不以宿主机总量替代")


def counter_delta(before, after):
    if before["process"]["start_ticks"] != after["process"]["start_ticks"]:
        raise ValueError("测量期间MySQL进程更换")
    if before["cgroup"]["path"] != after["cgroup"]["path"]:
        raise ValueError("测量期间cgroup路径更换")
    elapsed = after["at"]-before["at"]
    if elapsed <= 0:
        raise ValueError("计数器边界时长不合法")
    def diff(a, b, key):
        if a[key] is None or b[key] is None:
            return None
        result = b[key]-a[key]
        if result < 0:
            raise ValueError(f"计数器重置：{key}")
        return result
    p, q = before["process"], after["process"]
    c, d = before["cgroup"], after["cgroup"]
    periods = diff(c, d, "nr_periods")
    throttled = diff(c, d, "nr_throttled")
    usage = diff(c, d, "usage_seconds")
    status = {k: diff(before["mysql"], after["mysql"], k)
              for k in before["mysql"] if k in after["mysql"]}
    return {"boundary_elapsed_seconds": elapsed,
            "mysqld_cpu_cores": diff(p, q, "cpu_seconds")/elapsed,
            "mysqld_read_bytes": diff(p, q, "read_bytes"),
            "mysqld_write_bytes": diff(p, q, "write_bytes"),
            "cgroup_cpu_cores": None if usage is None else usage/elapsed,
            "nr_periods": periods, "nr_throttled": throttled,
            "throttled_period_fraction": throttled/periods if periods else None,
            # v1节流时长来自实体累计时间，绝不除以墙钟后宣称CPU阻塞百分比。
            "throttled_seconds": diff(c, d, "throttled_seconds"), "mysql": status}


class DiagnosticConnector(LocalConnector):
    capture = None
    fixed_rate = None

    def run(self, command, **kwargs):
        tokens = shlex.split(command)
        benchmark = bool(tokens and tokens[0] == "sysbench" and tokens[-1] == "run")
        measured = benchmark and "--time=60" in tokens
        if benchmark:
            if self.fixed_rate is not None:
                if self.fixed_rate != 1000 or any(t.startswith(("--rate=", "--tx-rate=")) for t in tokens):
                    raise ValueError("固定速率复验只允许显式1000且不能重复指定速率")
                tokens.insert(-1, "--rate=1000")
                command = shlex.join(tokens)
            # 仅保存可公开的控制参数，绝不保存带数据库密码的完整命令。
            evidence = {"script": tokens[1], "duration": next(
                (int(t.split("=", 1)[1]) for t in tokens if t.startswith("--time=")), None),
                "rate": self.fixed_rate, "rate_flag": next(
                (t for t in tokens if t.startswith("--rate=")), None)}
            if not hasattr(self, "command_evidence"):
                self.command_evidence = []
            self.command_evidence.append(evidence)
        if not measured or self.capture is None:
            return super().run(command, **kwargs)
        before = self.capture()
        try:
            return super().run(command, **kwargs)
        finally:
            after = self.capture()
            self.boundaries = {"before": before, "after": after,
                               "delta": counter_delta(before, after)}


def rate_check(log, tps, target=1000):
    """速率是期望平均事件率，不将60000期望事件数当作固定提交数。"""
    header = re.search(r"Target transaction rate:\s*(\d+)\s*/sec", log)
    queues = [int(v) for v in re.findall(r"(?:qlen|queue length):\s*(\d+)", log)]
    return {"target_tps": target, "expected_events": target*60,
            "native_header_rate": int(header.group(1)) if header else None,
            "achieved_fraction": None if tps is None else tps/target,
            "tolerance_fraction": .05, "max_reported_queue": max(queues) if queues else None,
            "passed": bool(header and int(header.group(1)) == target and tps is not None
                           and abs(tps/target-1) <= .05)}


def summarize(records):
    rows = []
    for name in NAMES:
        data = [r for r in records if r["name"] == name and r["valid"]]
        rows.append({"name": name, "n": len(data),
                     "median_tps": statistics.median(r["measurement"]["value"] for r in data) if data else None,
                     "median_p95_ms": statistics.median(r["measurement"]["metadata"]["p95_latency_ms"] for r in data) if data else None,
                     "median_throttled_fraction": statistics.median(
                         r["telemetry"]["delta"]["throttled_period_fraction"] for r in data
                         if r["telemetry"]["delta"]["throttled_period_fraction"] is not None) if
                     any(r["telemetry"]["delta"]["throttled_period_fraction"] is not None for r in data) else None})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("target", "benchmark", "knobs", "output", "restore-audit", "parent-run"):
        parser.add_argument("--"+key, type=Path, required=True)
    parser.add_argument("--dedicated-instance", action="store_true")
    parser.add_argument("--fixed-rate", type=int, choices=[1000])
    a = parser.parse_args()
    if not a.dedicated_instance:
        parser.error("必须指定专用实例")
    root = Path(__file__).resolve().parents[1]
    load_dotenv(override=True)
    app = load_app_config(project_root=root, target_path=a.target, benchmark_path=a.benchmark,
                          knobs_path=a.knobs, dry_run_override=False)
    w = app.benchmark.workload
    if (app.target.mysql.host not in {"localhost", "127.0.0.1"} or
            (w.threads, w.duration, w.warmup_time, w.rand_type, w.rand_seed) != (32, 60, 15, "uniform", 42)
            or w.script_path != "/usr/share/sysbench/oltp_read_only.lua"):
        parser.error("只接受冻结的本地32线程、uniform、15+60秒只读协议")
    audit = json.loads(a.restore_audit.read_text())
    parent = json.loads((a.parent_run/"summary.json").read_text())
    parent_manifest = json.loads((a.parent_run/"manifest.json").read_text())
    parent_hashes = json.loads((a.parent_run/"checksums.json").read_text())
    for name in ("summary.json", "manifest.json"):
        if hashlib.sha256((a.parent_run/name).read_bytes()).hexdigest() != parent_hashes[name]:
            parser.error("原运行清单或摘要校验失败")
    if not audit.get("completed") or parent.get("completed_until") != 71 or not parent.get("initial_configuration_restored"):
        parser.error("需要完成的冷恢复审计及原71次运行")
    if parent_manifest["benchmark"] != asdict(app.benchmark):
        parser.error("与原100分钟运行的基准协议不一致")
    lock = instance_lock(app.target.mysql.config_path)
    a.output.mkdir(mode=0o700, parents=True, exist_ok=False)
    space = load_space(a.knobs, memory_bytes=app.target.hardware_memory_bytes)
    connector = DiagnosticConnector(app.target.ssh, config_path=app.target.mysql.config_path,
                                    dbms="mysql", dry_run=False)
    connector.fixed_rate = a.fixed_rate
    executor = PaperExecutor(app, space, Objective(), a.output, allow_mutations=True,
                             local=True, connector=connector)
    records, result = [], {"completed": False, "llm_calls": 0}
    started = time.monotonic()
    try:
        preflight = executor.preflight()
        if preflight["config_sha256"] != audit["config_sha256"]:
            raise ValueError("初始配置与冷恢复不一致")
        configs = configurations(preflight["current_config"], parent["incumbent"], space)
        pid = int(Path(app.target.mysql.pid_file).read_text())
        same_cgroup = (Path(f"/proc/{pid}/cgroup").read_text() == Path("/proc/self/cgroup").read_text())
        if not same_cgroup:
            raise ValueError("MySQL与采样进程不在同一cgroup，拒绝错误归因")
        def capture():
            return {"at": time.monotonic(), "cgroup": cgroup_sample(), "process": process_sample(pid),
                    "mysql": status_sample(executor.db)}
        connector.capture = capture
        write_json(a.output/"manifest.json", {"mode": "zero_llm_rate_diagnostic" if a.fixed_rate else "zero_llm_static_diagnostic", "plan": plan(),
            "configs": configs, "benchmark": asdict(app.benchmark), "preflight": preflight,
            "fixed_rate_tps": a.fixed_rate,
            "rate_policy": {"warmup_and_measurement": True, "achieved_tolerance_fraction": .05,
                            "expected_events_are_not_exact_arrivals": True} if a.fixed_rate else None,
            "hardware": hardware_context(), "same_cgroup": same_cgroup, "restore": audit,
            "parent_summary_sha256": hashlib.sha256((a.parent_run/"summary.json").read_bytes()).hexdigest(),
            "parent_manifest_sha256": hashlib.sha256((a.parent_run/"manifest.json").read_bytes()).hexdigest(),
            "telemetry_scope": "60-second measured command boundaries; excludes 15-second warmup; includes small sampling overhead",
            "cache_protocol": "one initial cold restore; restore Default before/after each trial; balanced Latin blocks; no per-trial cold restore",
            "created_at_utc": datetime.now(timezone.utc).isoformat(), **source_provenance()})
        for spec in plan():
            if time.monotonic()-started > 1500:
                raise TimeoutError("诊断预算1500秒已到，不再提交测量")
            executor.restore_initial()
            folder = a.output/"trials"/f"{spec['seq']:06d}"
            folder.mkdir(parents=True, mode=0o700)
            executor.run_dir = folder
            connector.boundaries = None
            connector.command_evidence = []
            offset = len(executor.resize_checks)
            m = executor.measure(configs[spec["name"]], [0.0])
            boundaries = connector.boundaries
            executor.restore_initial()
            valid = bool(m.status == "measured" and m.complete and m.effective and m.constraints_ok and boundaries)
            gate = None
            if a.fixed_rate:
                raw = folder/m.metadata["benchmark_log"] if m.metadata.get("benchmark_log") else None
                gate = rate_check(raw.read_text() if raw and raw.exists() else "", m.value, a.fixed_rate)
                expected_commands = [(15, "--rate=1000"), (60, "--rate=1000")]
                actual_commands = [(c["duration"], c["rate_flag"]) for c in connector.command_evidence]
                valid = valid and gate["passed"] and actual_commands == expected_commands
            record = {**spec, "config": configs[spec["name"]], "valid": valid, "measurement": asdict(m),
                      "telemetry": boundaries, "resize_checks": executor.resize_checks[offset:],
                      "rate_check": gate, "command_evidence": connector.command_evidence,
                      "elapsed_seconds": time.monotonic()-started,
                      "benchmark_log": str(folder.relative_to(a.output)/m.metadata["benchmark_log"]) if m.metadata.get("benchmark_log") else None}
            records.append(record)
            with (a.output/"trials.jsonl").open("a") as f:
                f.write(json.dumps(record, allow_nan=False)+"\n")
            write_json(a.output/"progress.json", {"completed_trials": len(records), "last": record})
            if not valid:
                raise ValueError("无效观测，停止诊断，不补造成功记录")
        result.update(completed=True, aggregates=summarize(records))
    except Exception as exc:
        result["error"] = redact(str(exc))
    finally:
        try:
            executor.restore_initial()
            result["initial_configuration_restored"] = executor.initial is not None
        except Exception as exc:
            result.update(completed=False, recovery_error=redact(str(exc)))
        result.update(completed_trials=len(records), elapsed_seconds=time.monotonic()-started,
                      finished_at_utc=datetime.now(timezone.utc).isoformat())
        write_json(a.output/"summary.json", result)
        write_csv(a.output/"metrics.csv", [
            {**{k: r[k] for k in ("seq", "block", "position", "name", "valid")},
             "tps": r["measurement"]["value"], "p95_ms": r["measurement"]["metadata"].get("p95_latency_ms"),
             **({k: v for k, v in r["telemetry"]["delta"].items() if k != "mysql"} if r["telemetry"] else {})}
            for r in records], ["seq", "block", "position", "name", "valid", "tps", "p95_ms",
                               "boundary_elapsed_seconds", "mysqld_cpu_cores", "mysqld_read_bytes", "mysqld_write_bytes",
                               "cgroup_cpu_cores", "nr_periods", "nr_throttled", "throttled_period_fraction", "throttled_seconds"])
        write_json(a.output/"checksums.json", {
            str(f.relative_to(a.output)): hashlib.sha256(f.read_bytes()).hexdigest()
            for f in a.output.rglob("*") if f.is_file() and f.suffix in {".json", ".jsonl", ".csv", ".log"}
            and f.name != "checksums.json"})
        executor.close()
        lock.close()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("completed") and result.get("initial_configuration_restored") else 1


if __name__ == "__main__":
    raise SystemExit(main())
