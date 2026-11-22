"""将回放原始证据离线导出为绘图 CSV，不调用数据库或模型。"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def export_replay(root: Path):
    offered = read_jsonl(root / "offered.jsonl")
    results = read_jsonl(root / "requests.jsonl")
    windows = read_jsonl(root / "windows.jsonl")
    result_by_id = {r["seq"]: r for r in results}
    if len(result_by_id) != len(results) or len({r["seq"] for r in offered}) != len(offered):
        raise ValueError("Duplicate request evidence")
    if set(result_by_id) - {r["seq"] for r in offered if r["submitted"]}:
        raise ValueError("Completion without submitted request")
    raw = []
    for record in offered:
        r = result_by_id.get(record["seq"], {})
        begin, end = r.get("started"), r.get("completed")
        raw.append({"seq": record["seq"], "phase_label": record["phase"], "operation": record["operation"],
                    "channel": record["channel"], "table": record["table"], "row": record["row"],
                    "scheduled_s": record["scheduled"], "emitted_s": record["emitted"],
                    "submitted": record["submitted"], "started_s": begin, "completed_s": end,
                    "ok": r.get("ok"), "error_type": r.get("error_type", ""),
                    "emit_lag_ms": 1000 * record["emit_lag_seconds"],
                    "queue_wait_ms": None if begin is None else 1000 * (begin - record["emitted"]),
                    "service_ms": None if end is None else 1000 * (end - begin),
                    "e2e_ms": None if end is None else 1000 * (end - record["emitted"]),
                    "scheduled_to_completion_ms": None if end is None else 1000 * (end - record["scheduled"])})
    fields = ["seq", "phase_label", "operation", "channel", "table", "row", "scheduled_s", "emitted_s",
              "submitted", "started_s", "completed_s", "ok", "error_type", "emit_lag_ms", "queue_wait_ms",
              "service_ms", "e2e_ms", "scheduled_to_completion_ms"]
    write_csv(root / "requests.csv", raw, fields)
    aggregate, channels, events = [], [], []
    for item in windows:
        w, detection = item["window"], item["detection"]
        start, end = w["start"], w["end"]
        cohort = [r for r in raw if start <= r["scheduled_s"] < end]
        emitted = [r for r in raw if start <= r["emitted_s"] < end]
        completed = [r for r in raw if r["completed_s"] is not None and start <= r["completed_s"] < end]
        latencies = [r["e2e_ms"] for r in cohort if r["ok"]]
        offered_count = sum(map(sum, w["counts"]))
        if offered_count != len(cohort):
            raise ValueError("Window count does not match offered evidence")
        aggregate.append({"start_s": start, "end_s": end, "scheduled_offered": len(cohort),
                          "offered_qps": len(cohort) / (end - start), "actual_emitted": len(emitted),
                          "submitted": sum(r["submitted"] for r in cohort),
                          "dropped": sum(not r["submitted"] for r in cohort),
                          "cohort_failed": sum(r["ok"] is False for r in cohort),
                          "completed_success": sum(r["ok"] is True for r in completed),
                          "completed_qps": sum(r["ok"] is True for r in completed) / (end - start),
                          "cohort_read_share": sum(r["operation"] == "read" for r in cohort) / max(1, len(cohort)),
                          "cohort_e2e_p50_ms": float(np.percentile(latencies, 50)) if latencies else None,
                          "cohort_e2e_p95_ms": float(np.percentile(latencies, 95)) if latencies else None,
                          "emit_lag_max_ms": max((r["emit_lag_ms"] for r in cohort), default=0),
                          "detector_state": detection["state"], "detector_event": detection["event"],
                          "detector_phase_id": detection["phase_id"], "observed_at_s": item["observed_at"]})
        for i, counts in enumerate(w["counts"]):
            seconds = (end - start) / len(w["counts"])
            for channel, count in enumerate(counts):
                channels.append({"window_start_s": start, "slice_start_s": start + i * seconds,
                                 "slice_end_s": start + (i + 1) * seconds,
                                 "channel": channel, "offered_count": count, "offered_qps": count / seconds})
        if detection["event"]:
            events.append({"window_end_s": end, "observed_at_s": item["observed_at"],
                           "event": detection["event"], "state": detection["state"], "phase_id": detection["phase_id"]})
    write_csv(root / "windows.csv", aggregate, list(aggregate[0]) if aggregate else ["start_s", "end_s"])
    write_csv(root / "template_rates.csv", channels, ["window_start_s", "slice_start_s", "slice_end_s", "channel", "offered_count", "offered_qps"])
    write_csv(root / "events.csv", events, ["window_end_s", "observed_at_s", "event", "state", "phase_id"])
    checksums = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir() if p.is_file() and p.suffix in {".json", ".jsonl", ".csv"} and p.name != "checksums.json"}
    (root / "checksums.json").write_text(json.dumps(checksums, indent=2) + "\n")
    return {"requests": len(raw), "windows": len(aggregate), "events": len(events)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(export_replay(args.run_dir)))


if __name__ == "__main__":
    main()
