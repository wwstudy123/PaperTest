"""只读诊断原ADMTune首阶段；不连接数据库/LLM，不改原记录或把修复收益当实测。"""
import argparse
import ast
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sqlite3
from zoneinfo import ZoneInfo

import numpy as np

from admtune.continuous import atomic_json
from admtune.replay_report import read_jsonl, write_csv
from tools.plot_dynamic_matrix import service_metrics
from tools.analyze_dynamic_stage1 import adaptation


def analyze(batch, output):
    root = batch / "seed-43/ADMTune"
    output.mkdir(parents=True, exist_ok=False)
    summary = json.loads((root / "summary.json").read_text())
    origin = datetime.fromisoformat(summary["replay_started_at_utc"])
    records = read_jsonl(root / "trials.jsonl")
    phase = [r for r in records if r["phase_id"] == "phase-1"]
    db = sqlite3.connect(f"file:{root / 'records.sqlite3'}?mode=ro", uri=True)
    try:
        events = [(k, json.loads(v)) for k, v in db.execute("SELECT kind,payload FROM events ORDER BY seq")]
    finally:
        db.close()
    timeline, failures = [], []
    for i, r in enumerate(phase):
        meta = r["metadata"]
        guard = meta["phase_guard"]
        next_start = (phase[i + 1]["metadata"]["phase_guard"]["start"]
                      if i + 1 < len(phase) else guard.get("sealed_at_window_end"))
        row = {"trial_id": r["trial_id"], "kind": meta["kind"], "status": r["status"],
               "guard_start_s": guard["start"], "next_trial_start_or_last_seal_s": next_start,
               "measurement_start_s": meta.get("measurement_start"),
               "measurement_end_s": meta.get("measurement_end"),
               "sealed_at_s": guard.get("sealed_at_window_end"),
               "p95_ms": r["value"], "accepted": meta.get("accepted"),
               "buffer_gib": r["config"]["innodb_buffer_pool_size"] / 1024**3}
        timeline.append(row)
        prefix = "Buffer-pool resize did not settle: "
        reason = meta.get("reason", "")
        if r["status"] != "measured":
            detail = ast.literal_eval(reason[len(prefix):]) if reason.startswith(prefix) else {}
            failures.append({**row, "reason": reason, "resize_status_code": detail.get("status_code"),
                "recorded_resize_wait_s": detail.get("wait_seconds", 0),
                "failed_trial_to_next_start_s": next_start - guard["start"]})
    calls = []
    for f in sorted((root / "llm_calls").glob("*.json")):
        d = json.loads(f.read_text())
        end = datetime.fromisoformat(d["usage"]["timestamp"]).replace(tzinfo=ZoneInfo("Asia/Shanghai"))
        end_s = (end - origin).total_seconds()
        experiences = d["payload"].get("experiences", [])
        calls.append({"file": f.name, "role": d["role"], "end_s": end_s,
                      "start_s": end_s - d["elapsed_seconds"], "duration_s": d["elapsed_seconds"],
                      "tokens": d["usage"]["total_tokens"],
                      "measurement_experiences": sum(x.get("type") == "measurement" for x in experiences),
                      "reflection_experiences": sum(x.get("type") == "reflection" for x in experiences)})
    window_rows = []
    for method in ("Default", "AgenticDB", "AgentTune", "ADMTune"):
        _, windows, _, _ = service_metrics(read_jsonl(batch / "seed-43" / method / "requests.jsonl"))
        window_rows.extend({"method": method, **w} for w in windows[:30])
    reference = [r["p95_ms"] for r in window_rows if r["method"] == "Default"]
    threshold = .9 * float(np.median(reference))
    own = [r for r in window_rows if r["method"] == "ADMTune"]
    first_good = next(r for r in timeline if r["status"] == "measured"
                      and r["kind"] != "verification" and r["p95_ms"] <= threshold)
    prescreens = [p for k, p in events if k == "prescreen" and p["phase_id"] == "phase-1"]
    result = {
        "scope": "offline diagnosis of original seed43; no new performance measurement",
        "database_calls": 0, "llm_calls": 0, "a1_target_ms": threshold,
        "first_good_observation_id": first_good["trial_id"],
        "first_good_observation_sealed_at_s": first_good["sealed_at_s"],
        "first_good_observation_p95_ms": first_good["p95_ms"],
        "sustained_service_attainment_s": adaptation(own, 0, threshold),
        "a1_prescreen_events": len(prescreens), "a1_failure_count": len(failures),
        "recorded_failed_resize_wait_s": sum(r["recorded_resize_wait_s"] for r in failures),
        "failed_trial_to_next_start_intervals_s": sum(r["failed_trial_to_next_start_s"] for r in failures),
        "interval_note": "Includes apply/recovery/computation; not all of it is resize wait.",
        "a1_selector_pruner_recommender_s": sum(r["duration_s"] for r in calls
                                              if r["end_s"] < 600 and r["role"] != "reflection"),
        "all_trial_slots": len(records), "valid_observations": sum(r["status"] == "measured" for r in records),
        "source_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                          for name in ("summary.json", "trials.jsonl", "requests.jsonl")},
    }
    for name, rows in (("trial_timeline", timeline), ("failed_trials", failures),
                       ("llm_and_prompt_trace", calls), ("window_trace", window_rows)):
        write_csv(output / f"{name}.csv", rows, list(rows[0]))
    atomic_json(output / "summary.json", result)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 10, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(8, 3.9))
    for method, color in (("Default", "#777777"), ("AgenticDB", "#1f77b4"),
                          ("AgentTune", "#ff7f0e"), ("ADMTune", "#d62728")):
        rs = [r for r in window_rows if r["method"] == method]
        ax.plot([r["end_s"] / 60 for r in rs], [r["p95_ms"] for r in rs],
                color=color, lw=1.2, label=method, ls="--" if method == "Default" else "-")
    for i, r in enumerate(failures):
        ax.axvspan(r["guard_start_s"] / 60, r["next_trial_start_or_last_seal_s"] / 60,
                   color="#aaaaaa", alpha=.2, label="Failed trial + recovery" if i == 0 else None)
    ax.axhline(threshold, color="#555555", ls=":", lw=.9, label="Frozen attainment target")
    ax.axvline(first_good["sealed_at_s"] / 60, color="#2ca02c", ls="--", lw=1,
               label="First good observation sealed")
    ax.set(xlabel="A1 workload time (min)", ylabel="P95 latency (ms; 20 s cohorts)",
           xlim=(0, 10), title="A1 service latency and original search interruptions")
    ax.tick_params(direction="in", top=True, right=True)
    ax.grid(alpha=.2)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(output / f"a1_diagnosis.{ext}", dpi=260, bbox_inches="tight")
    plt.close(fig)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batch", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(analyze(a.batch, a.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
