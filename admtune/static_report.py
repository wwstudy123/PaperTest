"""将真实静态调优的既有记录导出为 CSV；不连接数据库或模型。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .replay_report import write_csv


def export_static(root: Path):
    records = [json.loads(line) for line in (root / "trials.jsonl").read_text().splitlines() if line.strip()]
    if len({r["trial_id"] for r in records}) != len(records):
        raise ValueError("Duplicate trial IDs")
    summary = json.loads((root / "summary.json").read_text())
    names = sorted({name for r in records for name in r["config"]})
    rows = []
    for index, r in enumerate(records, 1):
        metadata, prediction = r.get("metadata", {}), r.get("prediction") or {}
        rows.append({"trial_index": index, "trial_id": r["trial_id"], "phase_id": r["phase_id"],
                     "kind": metadata.get("kind"), "status": r["status"],
                     "complete": r["complete"], "effective": r["effective"],
                     "constraints_ok": r["constraints_ok"], "simulated": metadata.get("simulated"),
                     "accepted": metadata.get("accepted"), "objective": r["objective_key"],
                     "value": r["value"], "p95_latency_ms": metadata.get("p95_latency_ms"),
                     "prediction_log_mean": prediction.get("mean"),
                     "prediction_log_latent_std": prediction.get("latent_std"),
                     "prediction_log_observation_std": prediction.get("observation_std"),
                     **{"config_" + name: r["config"].get(name) for name in names}})
    fields = ["trial_index", "trial_id", "phase_id", "kind", "status", "complete", "effective",
              "constraints_ok", "simulated", "accepted", "objective", "value", "p95_latency_ms",
              "prediction_log_mean", "prediction_log_latent_std", "prediction_log_observation_std"]
    write_csv(root / "static_trials.csv", rows, fields + ["config_" + name for name in names])
    calls = []
    if (root / "llm_usage.json").exists():
        usage = json.loads((root / "llm_usage.json").read_text())
        for index, r in enumerate(usage.get("records", []), 1):
            calls.append({"usage_index": index, **{name: r.get(name) for name in [
                "timestamp", "operation", "model", "duration_s", "input_tokens", "output_tokens", "total_tokens", "cached_tokens"]}})
    write_csv(root / "static_llm_usage.csv", calls, ["usage_index", "timestamp", "operation", "model",
              "duration_s", "input_tokens", "output_tokens", "total_tokens", "cached_tokens"])
    repeats = summary.get("independent_verification_values", [])
    write_csv(root / "static_verification.csv", [
        {"repeat_index": i, "value": value, "objective": summary.get("objective")}
        for i, value in enumerate(repeats, 1)
    ], ["repeat_index", "value", "objective"])
    return {"trials": len(rows), "usage_records": len(calls), "verification_repeats": len(repeats),
            "note": "Only successful returned usage records; unknown-usage failures remain in original logs."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(export_static(args.run_dir)))


if __name__ == "__main__":
    main()
