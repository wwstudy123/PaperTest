"""只读核对消融开关、真实GP观测和反思通路，不连接数据库或调用LLM。"""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sqlite3

from admtune.continuous import atomic_json
from admtune.replay_report import read_jsonl, write_csv


def inspect_run(root):
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("method") != "ADMTune":
        return None, []
    db = sqlite3.connect(f"file:{root / 'records.sqlite3'}?mode=ro", uri=True)
    try:
        events = [(k, json.loads(v)) for k, v in db.execute("SELECT kind,payload FROM events ORDER BY seq")]
        reflections = db.execute("SELECT COUNT(*) FROM reflections").fetchone()[0]
    finally:
        db.close()
    trials = read_jsonl(root / "trials.jsonl")
    lookup = {r["trial_id"]: r for r in trials}
    variant = manifest.get("ablation", "full")
    fits, cross_phase_links = [], 0
    for kind, item in events:
        if kind == "gp_fit":
            phase = item["path"].split("-gp-")[0]
            ids = item["training_ids"]
            missing = set(ids) - set(lookup)
            if missing:
                raise ValueError("GP training ID has no recorded measurement")
            old = sum(lookup[t]["phase_id"] != phase for t in ids)
            fits.append({"phase_id": phase, "training_count": len(ids), "cross_phase_rows": old})
            cross_phase_links += old
    roles = Counter()
    reflection_prompts = 0
    for path in sorted((root / "llm_calls").glob("*.json")):
        record = json.loads(path.read_text())
        roles[record.get("role", "unknown")] += 1
        if record.get("role") in ("selector", "pruner", "recommender"):
            if any(e.get("type") == "reflection" for e in record.get("payload", {}).get("experiences", [])):
                reflection_prompts += 1
    screens = [e for k, e in events if k == "prescreen"]
    predictions = []
    for trial in trials:
        pred = trial.get("prediction")
        if (not pred or not trial["complete"] or not trial["effective"]
                or trial["status"] != "measured" or trial.get("value", 0) <= 0):
            continue
        mean = pred["mean"]
        std = pred["observation_std"]
        target = -math.log(trial["value"])
        predictions.append({"method": root.name, "trial_id": trial["trial_id"], "phase_id": trial["phase_id"],
                            "kind": trial.get("metadata", {}).get("kind"),
                            "actual_p95_ms": trial["value"], "predicted_median_p95_ms": math.exp(-mean),
                            "interval_lower_ms": math.exp(-(mean + 1.96 * std)),
                            "interval_upper_ms": math.exp(-(mean - 1.96 * std)),
                            "covered": abs(target - mean) <= 1.96 * std,
                            "model_version": pred["model_version"]})
    if variant == "no_gp" and screens:
        raise ValueError("no_gp variant unexpectedly performed GP prescreening")
    if variant == "no_history" and cross_phase_links:
        raise ValueError("no_history variant trained on prior-phase records")
    if variant == "no_reflection" and (reflections or roles["reflection"] or reflection_prompts):
        raise ValueError("no_reflection variant unexpectedly used reflection")
    if bool(manifest["plan"].get("single_slice", False)) != (variant == "single_slice"):
        raise ValueError("Single-slice switch differs from declared variant")
    row = {"method": root.name, "ablation": variant, "source_revision": manifest.get("git_revision"),
           "single_slice": manifest["plan"].get("single_slice", False),
           "gp_fit_events": len(fits), "gp_training_max_rows": max((x["training_count"] for x in fits), default=0),
           "cross_phase_training_row_appearances": cross_phase_links,
           "a2_cross_phase_training_max_rows": max((x["cross_phase_rows"] for x in fits
                                                  if x["phase_id"] == "phase-3"), default=0),
           "prescreen_events": len(screens),
           "screened_candidate_appearances": sum(len(e["config_ids"]) for e in screens),
           "eligible_candidate_appearances": sum(len(e["eligible"]) for e in screens),
           "committed_reflections": reflections, "reflection_api_calls": roles["reflection"],
           "agent_prompts_containing_reflection": reflection_prompts,
           "premeasurement_predictions": len(predictions),
           "empirical_interval_coverage": (sum(p["covered"] for p in predictions) / len(predictions)
                                          if predictions else None)}
    return row, predictions


def audit(output):
    state = json.loads((output / "status.json").read_text())
    rows, predictions = [], []
    for job in state["completed"]:
        root = Path(job["output"]) / f"seed-{state['seed']}" / job["label"]
        row, ps = inspect_run(root)
        if row:
            rows.append(row)
            predictions.extend(ps)
    public = output / "figures_and_data"
    public.mkdir(exist_ok=True)
    if rows:
        write_csv(public / "mechanism_audit.csv", rows, list(rows[0]))
    if predictions:
        write_csv(public / "gp_premeasurement_predictions.csv", predictions, list(predictions[0]))
    result = {"completed_admtune_variants": len(rows), "database_calls_added": 0, "llm_calls_added": 0,
              "note": "Mechanism execution evidence, not causal benefit; repeated training rows are not new observations. "
                      "Prediction rows include selected configurations and verification repeats, not independent held-out data.",
              "rows": rows}
    atomic_json(public / "mechanism_audit.json", result)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    print(json.dumps(audit(p.parse_args().output), ensure_ascii=False, indent=2))
