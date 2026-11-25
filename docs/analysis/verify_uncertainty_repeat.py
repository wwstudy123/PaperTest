"""离线重建九次重复及固定6/3时间切分；不调用模型接口，不改在线GP。"""
from pathlib import Path
import csv
import hashlib
import json
import math
import runpy
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from admtune.space import Knob, Space
from admtune.surrogate import GaussianProcess
from admtune.uncertainty_calibration import (
    pooled_log_variance, refit_first_six, with_noise_floor, interval_row)

sys.argv = [sys.argv[0], "--run-id", "uncertainty-repeat-001", *sys.argv[1:]]
state = runpy.run_path(str(ROOT/"docs/analysis/analyze_concurrency_probe.py"))
records, manifest, out, dest = (state[k] for k in ("records", "manifest", "out", "dest"))


def load(path):
    return json.loads(path.read_text())


def save_csv(name, rows):
    with (out/name).open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


parent = dest/"runs/uncertainty-parent"
pm, ps = load(parent/"manifest.json"), load(parent/"summary.json")
assert ps["closed_loop_verified"] and ps["completed_trials"] == 7
assert pm["objective"] == "tps_per_p95:maximize" and pm["benchmark"] == manifest["benchmark"]
assert pm["space"] == manifest["space"]
pr = [json.loads(line) for line in (parent/"trials.jsonl").read_text().splitlines()]
byid = {r["trial_id"]: r for r in pr}
space = Space([Knob(**spec) for spec in manifest["space"]])
snapshot = max((load(p) for p in (parent/"gp_snapshots").glob("*.json")), key=lambda s: s["version"])
assert len(snapshot["training_ids"]) == 7
for ident, x, y in zip(snapshot["training_ids"], snapshot["x"], snapshot["target"]):
    r = byid[ident]
    assert r["status"] == "measured" and r["complete"] and r["effective"] and r["constraints_ok"]
    assert r["value"] == round(r["metadata"]["raw_tps"]/r["metadata"]["p95_latency_ms"], 6)
    expected = space.encode(r["config"])+r["context"]
    assert len(x) == len(expected) and all(abs(a-b) < 1e-12 for a, b in zip(x, expected))
    assert math.isclose(y, math.log(r["value"]), abs_tol=1e-12)
frozen = GaussianProcess.from_snapshot(snapshot)
points = [{"seq": r["seq"], "name": r["name"], "config": r["config"],
           "objective_value": round(r["joint_value"], 6)} for r in records]
training, holdout = points[:6], points[6:]
assert [r["seq"] for r in holdout] == [7, 8, 9]
noise = pooled_log_variance(training)
refit = refit_first_six(space, training)
new_snapshot = refit.snapshot([f"repeat-{r['seq']}" for r in training])
rows, frozen_rows = [], []
for point in points:
    x = space.encode(point["config"])+[0.0]
    frozen_rows.append({"trial": point["seq"], "name": point["name"],
                        **interval_row(frozen.predict([x])[0], point["objective_value"])})
for point in holdout:
    x = space.encode(point["config"])+[0.0]
    for name, model in (("Frozen7", frozen), ("Refit6", refit)):
        pred = model.predict([x])[0]
        for suffix, adjusted in (("", pred), ("+floor", with_noise_floor(pred, noise["variance"]))):
            rows.append({"trial": point["seq"], "name": point["name"], "model": name+suffix,
                         "model_training_n": len(model.x), "floor_training_n": 6 if suffix else 0,
                         **interval_row(adjusted, point["objective_value"])})
groups = []
for name in ("Frozen7", "Frozen7+floor", "Refit6", "Refit6+floor"):
    data = [r for r in rows if r["model"] == name]
    groups.append({"model": name, "holdout_n": 3, "covered": sum(r["covered"] for r in data),
                   "coverage": statistics.mean(r["covered"] for r in data),
                   "mape_pct": statistics.mean(r["absolute_percent_error"] for r in data),
                   "mean_interval_width": statistics.mean(r["upper"]-r["lower"] for r in data)})
for base in ("Frozen7", "Refit6"):
    for point in holdout:
        a, b = [next(r for r in rows if r["model"] == m and r["trial"] == point["seq"])
                for m in (base, base+"+floor")]
        assert a["predicted_objective"] == b["predicted_objective"]
        assert b["lower"] <= a["lower"]+1e-12 and b["upper"] >= a["upper"]-1e-12
save_csv("holdout_predictions.csv", rows)
save_csv("holdout_metrics.csv", groups)
save_csv("frozen_parent_predictions.csv", frozen_rows)
save_csv("canonical_targets.csv", [
    {**{k: r[k] for k in ("seq", "name", "objective_value")},
     "split": "calibration" if r["seq"] <= 6 else "holdout"} for r in points])
variability = []
for name in manifest["case_names"]:
    values = [r["objective_value"] for r in points if r["name"] == name]
    train_values = [math.log(r["objective_value"]) for r in training if r["name"] == name]
    variability.append({
        "name": name, "descriptive_n": 3, "calibration_n": 2,
        "calibration_log_std": statistics.stdev(train_values),
        "all_three_mean_objective": statistics.mean(values),
        "all_three_sample_std": statistics.stdev(values),
        "all_three_cv_pct": 100*statistics.stdev(values)/statistics.mean(values),
        "all_three_min": min(values), "all_three_max": max(values),
        "all_three_is_descriptive_not_used_for_fit": True})
save_csv("repeat_variability.csv", variability)
(out/"models.json").write_text(json.dumps({
    "frozen_parent": snapshot, "refit_first_six": new_snapshot,
    "pooled_log_noise": noise, "holdout_sequences": [7, 8, 9],
    "floor_changes_only_observation_variance": True,
    "offline_time_split_not_online_prediction": True}, indent=2)+"\n")
verification = load(out/"verification.json")
verification.update({
    "protocol": "UNCERTAINTY_REPEAT_PROTOCOL_20261125",
    "calibration_sequences": list(range(1, 7)), "holdout_sequences": [7, 8, 9],
    "pooled_log_noise": noise, "holdout_metrics": groups,
    "frozen_parent_training_n": 7, "refit_training_n": 6,
    "refit_hyperparameters": new_snapshot["hyperparameters"],
    "refit_optimizer": new_snapshot["optimizer"],
    "frozen_noise_log_std": snapshot["hyperparameters"]["noise_std"]*snapshot["target_scale"],
    "refit_noise_log_std": new_snapshot["hyperparameters"]["noise_std"]*new_snapshot["target_scale"],
    "online_model_changed": False,
    "offline_time_split_not_online_prediction": True})
(out/"verification.json").write_text(json.dumps(verification, ensure_ascii=False, indent=2)+"\n")
(out/"checksums.json").write_text(json.dumps({
    p.name: hashlib.sha256(p.read_bytes()).hexdigest()
    for p in out.iterdir() if p.is_file() and p.name != "checksums.json"}, indent=2)+"\n")
print(json.dumps({"noise": noise, "holdout_metrics": groups}, ensure_ascii=False, indent=2))
