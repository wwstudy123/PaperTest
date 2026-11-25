"""核验独立零LLM顺序预测：逐点重建训练集、协方差噪声与测量前封存证据。"""
from dataclasses import asdict
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
from admtune.contracts import Trial
from admtune.gp_prediction_audit import AUDIT_POLICY
from admtune.repeat_aware_gp import record_fingerprint, repeat_noise
from admtune.space import Knob, Space
from admtune.surrogate import GaussianProcess
from admtune.uncertainty_calibration import interval_row

sys.argv = [sys.argv[0], "--run-id", "gp-noise-validation-001", *sys.argv[1:]]
state = runpy.run_path(str(ROOT/"docs/analysis/analyze_concurrency_probe.py"))
records, manifest, out, p = (state[k] for k in ("records", "manifest", "out", "p"))
assert manifest["gp_prediction_audit"] == AUDIT_POLICY
space = Space([Knob(**spec) for spec in manifest["space"]])
# 编码按公开目录重建；完整资源约束Space标识与实机preflight直接对照。
forecasts, parameters, sealed_files = [], [], []
previous_submit = 0
for record in records[:6]:
    assert record["gp_prediction_audit"] is None
for record in records[6:]:
    seq = record["seq"]
    sealed = json.loads((p/f"gp_prediction_audit/before-{seq:06d}.json").read_text())
    assert sealed == record["gp_prediction_audit"] and sealed["policy"] == AUDIT_POLICY
    assert sealed["next_seq"] == seq and sealed["config"] == record["config"]
    assert sealed["training_sequences"] == list(range(1, seq))
    assert previous_submit < sealed["submitted_elapsed_seconds"] < record["elapsed_seconds"]-60
    previous_submit = sealed["submitted_elapsed_seconds"]
    assert "actual_objective" not in sealed
    past = [Trial(f"repeat-{r['seq']}", "repeat-audit", sealed["environment_key"],
                  "tps_per_p95:maximize", r["config"], [0.], r["measurement"]["status"],
                  round(r["joint_value"], 6), r["measurement"]["complete"],
                  r["measurement"]["effective"], r["measurement"]["constraints_ok"])
            for r in records[:seq-1]]
    ids = [t.trial_id for t in past]
    estimated = repeat_noise(past)
    for name in ("legacy", "repeat_aware"):
        entry = sealed["models"][name]
        if name == "repeat_aware":
            wrapper = entry["snapshot"]
            assert wrapper["space_key"] == manifest["preflight"]["space_key"]
            assert wrapper["training_ids"] == ids
            assert wrapper["record_fingerprints"] == [record_fingerprint(t) for t in past]
            for key in ("training_ids", "repeat_groups", "degrees_of_freedom", "floor_eligible"):
                assert wrapper["audit"][key] == estimated[key]
            for key in ("sse", "estimated_log_std", "fixed_noise_log_std"):
                assert math.isclose(wrapper["audit"][key], estimated[key], rel_tol=1e-9, abs_tol=1e-12)
            saved = wrapper["gp"]
            assert saved["optimizer"]["optimized"]
            fixed = estimated["fixed_noise_log_std"]
        else:
            saved, fixed = entry["snapshot"], 0.0
            assert saved["optimizer"]["optimized"] == (seq == 7)
        assert saved["training_ids"] == ids and saved["version"] == seq-6
        assert len(saved["x"]) == len(saved["target"]) == len(past)
        for t, x, y in zip(past, saved["x"], saved["target"]):
            expected = space.encode(t.config)+t.context
            assert len(x) == len(expected) and all(abs(a-b) < 1e-12 for a, b in zip(x, expected))
            assert math.isclose(y, math.log(t.value), abs_tol=1e-12)
        model = GaussianProcess.from_snapshot(saved)
        assert math.isclose(model.fixed_noise_log_std, fixed, abs_tol=1e-12)
        pred = model.predict([space.encode(record["config"])+[0.]])[0]
        for key in ("mean", "latent_std", "observation_std"):
            assert math.isclose(asdict(pred)[key], entry["prediction"][key], rel_tol=1e-6, abs_tol=1e-8)
        actual = round(record["joint_value"], 6)
        forecasts.append({"trial": seq, "name": record["name"], "model": name,
                          "training_n": len(past), **interval_row(pred, actual)})
        residual_log = saved["hyperparameters"]["noise_std"]*saved["target_scale"]
        parameters.append({"trial": seq, "model": name, "training_n": len(past),
                           **saved["hyperparameters"], "fixed_noise_log_std": fixed,
                           "residual_noise_log_std": residual_log,
                           "effective_noise_log_std": math.hypot(residual_log, fixed),
                           "repeat_degrees_of_freedom": estimated["degrees_of_freedom"],
                           "optimized": saved["optimizer"]["optimized"]})
    sealed_files.append(sealed)
metrics = []
for name in ("legacy", "repeat_aware"):
    rows = [r for r in forecasts if r["model"] == name]
    metrics.append({"model": name, "n": len(rows), "covered": sum(r["covered"] for r in rows),
                    "mape_pct": statistics.mean(r["absolute_percent_error"] for r in rows),
                    "mean_interval_width": statistics.mean(r["upper"]-r["lower"] for r in rows)})
for name, rows in (("predictions.csv", forecasts), ("gp_parameters.csv", parameters),
                   ("metrics.csv", metrics)):
    with (out/name).open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader(); w.writerows(rows)
(out/"sealed_forecasts.json").write_text(json.dumps(sealed_files, indent=2)+"\n")
(out/"space.json").write_text(json.dumps(manifest["space"], ensure_ascii=False, indent=2)+"\n")
verification = json.loads((out/"verification.json").read_text())
verification.update({"prospective_submissions_verified": True,
                     "fixed_configuration_audit_not_search": True,
                     "test_sequences": [7, 8, 9], "test_observations": 3,
                     "past_test_observation_may_train_next_prediction": True, "metrics": metrics})
(out/"verification.json").write_text(json.dumps(verification, ensure_ascii=False, indent=2)+"\n")
(out/"checksums.json").write_text(json.dumps({
    f.name: hashlib.sha256(f.read_bytes()).hexdigest()
    for f in out.iterdir() if f.is_file() and f.name != "checksums.json"}, indent=2)+"\n")
print(json.dumps({"verified": True, "metrics": metrics}, indent=2))
