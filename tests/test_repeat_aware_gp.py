from copy import deepcopy
import csv
import json
import math
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from admtune.contracts import Trial
from admtune.repeat_aware_gp import POLICY, RepeatAwareGP, repeat_noise
from admtune.space import Knob, Space
from admtune.surrogate import GaussianProcess


class FixedTrainingNoiseTests(unittest.TestCase):
    x = [[0., 0.], [0., 0.], [1., 0.], [1., 0.]]
    y = [0., .2, 1., 1.2]

    def test_floor_is_in_training_covariance_and_latent_posterior(self):
        base, noisy = GaussianProcess(), GaussianProcess()
        base.fit(self.x, self.y, optimize=False)
        noisy.fit(self.x, self.y, optimize=False, fixed_noise_log_std=.25)
        residual = math.exp(noisy.theta[2])
        expected = noisy.covariance(noisy.x, noisy.x) + np.eye(4)*(
            residual**2+(.25/noisy.y_scale)**2+1e-9)
        np.testing.assert_allclose(noisy.lower @ noisy.lower.T, expected, rtol=1e-12, atol=1e-12)
        a, b = base.predict([[0, 0]])[0], noisy.predict([[0, 0]])[0]
        self.assertGreater(b.latent_std, a.latent_std)
        self.assertGreater(abs(b.mean-a.mean), 1e-8)
        self.assertAlmostEqual(b.observation_std**2-b.latent_std**2,
                               (noisy.y_scale*residual)**2+.25**2, places=12)

    def test_fixed_noise_is_in_original_target_units_under_rescaling(self):
        for multiplier in (1, 10):
            model = GaussianProcess()
            model.fit(self.x, [y*multiplier for y in self.y],
                      optimize=False, fixed_noise_log_std=.25)
            pred = model.predict([[0, 0]])[0]
            residual_var = (model.y_scale*math.exp(model.theta[2]))**2
            self.assertAlmostEqual(pred.observation_std**2-pred.latent_std**2-residual_var,
                                   .25**2, places=12)

    def test_floor_larger_than_normalized_noise_bound_is_supported(self):
        model = GaussianProcess()
        model.fit([[0], [1]], [1, 1], optimize=True, fixed_noise_log_std=.5)
        p = model.predict([[0]])[0]
        self.assertGreaterEqual(p.observation_std, .5)
        self.assertTrue(math.isfinite(p.latent_std))

    def test_default_is_zero_floor_and_legacy_snapshot_has_no_new_fields(self):
        a, b = GaussianProcess(seed=42), GaussianProcess(seed=42)
        a.fit(self.x, self.y)
        b.fit(self.x, self.y, fixed_noise_log_std=0)
        self.assertEqual(a.snapshot(["1", "2", "3", "4"]), b.snapshot(["1", "2", "3", "4"]))
        self.assertNotIn("noise_policy", a.snapshot(["1", "2", "3", "4"]))

    def test_existing_real_parent_snapshot_prediction_is_unchanged(self):
        root = Path(__file__).resolve().parents[1]
        data = root/"docs/experiment_data/uncertainty_repeat_001"
        saved = json.loads((data/"models.json").read_text())["frozen_parent"]
        with (data/"frozen_parent_predictions.csv").open() as f:
            expected = next(r for r in csv.DictReader(f) if r["name"] == "Default")
        model = GaussianProcess.from_snapshot(saved)
        prediction = model.predict([saved["x"][0]])[0]
        # 同一旧快照在macOS/Linux的线性代数后端上可能有极小舍入差异。
        self.assertTrue(math.isclose(math.exp(prediction.mean), float(expected["predicted_objective"]),
                                     rel_tol=1e-10, abs_tol=1e-9))
        self.assertAlmostEqual(prediction.observation_std, float(expected["observation_log_std"]), places=12)

    def test_new_snapshot_roundtrip_and_noise_tamper_rejection(self):
        model = GaussianProcess()
        model.fit(self.x, self.y, fixed_noise_log_std=.2)
        saved = model.snapshot(["1", "2", "3", "4"])
        restored = GaussianProcess.from_snapshot(saved)
        self.assertEqual(model.predict([[0, 0]]), restored.predict([[0, 0]]))
        for key, value in (("parameterization", "unknown"), ("effective_noise_log_std", 99),
                           ("fixed_noise_log_std", -1)):
            altered = deepcopy(saved); altered["noise_policy"][key] = value
            with self.assertRaises(ValueError):
                GaussianProcess.from_snapshot(altered)

    def test_bad_fixed_noise_is_rejected(self):
        for value in (-1, True, float("nan"), float("inf"), 1e308):
            with self.assertRaises(ValueError):
                GaussianProcess().fit([[0], [1]], [1, 1], fixed_noise_log_std=value)


class RepeatAwareTests(unittest.TestCase):
    def setUp(self):
        self.space = Space([Knob("buffer", "integer", 1, 1, 10)])

    def rows(self, repeated=True):
        return [Trial(str(i), "p1", "env", "tps_per_p95:maximize",
                      {"buffer": i % 3+1 if repeated else i+1}, [0.],
                      "measured", math.exp(.1*(i % 3)+.02*(i//3)), True, True, True)
                for i in range(6)]

    def test_pooled_df_and_floor_activation_are_explicit(self):
        audit = repeat_noise(self.rows())
        self.assertEqual(audit["degrees_of_freedom"], 3)
        self.assertEqual(audit["repeat_groups"], 3)
        self.assertTrue(audit["floor_eligible"])
        self.assertAlmostEqual(audit["fixed_noise_log_std"], math.sqrt(.0002), places=12)
        unreplicated = repeat_noise(self.rows(False))
        self.assertEqual(unreplicated["fixed_noise_log_std"], 0)
        self.assertIsNone(unreplicated["estimated_log_std"])

    def test_context_and_phase_changes_are_not_repeats(self):
        for field, value in (("context", [1.]), ("phase_id", "p2")):
            rows = self.rows()
            for row in rows[3:]:
                setattr(row, field, value)
            self.assertEqual(repeat_noise(rows)["degrees_of_freedom"], 0)

    def test_failed_mixed_objective_and_duplicate_evidence_are_rejected(self):
        for field, value in (("status", "failed"), ("constraints_ok", False),
                             ("objective_key", "tps:maximize"), ("environment_key", "other"),
                             ("value", True), ("value", float("nan")), ("context", [float("inf")]),
                             ("trial_id", "1")):
            rows = self.rows(); setattr(rows[0], field, value)
            with self.assertRaises(ValueError):
                repeat_noise(rows)

    def test_every_update_reoptimizes_and_identical_history_is_noop(self):
        wrapper = RepeatAwareGP(self.space)
        rows = self.rows()
        self.assertTrue(wrapper.update(rows))
        self.assertTrue(wrapper.model.optimizer["optimized"])
        self.assertGreater(wrapper.model.fixed_noise_log_std, 0)
        self.assertFalse(wrapper.update(rows))
        self.assertEqual(wrapper.model.version, 1)
        rows.append(Trial("6", "p1", "env", "tps_per_p95:maximize", {"buffer": 1}, [0.],
                          "measured", math.exp(.08), True, True, True))
        self.assertTrue(wrapper.update(rows))
        self.assertEqual(wrapper.model.version, 2)
        self.assertTrue(wrapper.model.optimizer["optimized"])
        self.assertEqual(len(wrapper.model.x), 7)
        self.assertEqual(wrapper.audit["degrees_of_freedom"], 4)

    def test_first_pair_reoptimizes_even_below_floor_minimum_df(self):
        rows = self.rows(False); wrapper = RepeatAwareGP(self.space)
        wrapper.update(rows)
        rows.append(Trial("6", "p1", "env", "tps_per_p95:maximize", {"buffer": 1}, [0.],
                          "measured", 1.05, True, True, True))
        wrapper.update(rows)
        self.assertEqual(wrapper.audit["degrees_of_freedom"], 1)
        self.assertEqual(wrapper.model.fixed_noise_log_std, 0)
        self.assertTrue(wrapper.model.optimizer["optimized"])

    def test_uniform_foreign_objective_is_not_silently_modeled(self):
        rows = self.rows()
        for row in rows:
            row.objective_key = "p95_latency_ms:minimize"
        with self.assertRaises(ValueError):
            repeat_noise(rows)

    def test_zero_sse_with_sufficient_df_is_eligible_but_has_zero_floor(self):
        rows = self.rows()
        for row in rows:
            row.value = float(row.config["buffer"])
        wrapper = RepeatAwareGP(self.space)
        wrapper.update(rows)
        self.assertTrue(wrapper.audit["floor_eligible"])
        self.assertEqual(wrapper.model.fixed_noise_log_std, 0)
        restored = RepeatAwareGP.from_snapshot(self.space, wrapper.snapshot())
        self.assertFalse(restored.update(rows))

    def test_history_must_be_append_only_and_values_immutable(self):
        wrapper = RepeatAwareGP(self.space); rows = self.rows(); wrapper.update(rows)
        before = wrapper.snapshot()
        for altered in (list(reversed(rows)), rows[:5]):
            with self.assertRaises(ValueError):
                wrapper.update(altered)
        altered = deepcopy(rows); altered[0].value += .1
        with self.assertRaises(ValueError):
            wrapper.update(altered)
        self.assertEqual(before, wrapper.snapshot())

    def test_failed_fit_does_not_corrupt_previous_model(self):
        wrapper = RepeatAwareGP(self.space); rows = self.rows(); wrapper.update(rows)
        before = wrapper.snapshot()
        rows.append(Trial("6", "p1", "env", "tps_per_p95:maximize", {"buffer": 1}, [0.],
                          "measured", 1.05, True, True, True))
        with patch("admtune.repeat_aware_gp.GaussianProcess.fit", side_effect=RuntimeError("fixture")):
            with self.assertRaises(RuntimeError):
                wrapper.update(rows)
        self.assertEqual(before, wrapper.snapshot())

    def test_snapshot_restores_model_and_does_not_share_mutable_provenance(self):
        wrapper = RepeatAwareGP(self.space); rows = self.rows(); wrapper.update(rows)
        saved = wrapper.snapshot()
        restored = RepeatAwareGP.from_snapshot(self.space, saved)
        self.assertFalse(restored.update(rows))
        self.assertEqual(wrapper.model.predict([[0., 0.]]), restored.model.predict([[0., 0.]]))
        saved["training_ids"].append("bad")
        saved["gp"]["training_ids"].append("bad")
        saved["audit"]["training_ids"].append("bad")
        saved["policy"]["grouping"].append("bad")
        self.assertEqual(len(wrapper.training_ids), 6)
        self.assertEqual(len(restored.audit["training_ids"]), 6)
        self.assertNotIn("bad", POLICY["grouping"])

    def test_policy_space_noise_and_training_tampering_are_rejected(self):
        wrapper = RepeatAwareGP(self.space); rows = self.rows(); wrapper.update(rows)
        for key in ("policy", "space_key", "audit"):
            saved = wrapper.snapshot()
            if key == "policy":
                saved[key]["version"] = "different"
            elif key == "space_key":
                saved[key] = "different"
            else:
                saved[key]["fixed_noise_log_std"] = 0
            with self.assertRaises(ValueError):
                RepeatAwareGP.from_snapshot(self.space, saved)
        saved = wrapper.snapshot(); saved["gp"]["x"][0][0] += .001
        restored = RepeatAwareGP.from_snapshot(self.space, saved)
        with self.assertRaises(ValueError):
            restored.update(rows)
