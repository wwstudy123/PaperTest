import math
import unittest
from copy import deepcopy

from admtune.space import Knob, Space
from admtune.surrogate import Prediction
from admtune.uncertainty_calibration import (
    pooled_log_variance, refit_first_six, with_noise_floor, interval_row)


class UncertaintyTests(unittest.TestCase):
    def rows(self):
        return [{"seq": i+1, "config": {"buffer": i % 3+1},
                 "objective_value": math.exp((i % 3)+2*(i//3))} for i in range(6)]

    def test_pooled_variance_uses_within_config_not_between_config_signal(self):
        result = pooled_log_variance(self.rows())
        self.assertAlmostEqual(result["variance"], 2)
        self.assertEqual(result["degrees_of_freedom"], 3)
        self.assertEqual(result["training_sequences"], list(range(1, 7)))

    def test_holdout_and_incomplete_or_invalid_training_are_rejected(self):
        for rows in (self.rows()[:5], self.rows()+[{"seq": 7}],
                     [{**r, "seq": r["seq"]+1} for r in self.rows()]):
            with self.assertRaises(ValueError):
                pooled_log_variance(rows)
        for value in (0, -1, float("nan"), True):
            rows = self.rows(); rows[0]["objective_value"] = value
            with self.assertRaises(ValueError):
                pooled_log_variance(rows)
        rows = self.rows(); rows[0]["config"] = {"buffer": 4}
        with self.assertRaises(ValueError):
            pooled_log_variance(rows)

    def test_zero_repeat_difference_does_not_invent_a_floor(self):
        rows = self.rows()
        for row in rows:
            row["objective_value"] = float(row["config"]["buffer"])
        self.assertEqual(pooled_log_variance(rows)["variance"], 0)
        p = Prediction(1, .1, .2, 3)
        self.assertEqual(with_noise_floor(p, 0), p)

    def test_floor_never_changes_mean_latent_or_model_version(self):
        p = Prediction(1, .1, .2, 3)
        adjusted = with_noise_floor(p, .25)
        self.assertEqual((adjusted.mean, adjusted.latent_std, adjusted.model_version), (1, .1, 3))
        self.assertAlmostEqual(adjusted.observation_std**2, .26)
        self.assertGreater(adjusted.observation_std, p.observation_std)
        self.assertEqual(p.observation_std, .2)

    def test_invalid_variance_or_prediction_is_not_silently_accepted(self):
        for variance in (-1, float("nan"), float("inf"), True):
            with self.assertRaises(ValueError):
                with_noise_floor(Prediction(1, .1, .2, 1), variance)
        with self.assertRaises(ValueError):
            with_noise_floor(Prediction(1, .3, .2, 1), .1)

    def test_fit_has_exactly_six_inputs_and_does_not_mutate_data(self):
        s = Space([Knob("buffer", "integer", 1, 1, 3)])
        rows = self.rows(); before = deepcopy(rows)
        model = refit_first_six(s, rows)
        self.assertEqual(len(model.x), 6)
        self.assertEqual(rows, before)
        self.assertTrue(model.optimizer["optimized"])
        with self.assertRaises(ValueError):
            refit_first_six(s, rows+[{"seq": 7}])

    def test_observation_interval_and_error_are_not_latent_interval(self):
        p = Prediction(math.log(100), .01, .2, 1)
        row = interval_row(p, 120)
        self.assertTrue(row["covered"])
        self.assertAlmostEqual(row["absolute_percent_error"], 100/6)
        self.assertGreater(row["upper"], 120)
