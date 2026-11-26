import math
import unittest
from admtune.controller import Policy
from admtune.online_dynamic_calibration import evaluate, plan
from admtune.replay import ReplayPlan, make_schedule
from admtune.surrogate import GaussianProcess
from test_calibration import space


class OnlineDynamicTests(unittest.TestCase):
    def test_six_probe_plan_changes_one_parameter_and_is_readonly(self):
        s = space(); specs = plan(s, s.defaults)
        self.assertEqual(len(specs), 6)
        for r in specs:
            changed = [k for k in r["config"] if r["config"][k] != s.defaults[k]]
            self.assertEqual(len(changed), int(r["kind"] == "candidate"))
            self.assertEqual(r["read_percent"], 100)

    def test_signal_gate_and_noise_do_not_depend_on_candidate_variation(self):
        records = [{"block": b, "valid": True, "value_p95_ms": y}
                   for b in ("scan", "sort") for y in (100, 75, 104)]
        result = evaluate(records)
        self.assertTrue(result["passed"])
        self.assertEqual(result["repeat_degrees_of_freedom"], 2)
        self.assertGreaterEqual(result["fixed_noise_log_std"], .02)
        records[1]["value_p95_ms"] = 50
        self.assertEqual(evaluate(records)["fixed_noise_log_std"], result["fixed_noise_log_std"])
        records[-1]["value_p95_ms"] = 150
        self.assertFalse(evaluate(records)["passed"])

    def test_noise_floor_is_training_covariance_not_only_plot_interval(self):
        floor = Policy(fixed_noise_log_std=.03).fixed_noise_log_std
        g = GaussianProcess()
        g.fit([[0.], [.5], [1.]], [1., 2., 1.], fixed_noise_log_std=floor)
        self.assertEqual(g.fixed_noise_log_std, floor)
        self.assertGreaterEqual(g.predict([[0.]])[0].observation_std, floor)
        with self.assertRaises(ValueError):
            Policy(fixed_noise_log_std=float("nan"))

    def test_common_seed_and_readonly_aba_schedule(self):
        kwargs = dict(phase_seconds=600, table_size=2000000, read_percent=100,
                      query_profile="range-mixed", range_rows=32768, top_k=8192)
        a = ReplayPlan(seed=42, **kwargs); b = ReplayPlan(seed=42+0, **kwargs)
        self.assertEqual(make_schedule(a), make_schedule(b))
        rs = make_schedule(a)
        self.assertEqual(len(rs), 18000)
        self.assertTrue(all(r.operation == "read" for r in rs))
        self.assertEqual(sum(r.top_k == 0 for r in rs[:6000]), 5400)
        self.assertEqual(sum(r.top_k == 0 for r in rs[6000:12000]), 600)


if __name__ == "__main__":
    unittest.main()
