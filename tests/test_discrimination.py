import unittest
from dataclasses import replace
from config import WorkloadSettings
from admtune.discrimination import evaluate, plan


class DiscriminationTests(unittest.TestCase):
    def rows(self, values):
        return [dict(distribution="uniform", valid=True, tps=v) for v in values]

    def test_frozen_plan_and_gate(self):
        self.assertEqual(len(plan()), 8)
        self.assertTrue(evaluate(self.rows([100, 120, 101, 125, 100]))["passed"])
        self.assertFalse(evaluate(self.rows([100, 102, 101, 103, 100]))["passed"])
        self.assertFalse(evaluate(self.rows([100, 130, 120, 150, 100]))["passed"])
        self.assertFalse(evaluate(self.rows([100, 130, 100]))["passed"])

    def test_no_invalid_observations_or_distribution_injection(self):
        rows = self.rows([100, 130, 100, 130, 100]); rows[2]["valid"] = False
        self.assertFalse(evaluate(rows)["passed"])
        w = WorkloadSettings(kind="sysbench", mode="read")
        self.assertEqual(w.rand_type, "special")
        self.assertEqual(replace(w, rand_type="uniform").rand_type, "uniform")
        with self.assertRaises(ValueError):
            replace(w, rand_type="uniform; exit")
