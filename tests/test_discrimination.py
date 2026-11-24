import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from config import WorkloadSettings
from config import load_app_config
from admtune.discrimination import evaluate, plan
from tools.run_benchmark import _run_sysbench


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

    def test_distribution_reaches_both_warmup_and_measurement_command(self):
        root = Path(__file__).resolve().parents[1]
        app = load_app_config(project_root=root, target_path=root / "configs/examples/target.yaml",
                              benchmark_path=root / "configs/examples/benchmark.yaml",
                              knobs_path=root / "configs/examples/knobs.yaml", dry_run_override=True)
        app.benchmark.workload.rand_type = "uniform"
        app.benchmark.workload.warmup_time = 15
        connector = Mock()
        connector.run.return_value = SimpleNamespace(stdout="fixture")
        with patch("tools.run_benchmark.resolve_env", return_value="fixture"), \
                patch("tools.run_benchmark._reset_runtime_stats_before_measurement"):
            _run_sysbench(connector, app.benchmark, app.target.active_db_settings)
        self.assertEqual(len(connector.run.call_args_list), 2)
        self.assertTrue(all("--rand-type=uniform" in c.args[0] for c in connector.run.call_args_list))
