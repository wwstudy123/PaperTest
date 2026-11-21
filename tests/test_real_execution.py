from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from config import load_app_config
from models import BenchmarkResult
from admtune.contracts import Objective
from admtune.contracts import Environment
from admtune.execution import RecoveryFailed
from admtune.live import load_space, static_tune, main
from admtune.real_execution import AgenticDBExecutor


ROOT = Path(__file__).resolve().parents[1]


def complete_output(tps=100, duration=120, errors=0):
    return (
        "SQL statistics:\n"
        f"transactions: 12000 ({tps} per sec.)\n"
        f"ignored errors: {errors} (0.00 per sec.)\n"
        "General statistics:\n"
        f"total time: {duration}.000s\n"
        "Latency (ms):\n"
        "95th percentile: 10.00\n"
    )


class RealAdapterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        app = load_app_config(
            project_root=ROOT, target_path=ROOT / "configs/examples/target.yaml",
            benchmark_path=ROOT / "configs/examples/benchmark.yaml",
            knobs_path=ROOT / "configs/examples/knobs.yaml", dry_run_override=True,
        )
        self.space = load_space(ROOT / "configs/examples/knobs.yaml", memory_bytes=32 * 1024**3)
        self.executor = AgenticDBExecutor(
            app, self.space, Objective(), Path(self.temp.name), allow_mutations=True,
        )
        variables = self.executor.connector.fake_state["mysql_variables"]
        variables.update({
            "sort_buffer_size": 262144, "transaction_isolation": "REPEATABLE-READ",
            "performance_schema": "ON",
        })
        self.executor.preflight()
        self.before = self.executor.readback()

    def tearDown(self):
        self.executor.close()
        self.temp.cleanup()

    def benchmark(self, output=None):
        return BenchmarkResult(
            score=100, raw_output=output or complete_output(), tps=100,
            p95_latency_ms=10, primary_metric_name="tps", primary_metric_value=100,
        )

    def test_preflight_does_not_apply_settings(self):
        before = dict(self.executor.connector.fake_state["mysql_variables"])
        self.executor.allow_mutations = False
        self.executor.preflight()
        self.assertEqual(before, self.executor.connector.fake_state["mysql_variables"])
        with self.assertRaises(PermissionError):
            self.executor.measure(self.before, [0])

    def test_apply_readback_measure_restore(self):
        candidate = self.space.merge(self.before, {"innodb_io_capacity": 400})
        with patch("admtune.real_execution.run_benchmark", return_value=self.benchmark()):
            measured = self.executor.measure(candidate, [0])
        self.assertTrue(measured.complete)
        self.assertEqual(self.executor.readback(), candidate)
        self.executor.restore(self.before)
        self.assertEqual(self.executor.readback(), self.before)
        self.assertEqual(
            self.executor.connector.read_file(self.executor.settings.config_path, sudo=True),
            self.executor.initial.file_content,
        )

    def test_incomplete_benchmark_is_not_a_label(self):
        candidate = self.space.merge(self.before, {"innodb_io_capacity": 400})
        with patch("admtune.real_execution.run_benchmark", return_value=self.benchmark("transactions: 1 (100 per sec.)")):
            result = self.executor.measure(candidate, [0])
        self.assertFalse(result.complete)
        self.assertIsNone(result.value)
        self.assertEqual(self.executor.readback(), self.before)

    def test_readback_mismatch_restores(self):
        candidate = self.space.merge(self.before, {"innodb_io_capacity": 400})
        # Simulate an apply function that silently does nothing.
        with patch("admtune.real_execution.apply_db_config"):
            result = self.executor.measure(candidate, [0])
        self.assertEqual(result.status, "execution_failed")
        self.assertEqual(self.executor.readback(), self.before)

    def test_service_failure_and_failed_recovery_stop(self):
        candidate = self.space.merge(self.before, {"innodb_io_capacity": 400})
        def crash(*args, **kwargs):
            self.executor.connector.fake_state["service_running"] = False
            raise RuntimeError("simulated service failure")
        with patch("admtune.real_execution.run_benchmark", side_effect=crash):
            with self.assertRaises(RecoveryFailed):
                self.executor.measure(candidate, [0])

    def test_invariants_are_frozen(self):
        def unsafe(*args, **kwargs):
            self.executor.connector.fake_state["mysql_variables"]["sync_binlog"] = 0
        candidate = self.space.merge(self.before, {"innodb_io_capacity": 400})
        with patch("admtune.real_execution.apply_db_config", side_effect=unsafe):
            # Restoration must not silently claim success after an external invariant change.
            with self.assertRaises(RecoveryFailed):
                self.executor.measure(candidate, [0])

    def test_p95_is_a_constraint_not_a_changed_objective(self):
        result = self.benchmark()
        result.p95_latency_ms = 2000
        with patch("admtune.real_execution.run_benchmark", return_value=result):
            measured = self.executor.measure(self.before, [0])
        self.assertEqual(measured.value, 100)
        self.assertFalse(measured.constraints_ok)

    def test_ignored_errors_and_short_runs_rejected(self):
        self.assertFalse(AgenticDBExecutor.complete_sysbench(complete_output(errors=1), 120))
        self.assertFalse(AgenticDBExecutor.complete_sysbench(complete_output(duration=10), 120))
        self.assertTrue(AgenticDBExecutor.complete_sysbench(complete_output(), 120))

    def test_restore_requires_saved_target(self):
        with self.assertRaises(RecoveryFailed):
            self.executor.restore(self.space.merge(self.before, {"innodb_io_capacity": 400}))

    def test_initial_configuration_restoration(self):
        candidate = self.space.merge(self.before, {"innodb_io_capacity": 400})
        with patch("admtune.real_execution.run_benchmark", return_value=self.benchmark()):
            self.executor.measure(candidate, [0])
        self.executor.restore_initial()
        self.assertEqual(self.executor.readback(), self.before)

    def test_static_controller_with_fake_agenticdb_adapter(self):
        output = Path(self.temp.name) / "static"
        output.mkdir()
        with patch("admtune.real_execution.run_benchmark", return_value=self.benchmark()):
            summary = static_tune(
                self.executor, self.space, Environment("mysql", "fixture", "h", "d", "s", "p"),
                output, seed=0, trials=12, wall_seconds=6000,
                backend_kind="heuristic", llm_calls=0, llm_tokens=0,
            )
        self.assertTrue(summary["simulated"])
        self.assertLessEqual(summary["budget"]["trial_slots_used"], 12)
        self.assertEqual(len(summary["independent_verification_values"]), 3)

    def test_cli_refuses_implicit_database_mutations(self):
        import sys
        with patch.object(sys, "argv", [
            "live", "pilot", "--target", "unused", "--benchmark", "unused",
            "--knobs", "unused", "--output", "unused",
        ]):
            with self.assertRaises(SystemExit) as exc:
                main()
        self.assertEqual(exc.exception.code, 2)

    def test_cli_refuses_implicit_paid_calls(self):
        import sys
        with patch.object(sys, "argv", [
            "live", "tune-static", "--target", "unused", "--benchmark", "unused",
            "--knobs", "unused", "--output", "unused", "--dedicated-instance", "--backend", "llm",
        ]):
            with self.assertRaises(SystemExit) as exc:
                main()
        self.assertEqual(exc.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
