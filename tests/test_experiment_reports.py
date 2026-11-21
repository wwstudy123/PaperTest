from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from admtune.experiments import run_suite
from admtune.security import redact


class ReportTests(unittest.TestCase):
    def test_small_ablation_matrix_is_free_and_marked_synthetic(self):
        with tempfile.TemporaryDirectory() as folder:
            report = run_suite(
                Path(folder) / "suite", seeds=[0], variants=["full", "no-surrogate", "no-reflection"],
                scenario="static", trials=12,
            )
            self.assertEqual(report["failures"], [])
            self.assertEqual(len(report["runs"]), 3)
            for run in report["runs"]:
                self.assertTrue(run["simulated"])
                self.assertFalse(run["is_thesis_evidence"])
                self.assertEqual(run["llm_api_calls"], 0)
                self.assertLessEqual(run["trial_slots_used"], 12)
                self.assertTrue(run["target_misses_are_null_not_dropped"])

    def test_duplicate_seeds_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):
                run_suite(Path(folder) / "suite", seeds=[0, 0], variants=["full"], scenario="static", trials=12)

    def test_environment_credentials_redacted(self):
        with patch.dict("os.environ", {"ADMTUNE_MYSQL_PASSWORD": "secret-test-value"}):
            self.assertEqual(redact("failed with secret-test-value"), "failed with <redacted>")


if __name__ == "__main__":
    unittest.main()
