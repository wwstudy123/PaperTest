from pathlib import Path
import sqlite3
import tempfile
import unittest

from admtune.cli import run_simulation
from admtune.memory import Store


class SimulationTests(unittest.TestCase):
    def test_aba_and_trial_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            summary = run_simulation(output=output, seed=0, trials=24)
            self.assertTrue(summary["simulated"])
            self.assertEqual(summary["budget"]["llm_calls"], 0)
            self.assertLessEqual(summary["budget"]["trial_slots_used"], 24)
            self.assertEqual(summary["detected_phases"], 3)
            self.assertEqual(len(summary["independent_verification_values"]), 3)
            store = Store(output / "records.sqlite3")
            records = store.trials()
            self.assertTrue(all(t.metadata["simulated"] for t in records))
            event_kinds = [row[0] for row in store.db.execute("SELECT kind FROM events ORDER BY seq")]
            self.assertEqual(event_kinds.count("trial_submitted"), len(records))
            for t in records:
                if t.prediction:
                    self.assertIn("model_version", t.prediction)
                    self.assertGreater(t.prediction["observation_std"], t.prediction["latent_std"])
            store.close()

    def test_no_surrogate_variant_is_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            summary = run_simulation(output=output, scenario="static", variant="no-surrogate", trials=12)
            store = Store(output / "records.sqlite3")
            self.assertTrue(all(t.prediction is None for t in store.trials()))
            self.assertLessEqual(summary["budget"]["trial_slots_used"], 12)
            store.close()

    def test_output_cannot_overwrite_a_previous_run(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileExistsError):
                run_simulation(output=Path(directory))


if __name__ == "__main__":
    unittest.main()
