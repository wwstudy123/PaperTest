from pathlib import Path
import json
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
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertIn("git_dirty", manifest)
            self.assertIn("working_source_sha256", manifest)
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

    def test_nonfinite_measurement_stops_phase_without_fake_gp_label(self):
        from admtune.agents import HeuristicBackend, Team
        from admtune.budget import Ledger, Limits
        from admtune.cli import demo_space
        from admtune.contracts import Environment, Objective
        from admtune.controller import Controller, Policy
        from admtune.execution import Measurement, SimulatedExecutor
        from admtune.reflection import Reflector
        from admtune.workload import PhaseDetector, WorkloadEncoder, Window
        class InvalidExecutor(SimulatedExecutor):
            def measure(self, config, context):
                return Measurement(float("nan"))
        with tempfile.TemporaryDirectory() as directory:
            store = Store(Path(directory) / "records.sqlite3")
            space, backend = demo_space(), HeuristicBackend()
            detector = PhaseDetector(
                WorkloadEncoder([Window(0, 1, [[1, 0]])]),
                change_threshold=1, stable_threshold=1,
            )
            controller = Controller(
                space=space, environment=Environment("s", "v", "h", "d", "s", "p"),
                objective=Objective(), detector=detector, executor=InvalidExecutor(space),
                team=Team(space, backend), reflector=Reflector(store, backend),
                store=store, ledger=Ledger(Limits()), policy=Policy(),
            )
            controller.begin_phase("p1", [0])
            self.assertFalse(controller.active)
            self.assertIsNone(store.trials()[0].value)
            self.assertEqual(store.trials()[0].status, "invalid_measurement")
            store.close()


if __name__ == "__main__":
    unittest.main()
