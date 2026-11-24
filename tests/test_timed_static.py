from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from admtune.agents import HeuristicBackend, Team
from admtune.continuous import Journal
from admtune.contracts import Environment
from admtune.timed_static import TimedLedger, TimedRun
from test_calibration import space
from test_continuous import ChargedFixture, FastGP, FixtureExecutor


class Clock:
    def __init__(self):
        self.value = 0.

    def __call__(self):
        return self.value


class TimedTests(unittest.TestCase):
    def test_final_verification_is_inside_time_budget_and_uses_a_real_slot(self):
        with tempfile.TemporaryDirectory() as tmp, patch("admtune.continuous.GaussianProcess", FastGP):
            root = Path(tmp); j = Journal(root / "state.sqlite3"); s = space(); clock = Clock()
            ledger = TimedLedger(j, 600, 105)
            ledger.clock = clock; ledger.started = ledger.session_started = 0.
            ex = FixtureExecutor(s); ex.estimated_trial_seconds = 105
            original = ex.measure
            def measure(config, context):
                clock.value += 80
                return original(config, context)
            ex.measure = measure
            r = TimedRun(j, ex, s, Environment("mysql", "test", "test", "test", "test", "test"),
                         root, ChargedFixture(ledger), ledger)
            result = r.run_budget()
            self.assertLessEqual(clock.value, 600)
            self.assertGreater(len(ex.calls), 2)
            self.assertEqual(len(ex.calls), j.count)
            self.assertEqual(j.store.trials()[-1].metadata["kind"], "verification")
            self.assertIsNotNone(result["final_verification_value"])
            j.audit(); j.store.close()

    def test_pruner_only_sees_selected_catalog_without_relaxing_validation(self):
        s = space()
        class Backend(HeuristicBackend):
            def complete(self, role, payload):
                if role == "selector":
                    return {"selected": ["innodb_buffer_pool_size"]}
                self.pruner_catalog = payload["catalog"]
                return super().complete(role, payload)
        backend = Backend()
        team = Team(s, backend)
        team.initialize([0.], [])
        self.assertEqual([k["name"] for k in backend.pruner_catalog], ["innodb_buffer_pool_size"])
        with self.assertRaises(ValueError):
            team.set_ranges({**team.ranges, "sort_buffer_size": {}})
