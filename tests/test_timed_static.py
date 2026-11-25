from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from admtune.agents import HeuristicBackend, Team, PROMPTS
from admtune.space import Knob, Space
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

    def test_numeric_special_only_null_is_rejected_with_actionable_error(self):
        s = Space([Knob("innodb_thread_concurrency", "integer", 0, 1, 32, special=(0,))])
        team = Team(s, HeuristicBackend())
        team.selected = ["innodb_thread_concurrency"]
        for low, high in [(None, None), (True, 8), (1, False)]:
            with self.assertRaisesRegex(ValueError, "innodb_thread_concurrency: Numeric preferred bounds"):
                team.set_ranges({"innodb_thread_concurrency": {
                    "lower": low, "upper": high, "choices": [], "special": [0]}})
            self.assertEqual(team.ranges, {})
        team.set_ranges({"innodb_thread_concurrency": {
            "lower": 1, "upper": 32, "choices": [], "special": [0]}})
        self.assertEqual(team.ranges["innodb_thread_concurrency"]["special"], [0])
        self.assertIn("仅特殊值范围", PROMPTS["pruner"])

    def test_pruner_retry_reuses_valid_selection_but_never_repairs_llm_numbers(self):
        s = space()
        class Backend(HeuristicBackend):
            def __init__(self):
                super().__init__()
                self.roles, self.contract = [], None
            def complete(self, role, payload):
                self.roles.append(role)
                out = super().complete(role, payload)
                if role == "pruner":
                    self.contract = payload["numeric_range_contract"]
                    if self.roles.count("pruner") == 1:
                        name = next(iter(out["ranges"]))
                        out["ranges"][name]["lower"] = None
                return out
        backend = Backend()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); j = Journal(root / "state.sqlite3")
            ledger = TimedLedger(j, 600, 105)
            run = TimedRun(j, FixtureExecutor(s), s,
                           Environment("mysql", "test", "test", "test", "test", "test"),
                           root, backend, ledger)
            run.replenish(2)
            self.assertEqual(backend.roles, ["selector", "pruner", "pruner", "recommender"])
            self.assertTrue(backend.contract)
            self.assertTrue(all(v["null_allowed"] is False for v in backend.contract.values()))
            self.assertTrue(run.state["pool"])
            self.assertIn("Numeric preferred bounds", run.team.feedback["reason"])
            j.store.close()

    def test_second_invalid_pruner_still_stops_without_fallback(self):
        class Invalid(HeuristicBackend):
            def __init__(self):
                super().__init__(); self.roles = []
            def complete(self, role, payload):
                self.roles.append(role)
                out = super().complete(role, payload)
                if role == "pruner":
                    out["ranges"][next(iter(out["ranges"]))]["lower"] = None
                return out
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); j = Journal(root / "state.sqlite3"); s = space()
            ledger = TimedLedger(j, 600, 105); backend = Invalid()
            run = TimedRun(j, FixtureExecutor(s), s,
                           Environment("mysql", "test", "test", "test", "test", "test"),
                           root, backend, ledger)
            with self.assertRaisesRegex(ValueError, "Numeric preferred bounds"):
                run.replenish(2)
            self.assertEqual(backend.roles, ["selector", "pruner", "pruner"])
            self.assertEqual(j.count, 0)
            self.assertEqual(run.team.ranges, {})
            j.store.close()
