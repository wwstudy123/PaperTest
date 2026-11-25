from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from admtune.composite_static import (
    POLICY, CompositeExecutor, CompositeLedger, CompositeRun, pilot_limits, runtime_policy)
from admtune.continuous import Journal
from admtune.contracts import Environment, fingerprint
from admtune.execution import Measurement
from admtune.repeat_aware_gp import repeat_noise
from test_calibration import space
from test_composite_static import ScriptedBackend
from test_continuous import FixtureExecutor


class CompositePolicyTests(unittest.TestCase):
    def make(self, root, mode="repeat-aware", repeats=3):
        s = space(); raw = FixtureExecutor(s); raw.run_dir = root
        original = raw.measure
        def measured(config, context):
            result = original(config, context)
            return replace(result, value=result.value*(1+.02*(len(raw.calls)-1)))
        raw.measure = measured
        executor = CompositeExecutor(raw); journal = Journal(root/"records.sqlite3")
        ledger = CompositeLedger(journal, pilot_limits(repeats), 0)
        backend = ScriptedBackend(ledger)
        env = Environment("mysql", "fixture", "fixture", "fixture", "fixture", "fixture")
        try:
            runner = CompositeRun(journal, executor, s, env, root, backend, ledger,
                                  gp_policy=mode, initial_noise_repeats=repeats)
        except Exception:
            journal.store.close()
            raise
        return journal, runner, raw, backend

    def test_default_policy_and_limits_remain_legacy(self):
        self.assertEqual(runtime_policy(), POLICY)
        self.assertEqual(pilot_limits().trials, 8)
        self.assertEqual(pilot_limits().llm_calls, 8)
        self.assertEqual(pilot_limits(3).trials, 10)
        self.assertEqual(pilot_limits(3).llm_calls, 6)
        self.assertEqual(pilot_limits(3).llm_tokens, 24000)
        self.assertNotEqual(fingerprint(runtime_policy("legacy", 3)),
                            fingerprint(runtime_policy("repeat-aware", 3)))
        p = runtime_policy("repeat-aware", 3)
        p["gp_training"]["grouping"].append("bad")
        self.assertNotIn("bad", runtime_policy("repeat-aware", 3)["gp_training"]["grouping"])
        for mode, repeats in (("unknown", 0), ("legacy", 2), ("repeat-aware", True)):
            with self.assertRaises(ValueError):
                runtime_policy(mode, repeats)

    def test_three_predeclared_repeats_are_charged_and_never_promote(self):
        with tempfile.TemporaryDirectory() as tmp:
            j, runner, raw, backend = self.make(Path(tmp))
            for _ in range(4):
                runner.step()
            rows = j.store.trials()
            self.assertEqual([t.metadata["kind"] for t in rows],
                             ["baseline"]+["noise_calibration"]*3)
            self.assertTrue(all(t.config == rows[0].config for t in rows))
            self.assertGreater(rows[-1].value, rows[0].value*1.01)
            self.assertEqual(runner.state["baseline"], rows[0].value)
            self.assertEqual(runner.state["best_value"], rows[0].value)
            self.assertFalse(any(t.metadata["accepted"] for t in rows))
            self.assertEqual(runner.ledger.trials, 4)
            self.assertEqual(backend.payloads, [])
            self.assertEqual(raw.readback(), runner.space.defaults)
            j.store.close()

    def test_fitted_noise_and_checkpoint_use_actual_calibration_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            j, runner, _, _ = self.make(Path(tmp))
            for _ in range(6):
                runner.step()
            runner.fit()
            self.assertEqual(runner.repeat_gp.audit, repeat_noise(j.store.trials()))
            self.assertEqual(runner.repeat_gp.audit["degrees_of_freedom"], 3)
            self.assertGreater(runner.gp.fixed_noise_log_std, 0)
            self.assertTrue(runner.state["gp"]["optimizer"]["optimized"])
            version = runner.gp.version
            runner.fit()
            self.assertEqual(runner.gp.version, version)
            state = runner.journal.get("state")
            self.assertEqual(state["repeat_gp"]["gp"], state["gp"])
            j.store.close()
            j, restored, _, _ = self.make(Path(tmp))
            self.assertEqual(restored.gp.version, version)
            restored.fit()
            self.assertEqual(restored.gp.version, version)
            j.store.close()

    def test_existing_journal_cannot_switch_model_or_repeat_schedule(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); j, runner, _, _ = self.make(root, "legacy", 0)
            runner.step(); j.store.close()
            for mode, repeats in (("repeat-aware", 0), ("legacy", 3)):
                with self.assertRaisesRegex(ValueError, "change GP or calibration"):
                    self.make(root, mode, repeats)

    def test_final_verification_is_not_overridden_by_calibration_or_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            j, runner, _, _ = self.make(Path(tmp)); runner.step()
            runner.finalizing = True
            for seq in (2, 10):
                self.assertEqual(runner.choose(seq)[0], "verification")
            j.store.close()

    def test_invalid_calibration_stops_without_llm_or_extra_trial(self):
        with tempfile.TemporaryDirectory() as tmp:
            j, runner, raw, backend = self.make(Path(tmp)); runner.step()
            raw.measure = lambda *_: Measurement(None, "execution_failed", False, False, False)
            with self.assertRaisesRegex(RuntimeError, "Invalid predeclared calibration"):
                runner.step()
            self.assertEqual(j.count, 2)
            self.assertEqual(backend.payloads, [])
            self.assertEqual(raw.readback(), runner.space.defaults)
            j.audit(); j.store.close()

    def test_legacy_with_same_calibration_schedule_keeps_old_fit_cadence(self):
        with tempfile.TemporaryDirectory() as tmp:
            j, runner, _, _ = self.make(Path(tmp), "legacy", 3)
            for _ in range(6):
                runner.step()
            runner.fit()
            self.assertNotIn("noise_policy", runner.state["gp"])
            runner.finalizing = True
            runner.step(); runner.fit()
            self.assertFalse(runner.state["gp"]["optimizer"]["optimized"])
            j.store.close()

    def test_bounded_pilot_integrates_model_real_slots_and_reflection_fixture(self):
        with tempfile.TemporaryDirectory() as tmp:
            j, runner, raw, _ = self.make(Path(tmp))
            result = runner.run_pilot()
            rows = j.store.trials()
            self.assertLessEqual(len(rows), 10)
            self.assertGreaterEqual(len(rows), 7)
            self.assertEqual(rows[-1].metadata["kind"], "verification")
            self.assertNotIn("incumbent_repeat", [t.metadata["kind"] for t in rows])
            self.assertEqual(result["gp_policy"], "repeat-aware")
            self.assertTrue(result["final_reflection_committed"])
            self.assertGreaterEqual(result["repeat_noise_audit"]["degrees_of_freedom"], 3)
            self.assertTrue(runner.state["gp"]["optimizer"]["optimized"])
            self.assertEqual(raw.readback(), runner.space.defaults)
            j.audit(); j.store.close()

    def test_tenth_final_slot_reflects_once_using_final_reservation(self):
        with tempfile.TemporaryDirectory() as tmp:
            j, runner, _, backend = self.make(Path(tmp))
            original = runner.choose
            def choose(seq):
                if seq <= 4 or runner.finalizing:
                    return original(seq)
                return "startup", runner.space.merge(runner.state["initial"], {
                    "innodb_io_capacity": 300+seq*100}), {"rule": "explicit_fixture"}
            runner.choose = choose
            result = runner.run_pilot()
            self.assertEqual(j.count, 10)
            self.assertTrue(result["final_reflection_committed"])
            self.assertEqual(sum(role == "reflection" for role, _ in backend.payloads), 1)
            self.assertEqual(j.store.trials()[-1].metadata["kind"], "verification")
            j.audit(); j.store.close()
