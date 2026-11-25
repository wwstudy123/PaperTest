import csv
import json
import math
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from admtune.agents import HeuristicBackend
from admtune.budget import BudgetExhausted, Limits
from admtune.composite_static import (CompositeExecutor, CompositeLedger, CompositeRun,
                                     OBJECTIVE, SearchStopped)
from admtune.continuous import ContinuousRun, Journal
from admtune.contracts import Environment, Trial, fingerprint
from admtune.execution import Measurement
from admtune.surrogate import Prediction
from test_calibration import space
from test_continuous import FixtureExecutor


class ScriptedBackend(HeuristicBackend):
    def __init__(self, ledger):
        super().__init__(42)
        self.ledger, self.payloads = ledger, []

    def complete(self, role, payload):
        self.ledger.reserve_llm(100)
        self.payloads.append((role, payload))
        if role == "recommender":
            result = {"candidates": [{"innodb_io_capacity": 300+100*i} for i in range(8)]}
        else:
            result = super().complete(role, payload)
        self.ledger.settle_llm(100, 20)
        return result


class RejectGP:
    fitted, version = True, 9
    def predict(self, x):
        return [Prediction(math.log(40), .001, .001, 9) for _ in x]


class CompositeTests(unittest.TestCase):
    def make(self, root):
        s = space()
        raw = FixtureExecutor(s)
        raw.run_dir = root
        ex = CompositeExecutor(raw)
        j = Journal(root/"records.sqlite3")
        ledger = CompositeLedger(j, Limits(trials=8, wall_seconds=1200, llm_calls=8,
                                          llm_tokens=32000, recovery_reserve_seconds=60), 0)
        backend = ScriptedBackend(ledger)
        env = Environment("mysql", "fixture", "fixture", "fixture", "fixture", "composite")
        runner = CompositeRun(j, ex, s, env, root, backend, ledger)
        return j, runner, ex, backend

    def test_adapter_keeps_raw_metrics_and_declared_precision(self):
        raw = SimpleNamespace(measure=lambda c, x: Measurement(
            1000, metadata={"p95_latency_ms": 18.28, "simulated": True}), initial=None)
        ex = CompositeExecutor(raw)
        m = ex.measure({}, [])
        self.assertEqual(m.value, round(1000/18.28, 6))
        self.assertEqual(m.metadata["raw_tps"], 1000)
        self.assertEqual(m.metadata["joint_value_unrounded"], 1000/18.28)
        self.assertEqual(m.metadata["optimization_objective"], OBJECTIVE)

    def test_adapter_does_not_train_failures_or_nonfinite_values(self):
        for value, latency, status, complete in [
                (None, None, "execution_failed", False),
                (1000, float("nan"), "measured", True),
                (float("nan"), 20, "measured", True), (1000, 0, "measured", True)]:
            raw = SimpleNamespace(initial=None, measure=lambda c, x: Measurement(
                value, status, complete, complete, complete, {"p95_latency_ms": latency}))
            m = CompositeExecutor(raw).measure({}, [])
            self.assertIsNone(m.value)
            json.dumps(m.metadata, allow_nan=False)

    def test_throughput_gain_with_tail_regression_is_not_accepted_or_mislabeled(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); j, r, ex, _ = self.make(root)
            calls = 0
            def measure(config, context):
                nonlocal calls
                calls += 1
                return Measurement(2449.19 if calls == 1 else 3807.49,
                    metadata={"p95_latency_ms": 18.28 if calls == 1 else 68.05, "simulated": True})
            ex.raw.measure = measure
            r.step()
            initial, baseline = dict(r.state["incumbent"]), r.state["best_value"]
            config = {**initial, "innodb_buffer_pool_size": 2*1024**3}
            r.choose = lambda seq: ("startup", config, {"rule": "fixture"})
            bad = r.step()
            self.assertFalse(bad.metadata["accepted"])
            self.assertEqual(r.state["incumbent"], initial)
            self.assertEqual(r.state["best_value"], baseline)
            self.assertEqual(bad.objective_key, OBJECTIVE)
            self.assertEqual(bad.value, round(3807.49/68.05, 6))
            with (root/"curve.csv").open() as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(float(rows[1]["tps"]), 3807.49)
            self.assertEqual(float(rows[1]["objective_value"]), bad.value)
            self.assertEqual(float(rows[1]["best_config_tps"]), 2449.19)
            self.assertEqual(r.team.feedback["objective_key"], OBJECTIVE)
            j.store.close()

    def test_real_gp_labels_are_log_joint_score_and_predictions_are_prospective(self):
        with tempfile.TemporaryDirectory() as tmp:
            j, r, _, _ = self.make(Path(tmp))
            r.run_until(7)
            trials = j.store.trials()
            for ident, target in zip(r.state["gp"]["training_ids"], r.state["gp"]["target"]):
                t = next(t for t in trials if t.trial_id == ident)
                self.assertAlmostEqual(target, math.log(t.value), places=12)
                self.assertNotAlmostEqual(target, math.log(t.metadata["raw_tps"]), places=3)
            tested = [t for t in trials if t.prediction]
            self.assertTrue(tested)
            for t in tested:
                snapshot = json.loads((Path(tmp)/t.metadata["model_snapshot"]).read_text())
                self.assertTrue(all(next(x for x in trials if x.trial_id == ident).metadata["seq"]
                                    < t.metadata["seq"] for ident in snapshot["training_ids"]))
            j.store.close()

    def test_empty_admission_refreshes_once_then_stops_without_measurement(self):
        with tempfile.TemporaryDirectory() as tmp:
            j, r, ex, _ = self.make(Path(tmp))
            r.state.update(baseline=100, best_value=100)
            r.gp = RejectGP()
            candidate = {**r.state["initial"], "innodb_io_capacity": 300}
            def replenish(seq):
                if not r.state["pool"]:
                    r.state["pool"] = [candidate]
            r.replenish = replenish
            with self.assertRaises(SearchStopped):
                r.choose(7)
            self.assertEqual(r.state["refresh_count"], 1)
            self.assertEqual(j.count, 0)
            self.assertEqual(ex.raw.calls, [])
            self.assertEqual(r.state["pool"], [])
            self.assertEqual(j.db.execute(
                "SELECT COUNT(*) FROM events WHERE kind='prescreen_refused'").fetchone()[0], 2)
            j.store.close()

    def test_same_model_retired_candidates_removed_and_empty_pool_ignores_cooldown(self):
        with tempfile.TemporaryDirectory() as tmp:
            j, r, _, _ = self.make(Path(tmp))
            retired = {**r.state["initial"], "innodb_io_capacity": 300}
            fresh = {**r.state["initial"], "innodb_io_capacity": 400}
            r.state["retired_by_model"]["0"] = [fingerprint(retired)]
            r.state["next_proposal_at"] = 999
            def replenish(seq):
                self.assertEqual(r.state["next_proposal_at"], 7)
                r.state["pool"] = [retired, fresh]
            with patch("admtune.timed_static.TimedRun.replenish", side_effect=replenish):
                r.replenish(7)
            self.assertEqual(r.state["pool"], [fresh])
            j.store.close()

    def test_total_refresh_limit_stops_without_more_calls(self):
        with tempfile.TemporaryDirectory() as tmp:
            j, r, _, backend = self.make(Path(tmp))
            r.state.update(baseline=100, best_value=100, refresh_count=2)
            r.gp = RejectGP()
            r.state["pool"] = [{**r.state["initial"], "innodb_io_capacity": 300}]
            with self.assertRaises(SearchStopped):
                r.choose(7)
            self.assertEqual(backend.payloads, [])
            self.assertEqual(j.count, 0)
            j.store.close()

    def test_call_and_token_reserves_are_not_free_calls(self):
        with tempfile.TemporaryDirectory() as tmp:
            j, r, _, _ = self.make(Path(tmp))
            with self.assertRaises(BudgetExhausted):
                r.ledger.reserve_llm(23000)
            self.assertEqual(r.ledger.llm_calls, 0)
            for _ in range(7):
                r.ledger.reserve_llm(100); r.ledger.settle_llm(100, 20)
            with self.assertRaises(BudgetExhausted):
                r.ledger.reserve_llm(100)
            r.ledger.final_reflection = True
            r.ledger.reserve_llm(100); r.ledger.settle_llm(100, 20)
            self.assertEqual(r.ledger.llm_calls, 8)
            self.assertEqual(r.ledger.llm_tokens, 160)
            with self.assertRaises(BudgetExhausted):
                r.ledger.reserve_llm(100)
            j.store.close()

    def test_foreign_objective_journal_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            j, r, ex, backend = self.make(Path(tmp))
            j.store.append(Trial("legacy", "phase", r.environment.key, "tps:maximize",
                                 r.state["initial"], [0.], "measured", 100, True, True, True))
            with self.assertRaises(ValueError):
                CompositeRun(j, ex, r.space, r.environment, Path(tmp), backend, r.ledger)
            with self.assertRaises(ValueError):
                r.fit()
            j.store.close()

    def test_small_pilot_ends_with_independent_verification_and_incremental_reflection(self):
        with tempfile.TemporaryDirectory() as tmp:
            j, r, _, backend = self.make(Path(tmp))
            result = r.run_pilot()
            self.assertEqual(result["status"], "completed")
            self.assertLessEqual(j.count, 8)
            self.assertTrue(result["closed_loop_verified"])
            self.assertTrue(result["final_reflection_committed"])
            trials = j.store.trials()
            self.assertEqual(trials[-1].metadata["kind"], "verification")
            self.assertFalse(any(t.metadata["kind"] == "incumbent_repeat" for t in trials))
            reflection = next(p for role, p in backend.payloads if role == "reflection")
            self.assertEqual(reflection["conditions"]["objective_key"], OBJECTIVE)
            self.assertTrue(all("raw_tps" in x and x["label_precision_decimals"] == 6
                                for x in reflection["record_context"]))
            j.audit(); j.store.close()

    def test_time_reserved_for_final_trial_and_reflection(self):
        with tempfile.TemporaryDirectory() as tmp:
            j, r, _, _ = self.make(Path(tmp))
            clock = SimpleNamespace(value=0.)
            r.ledger.clock = lambda: clock.value
            r.ledger.started = 0.
            r.ledger.final_trial_seconds = 105
            self.assertEqual(r.ledger.remaining_seconds(), 1200-105-30)
            clock.value = 950.
            with self.assertRaises(BudgetExhausted):
                r.ledger.ensure_time(105)
            r.ledger.finalizing = True
            r.ledger.ensure_time(105)
            j.store.close()

    def test_reflection_fact_error_is_preserved_not_repaired(self):
        with tempfile.TemporaryDirectory() as tmp:
            j, r, _, backend = self.make(Path(tmp))
            original = backend.complete
            def complete(role, payload):
                result = original(role, payload)
                if role == "reflection":
                    result["facts"] = [dict(f) for f in result["facts"]]
                    result["facts"][0]["value"] += 1
                return result
            backend.complete = complete
            result = r.run_pilot()
            self.assertFalse(result["closed_loop_verified"])
            self.assertFalse(result["final_reflection_committed"])
            self.assertIsNotNone(result["reflection_error"])
            self.assertEqual(j.store.reflections(), [])
            self.assertEqual(len(j.store.pending()), j.count)
            j.store.close()
