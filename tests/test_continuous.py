from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from admtune.agents import HeuristicBackend
from admtune.continuous import ContinuousRun, DurableLedger, Journal, code_hash
from admtune.contracts import Environment
from admtune.execution import Measurement
from admtune.surrogate import Prediction
from test_calibration import space


class FastGP:
    """仅测试续接账本，不作为实验结果或模型效果。"""
    def __init__(self, **kwargs):
        self.version, self.fitted = 0, False

    def fit(self, x, y, **kwargs):
        self.x, self.y = x, y
        self.version += 1
        self.fitted = True

    def snapshot(self, ids):
        return {"version": self.version, "x": self.x, "target": self.y, "training_ids": ids}

    @classmethod
    def from_snapshot(cls, p):
        gp = cls(); gp.fit(p["x"], p["target"]); gp.version = p["version"]
        return gp

    def predict(self, x):
        return [Prediction(sum(self.y) / len(self.y), .1, .1, self.version) for _ in x]


class FixtureExecutor:
    simulated = True
    estimated_trial_seconds = 0

    def __init__(self, s):
        self.space, self.current, self.calls = s, s.defaults, []

    def readback(self):
        return dict(self.current)

    def diagnose(self):
        return {"fixture_only": True}

    def measure(self, config, context):
        self.calls.append(config)
        self.current = config
        return Measurement(100 + 10 * sum(self.space.encode(config)),
                           metadata={"p95_latency_ms": 10, "simulated": True})

    def restore_initial(self):
        self.current = self.space.defaults


class ChargedFixture:
    def __init__(self, ledger):
        self.ledger = ledger
        self.backend = HeuristicBackend(42)

    def complete(self, role, payload):
        self.ledger.reserve_llm(100)
        result = self.backend.complete(role, payload)
        self.ledger.settle_llm(100, 20)
        return result


class ContinuationTests(unittest.TestCase):
    def make(self, root):
        j = Journal(root / "records.sqlite3")
        s = space()
        ex = FixtureExecutor(s)
        env = Environment("mysql", "test", "test", "test", "test", "test")
        ledger = DurableLedger(j)
        run = ContinuousRun(j, ex, s, env, root, ChargedFixture(ledger), ledger)
        return j, run, ex

    def test_50_then_51_52_and_100_keep_baseline_ids_and_budget(self):
        with tempfile.TemporaryDirectory() as tmp, patch("admtune.continuous.GaussianProcess", FastGP):
            root = Path(tmp)
            j, runner, ex = self.make(root)
            result = runner.run_until(50)
            self.assertTrue(result["can_resume"])
            first = j.store.trials()[0].trial_id
            calls = runner.ledger.llm_calls
            best = runner.state["best_value"]
            self.assertEqual(len(ex.calls), 50)
            self.assertEqual(j.store.trials()[-1].metadata["kind"], "verification")
            j.store.close()
            for endpoint in (51, 52, 100):
                j, runner, ex = self.make(root)
                previous = j.count
                runner.run_until(endpoint)
                self.assertEqual(len(ex.calls), endpoint - previous)
                self.assertEqual(j.store.trials()[0].trial_id, first)
                self.assertGreaterEqual(runner.ledger.llm_calls, calls)
                self.assertGreaterEqual(runner.state["best_value"], best)
                self.assertEqual([t.metadata["seq"] for t in j.store.trials()], list(range(1, endpoint + 1)))
                self.assertEqual(len({t.trial_id for t in j.store.trials()}), endpoint)
                self.assertEqual(runner.ledger.trials, endpoint)
                j.audit(); j.store.close()

    def test_unknown_pending_is_charged_not_reexecuted_or_trained(self):
        with tempfile.TemporaryDirectory() as tmp, patch("admtune.continuous.GaussianProcess", FastGP):
            root = Path(tmp); j, runner, ex = self.make(root)
            runner.run_until(2)
            state = runner.save_state()
            submitted = j.reserve({"kind": "candidate", "config": runner.state["initial"]}, state)
            self.assertEqual(submitted["seq"], 3)
            j.store.close()
            j, runner, ex = self.make(root)
            self.assertEqual(j.recover_pending(runner.environment.key), 1)
            unknown = j.store.trials()[-1]
            self.assertFalse(unknown.trainable)
            runner.run_until(4)
            self.assertEqual(len(ex.calls), 1)
            self.assertEqual(j.store.trials()[-1].metadata["seq"], 4)
            j.audit(); j.store.close()

    def test_attempt_cannot_be_finished_twice(self):
        with tempfile.TemporaryDirectory() as tmp, patch("admtune.continuous.GaussianProcess", FastGP):
            root = Path(tmp); j, runner, _ = self.make(root)
            runner.run_until(1)
            record = j.store.trials()[0]
            with self.assertRaises(ValueError):
                j.finish({"seq": 1}, record, runner.save_state())
            self.assertEqual(j.count, 1)
            self.assertEqual(len(j.store.trials()), 1)
            j.store.close()

    def test_documentation_change_does_not_invalidate_code_but_python_change_does(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root / "config.py").write_text("x=1")
            before = code_hash(root)
            (root / "notes.md").write_text("new evidence")
            self.assertEqual(before, code_hash(root))
            (root / "config.py").write_text("x=2")
            self.assertNotEqual(before, code_hash(root))

    def test_real_gp_rebuilds_and_continues_without_reusing_snapshot_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            j, runner, _ = self.make(root)
            runner.run_until(8)
            original = {p.name: p.read_bytes() for p in (root / "gp_snapshots").iterdir()}
            ids = runner.state["gp"]["training_ids"]
            self.assertEqual(len(ids), 8)
            j.store.close()
            j, runner, _ = self.make(root)
            self.assertEqual(runner.state["gp"]["training_ids"], ids)
            runner.run_until(9)
            self.assertEqual(len(runner.state["gp"]["training_ids"]), 9)
            for name, raw in original.items():
                self.assertEqual((root / "gp_snapshots" / name).read_bytes(), raw)
            j.store.close()
