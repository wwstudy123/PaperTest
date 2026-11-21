from dataclasses import asdict
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from admtune.agents import HeuristicBackend, Team
from admtune.budget import BudgetExhausted, Ledger, Limits
from admtune.contracts import Environment, Objective, Trial
from admtune.memory import Store
from admtune.reflection import Reflector
from admtune.space import Knob, Space
from admtune.surrogate import GaussianProcess
from admtune.workload import PhaseDetector, TemplateCatalog, Window, WorkloadEncoder


def window(index, first=8, second=2):
    return Window(index * 30, (index + 1) * 30, [[first, second, 0]] * 6)


class SpaceTests(unittest.TestCase):
    def setUp(self):
        self.space = Space([
            Knob("buffer", "integer", 4, 1, 10, step=1),
            Knob("mode", "enum", "a", choices=("a", "b")),
        ])

    def test_full_parent_inheritance(self):
        config = self.space.merge(self.space.defaults, {"buffer": 6})
        self.assertEqual(config, {"buffer": 6, "mode": "a"})
        self.assertEqual(len(self.space.encode(config)), 3)

    def test_invalid_configs(self):
        for delta in [{"buffer": 1.1}, {"buffer": True}, {"buffer": float("nan")}, {"mode": "unknown"}, {"other": 1}]:
            with self.assertRaises(ValueError):
                self.space.merge(self.space.defaults, delta)

    def test_protected_knobs_never_enter_catalog(self):
        with self.assertRaises(ValueError):
            Knob("sync_binlog", "integer", 1, 0, 10)

    def test_special_encoding(self):
        knob = Knob("autobuffer", "integer", 1, 1, 10, special=(0,), log_scale=True)
        self.assertEqual(knob.encode(0), [0.0, 1.0])
        self.assertEqual(knob.encode(1), [0.0, 0.0])

    def test_joint_resource_checks(self):
        space = Space([
            Knob("max_connections", "integer", 100, 1, 500),
            Knob("sort_buffer_size", "integer", 1, 1, 1000),
        ], memory_bytes=1000)
        with self.assertRaises(ValueError):
            space.merge(space.defaults, {"sort_buffer_size": 100})

    def test_team_produces_valid_full_configs(self):
        team = Team(self.space, HeuristicBackend(8))
        team.initialize([0.0], [])
        for candidate in team.recommend(self.space.defaults, "p1", [0.0], []):
            self.space.merge(candidate.config, {})
            self.assertEqual(set(candidate.config), set(self.space.names))


class WorkloadTests(unittest.TestCase):
    def test_sql_ast_templates_and_unknown(self):
        catalog = TemplateCatalog(["select * from t where id=1", "select * from t where id=2"])
        self.assertEqual(catalog.channel("SELECT * FROM t WHERE id=3"), catalog.channel("select * from t where id=4"))
        self.assertEqual(catalog.channel("delete from t where id=1"), catalog.unknown_channel)

    def test_runtime_signals_rejected(self):
        with self.assertRaises(ValueError):
            Window(0, 30, [[2]], source="completed_tps").features()

    def test_phase_confirmation_then_fresh_stability(self):
        encoder = WorkloadEncoder([window(0), window(1)])
        detector = PhaseDetector(encoder, change_threshold=1, stable_threshold=0.5)
        self.assertIsNone(detector.observe(window(0)).phase_id)
        self.assertEqual(detector.observe(window(1)).event, "phase_ready")
        self.assertEqual(detector.observe(window(2, 2, 8)).event, "change_suspected")
        self.assertEqual(detector.observe(window(3, 2, 8)).event, "change_confirmed")
        self.assertEqual(detector.observe(window(4, 2, 8)).state, "waiting_stable")
        self.assertEqual(detector.observe(window(5, 2, 8)).phase_id, "phase-2")

    def test_overlap_and_dimension_changes_rejected(self):
        encoder = WorkloadEncoder([window(0)])
        detector = PhaseDetector(encoder, change_threshold=1, stable_threshold=1)
        detector.observe(window(0))
        with self.assertRaises(ValueError):
            detector.observe(Window(10, 40, [[8, 2, 0]] * 6))
        with self.assertRaises(ValueError):
            encoder.encode(Window(0, 30, [[1, 2]]))

    def test_single_slice_removes_only_internal_order(self):
        a = Window(0, 30, [[8, 2], [2, 8]])
        b = Window(30, 60, [[2, 8], [8, 2]])
        full, single = WorkloadEncoder([a]), WorkloadEncoder([a], single_slice=True)
        self.assertFalse(np.allclose(full.encode(a), full.encode(b)))
        self.assertTrue(np.allclose(single.encode(a), single.encode(b)))


class ModelTests(unittest.TestCase):
    def test_objective_and_stage_reference(self):
        obj = Objective()
        self.assertAlmostEqual(obj.utility(120, 100), 1.2)
        self.assertAlmostEqual(obj.utility(120, 120), 1)
        self.assertEqual(obj.target(120), obj.target(120))  # Label never takes a baseline argument.
        self.assertAlmostEqual(Objective("complete_latency_s", "minimize").utility(5, 10), 2)
        with self.assertRaises(ValueError):
            obj.target(0)

    def test_gp_predictions_and_noise_are_distinct(self):
        gp = GaussianProcess()
        gp.fit([[0, 0], [0.5, 0], [1, 0]], [math.log(10), math.log(20), math.log(15)])
        p = gp.predict([[0.5, 0]])[0]
        self.assertLess(abs(p.mean - math.log(20)), 0.1)
        self.assertGreater(p.observation_std, p.latent_std)
        with self.assertRaises(ValueError):
            gp.predict([[1]])

    def test_joint_kernel_can_represent_ranking_reversal(self):
        gp = GaussianProcess()
        gp.fit([[0, 0], [1, 0], [0, 1], [1, 1]], [1, 2, 2, 1], optimize=False)
        prediction = gp.predict([[0, 0], [1, 0], [0, 1], [1, 1]])
        self.assertLess(prediction[0].mean, prediction[1].mean)
        self.assertGreater(prediction[2].mean, prediction[3].mean)

    def test_single_observation_is_not_a_calibrated_gp(self):
        with self.assertRaises(ValueError):
            GaussianProcess().fit([[1]], [1])


class BudgetTests(unittest.TestCase):
    def test_global_trial_slots_are_not_refunded(self):
        ledger = Ledger(Limits(trials=1))
        ledger.charge_trial("failed")
        with self.assertRaises(BudgetExhausted):
            ledger.charge_trial("new_phase")

    def test_wall_reserve(self):
        clock = [0]
        ledger = Ledger(Limits(wall_seconds=100, recovery_reserve_seconds=20), clock=lambda: clock[0])
        clock[0] = 75
        with self.assertRaises(BudgetExhausted):
            ledger.charge_trial("candidate", 10)

    def test_llm_reservation(self):
        ledger = Ledger(Limits(llm_calls=1, llm_tokens=100))
        ledger.reserve_llm(90)
        ledger.settle_llm(90, 40)
        self.assertEqual(ledger.llm_tokens, 40)
        with self.assertRaises(BudgetExhausted):
            ledger.reserve_llm(10)


class MemoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = Store(Path(self.temp.name) / "records.sqlite3")
        self.record = Trial("t1", "p1", "env", "tps:maximize", {"buffer": 4}, [0.0],
                            "measured", 10, True, True, True)
        self.store.append(self.record)

    def tearDown(self):
        self.store.close()
        self.temp.cleanup()

    def test_context_and_environment_compatibility(self):
        self.assertEqual(len(self.store.compatible("env", "tps:maximize", [0])), 1)
        self.assertEqual(self.store.compatible("other", "tps:maximize", [0]), [])
        self.assertEqual(self.store.compatible("env", "tps:maximize", [100]), [])
        self.assertEqual(self.store.compatible("env", "tps:maximize", [0], phase_id="p2"), [])

    def test_failure_is_not_a_label(self):
        self.store.append(Trial("t2", "p1", "env", "tps:maximize", {}, [0], "timeout", 0))
        self.assertEqual(len(self.store.compatible("env", "tps:maximize", [0])), 1)

    def test_incremental_evidence_consumption(self):
        reflector = Reflector(self.store, HeuristicBackend(), min_batch=1)
        self.assertTrue(reflector.trigger(["new_best"]))
        self.assertFalse(reflector.trigger(["new_best"]))
        self.assertEqual(self.store.pending(), [])

    def test_invalid_reflection_does_not_consume(self):
        class Invalid:
            def complete(self, role, payload):
                return {"facts": [{"trial_id": "t1", "status": "measured", "value": 100000}],
                        "hypotheses": [], "conditions": {"environment_key": "env", "objective_key": "tps:maximize"},
                        "actions": [], "evidence_ids": ["t1"]}
        with self.assertRaises(ValueError):
            Reflector(self.store, Invalid(), min_batch=1).trigger(["event"])
        self.assertEqual(len(self.store.pending()), 1)
        self.assertEqual(self.store.reflections(), [])

    def test_duplicate_consumption_is_atomic(self):
        import sqlite3
        reflector = Reflector(self.store, HeuristicBackend(), min_batch=1)
        reflector.trigger(["event"])
        payload = self.store.reflections()[0]
        with self.assertRaises(sqlite3.IntegrityError):
            self.store.commit_reflection(payload, ["t1"])
        self.assertEqual(len(self.store.reflections()), 1)


if __name__ == "__main__":
    unittest.main()
