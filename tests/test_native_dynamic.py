from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock

from admtune.budget import BudgetExhausted, Ledger, Limits
from admtune.contracts import Environment, Objective
from admtune.dynamic import completion_checks, BatchedController
from admtune.controller import Policy
from admtune.execution import Measurement
from admtune.memory import Store
from admtune.native_dynamic import NativeDynamicController, native_record, PhaseBudgetLLM
from test_calibration import space
from test_formal_baselines import ScriptLLM
from admtune.replay import ReplayPlan, monitor, make_schedule
from dataclasses import replace
from tools.run_dynamic_matrix import method_order


class LatencyExecutor:
    simulated = True
    estimated_trial_seconds = 0
    def __init__(self, s): self.current = s.defaults
    def readback(self): return dict(self.current)
    def measure(self, config, context):
        self.current = dict(config)
        return Measurement(10 if config["innodb_buffer_pool_size"] == 1024**3 else 8)
    def restore(self, config): self.current = dict(config)


class NativeDynamicTests(unittest.TestCase):
    def build(self, root, method, llm):
        s = space(); store = Store(root/"records.sqlite3")
        c = NativeDynamicController(method=method, llm=llm, workload={"fixture": True},
            space=s, environment=Environment("mysql","fixture","fixture","fixture","fixture","fixture"),
            objective=Objective("p95_latency_ms", "minimize"),
            detector=SimpleNamespace(state="stable", changed_count=0), executor=LatencyExecutor(s),
            team=None, reflector=None, store=store,
            ledger=Ledger(Limits(trials=18, wall_seconds=3000, llm_calls=36, llm_tokens=120000)))
        return store, c

    def test_agenticdb_uses_inverse_p95_and_resets_auditor_not_facts(self):
        llm = Mock()
        llm.complete.return_value = {"action_type": "db_config",
                                    "candidate_config": {"innodb_buffer_pool_size": 2*1024**3}}
        with tempfile.TemporaryDirectory() as tmp:
            st, c = self.build(Path(tmp), "AgenticDB", llm)
            c.begin_phase("phase-1", [0.]); r = c.step()
            self.assertTrue(r.metadata["accepted"])
            self.assertEqual(c.best_value, 8)
            self.assertEqual(native_record(r)["value"], .125)
            self.assertIn("1 / P95", llm.complete.call_args.args[1]["objective"]["description"])
            c.begin_phase("phase-2", [1.])
            self.assertEqual(len(c.strategy.rounds), 1)
            self.assertEqual(c.strategy.rounds[0]["metadata"]["workload_phase_id"], "phase-1")
            self.assertEqual(c.strategy.current_workload_phase, "phase-2")
            self.assertEqual(len(c.strategy.earlier_phase_measurements), 2)
            c.fit([1.]); c.reflect(["phase_end"], force=True)
            self.assertFalse(c.gp.fitted)
            self.assertEqual(st.reflections(), [])
            st.close()

    def test_agenttune_measured_beam_and_new_phase_parent(self):
        llm = ScriptLLM(space())
        with tempfile.TemporaryDirectory() as tmp:
            st, c = self.build(Path(tmp), "AgentTune", llm)
            c.begin_phase("phase-1", [0.])
            c.step(); c.step(); c.step()
            reco = [r for r in llm.calls if r[0] == "agenttune_recommender"]
            self.assertEqual(len(reco), 9)
            self.assertIn("1 / P95", reco[0][1]["objective"])
            c.begin_phase("phase-2", [1.]); c.step()
            reco = [r for r in llm.calls if r[0] == "agenttune_recommender"]
            self.assertEqual(reco[-1][1]["parent_feedback"]["phase_id"], "phase-2")
            self.assertFalse(c.gp.fitted)
            st.close()

    def test_native_completion_does_not_fabricate_gp_or_reflection(self):
        phases = [{"phase_id": f"p{i}", "baseline_value": 10,
                   "verification_values": [10], "valid_candidates": 0} for i in range(3)]
        s = completion_checks(phases, [], [], require_gp=False, require_reflection=False, require_search=False)
        self.assertTrue(s["control_flow_completed"])
        self.assertFalse(s["gp_selected_execution_exercised"])
        self.assertFalse(completion_checks(phases, [], [])["control_flow_completed"])

    def test_phase_time_budget_prevents_unpaid_late_attempt(self):
        client = Mock(dry_run=False, api_key="fixture", model="fixture")
        with tempfile.TemporaryDirectory() as tmp:
            llm = PhaseBudgetLLM(client, Mock(), Path(tmp), available=lambda: 0)
            with self.assertRaises(BudgetExhausted):
                llm.complete("selector", {}, "JSON")
            client._call_openai_compatible.assert_not_called()

    def test_single_slice_changes_representation_not_request_plan(self):
        p = ReplayPlan(phase_seconds=24, read_percent=100, query_profile="range-mixed",
                       range_rows=100, top_k=20)
        single = replace(p, single_slice=True)
        self.assertEqual(make_schedule(p), make_schedule(single))
        self.assertEqual(monitor(p)[1].encoder.intensity_dimensions, 2)
        self.assertEqual(monitor(single)[1].encoder.intensity_dimensions, 1)

    def test_no_history_resets_model_and_filters_old_facts_without_resetting_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            s = space(); st = Store(Path(tmp)/"records.sqlite3")
            c = BatchedController(space=s, environment=Environment("mysql","t","t","t","t","t"),
                objective=Objective("p95_latency_ms","minimize"),
                detector=SimpleNamespace(state="stable",changed_count=0), executor=LatencyExecutor(s),
                team=Mock(), reflector=Mock(), store=st, ledger=Ledger(Limits()),
                policy=Policy(cross_phase_memory=False), seed=42)
            c.begin_phase("p1",[0.]); current = dict(c.current); c.gp.version = 25
            c.begin_phase("p2",[0.])
            self.assertEqual(c.current, current)
            self.assertEqual(c.gp.version, 0)
            self.assertEqual(len(c.histories([0.])), 1)
            c.reflection_enabled = False; c.reflect(["phase_end"], force=True)
            c.reflector.trigger.assert_not_called()
            st.close()

    def test_method_orders_rotate_and_do_not_reuse_pilot_seed(self):
        methods = ["Default","AgenticDB","AgentTune","ADMTune"]
        self.assertEqual(method_order(43,methods), ["AgenticDB","AgentTune","ADMTune","Default"])
        for seed in (43,44,45):
            self.assertEqual(set(method_order(seed,methods)), set(methods))
        self.assertNotEqual(method_order(43,methods), method_order(44,methods))

    def test_no_history_snapshot_versions_cannot_overwrite_previous_phase(self):
        with tempfile.TemporaryDirectory() as tmp:
            s = space(); root = Path(tmp); st = Store(root/"records.sqlite3")
            c = BatchedController(space=s, environment=Environment("mysql","t","t","t","t","t"),
                objective=Objective("p95_latency_ms","minimize"),
                detector=SimpleNamespace(state="stable",changed_count=0), executor=LatencyExecutor(s),
                team=Mock(), reflector=Mock(), store=st, ledger=Ledger(Limits()),
                policy=Policy(cross_phase_memory=False,bootstrap=2), seed=42,
                snapshot_dir=root/"models")
            c.begin_phase("p1",[0.])
            c.trial({**s.defaults,"innodb_buffer_pool_size":2*1024**3},[0.],kind="startup")
            c.fit([0.])
            c.begin_phase("p2",[0.])
            c.trial(s.defaults,[0.],kind="startup"); c.fit([0.])
            self.assertTrue((root/"models/p1-gp-0001.json").exists())
            self.assertTrue((root/"models/p2-gp-0001.json").exists())
            st.close()


if __name__ == "__main__":
    unittest.main()
