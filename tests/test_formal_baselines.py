import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from admtune.baseline_strategies import AgenticDBStrategy, AgentTuneStrategy, rank_centroid
from admtune.budget import BudgetExhausted, Limits
from admtune.continuous import Journal
from admtune.contracts import Environment
from admtune.formal_baselines import BaselineLLM, BaselineRun, FinalLedger
from admtune.composite_static import CompositeExecutor
from llm_client import LLMClient
from test_calibration import space
from test_continuous import FixtureExecutor
from tools.plot_formal_baselines import plot


def record(s, seq=1, value=10, config=None, kind="baseline"):
    return {"trial_id": str(seq), "config": config or s.defaults, "value": value,
            "status": "measured", "complete": True, "effective": True, "constraints_ok": True,
            "metadata": {"kind": kind, "accepted": False, "raw_tps": value*10, "p95_latency_ms": 10}}


class ScriptLLM:
    def __init__(self, s):
        self.s, self.calls, self.children = s, [], 0

    def complete(self, role, payload, prompt, **kwargs):
        self.calls.append((role, copy.deepcopy(payload), kwargs))
        if role == "agenttune_selector":
            return {"selected": list(payload["catalog"])}
        if role == "agenttune_pruner":
            return {"ranges": {a: {"min_value": k["lower"], "max_value": k["upper"],
                                   "step": k["step"], "special_values": list(k["special"])}
                               for a, k in payload["catalog"].items()}}
        self.children += 1
        cfg = dict(payload["current_configuration"])
        alias = next(a for a in cfg if self.s.knobs[dict(
            zip(cfg, self.s.names))[a]].name == "innodb_io_capacity")
        cfg[alias] = 300+100*(self.children % 10)
        return {"configuration": cfg}


class FormalBaselineTests(unittest.TestCase):
    def test_rank_exactly_matches_original_golden_cases(self):
        fixture = json.loads((Path(__file__).parent/"fixtures/agenttune_rank_golden.json").read_text())
        for c in fixture["cases"]:
            self.assertEqual(rank_centroid(c["configs"], ["a", "b"]), c["expected"])

    def test_agenttune_three_independent_identical_prompts_and_two_measured_beam(self):
        s = space(); llm = ScriptLLM(s)
        strategy = AgentTuneStrategy(s, llm, {}, lambda *a: None)
        records = [record(s)]
        for seq in range(2, 5):
            cfg, kind, _ = strategy.propose(records, s.defaults)
            r = record(s, seq, 11, cfg, kind); records.append(r); strategy.observe(r)
        child_calls = [c for c in llm.calls if c[0] == "agenttune_recommender"]
        self.assertEqual(len(child_calls), 9)  # 3 first parent; then 3 per measured parent.
        self.assertEqual(child_calls[0][1], child_calls[1][1])
        self.assertEqual(child_calls[1][1], child_calls[2][1])
        self.assertEqual(child_calls[0][2], {"temperature": 1, "top_p": .98})
        self.assertEqual({c[1]["parent_feedback"]["trial_id"] for c in child_calls[3:]}, {"2", "3"})
        self.assertNotIn("innodb_io_capacity", json.dumps(child_calls[0][1]))

    def test_agenttune_does_not_expand_until_whole_beam_measured(self):
        s = space(); strategy = AgentTuneStrategy(s, ScriptLLM(s), {}, lambda *a: None)
        strategy.propose([record(s)], s.defaults)
        strategy.propose([record(s)], s.defaults)
        with self.assertRaisesRegex(ValueError, "incomplete"):
            strategy.propose([record(s)], s.defaults)

    def test_malformed_child_is_charged_and_rejected_without_extra_resample(self):
        s = space(); llm = ScriptLLM(s); complete = llm.complete
        def broken(role, *args, **kwargs):
            if role == "agenttune_recommender" and llm.children == 0:
                llm.children += 1
                raise ValueError("Invalid JSON")
            return complete(role, *args, **kwargs)
        events = []; llm.complete = broken
        strategy = AgentTuneStrategy(s, llm, {}, lambda k, v: events.append(k))
        strategy.propose([record(s)], s.defaults)
        self.assertEqual(llm.children, 3)
        self.assertIn("agenttune_child_rejected", events)

    def test_budget_failure_not_swallowed_as_malformed_child(self):
        s = space(); llm = ScriptLLM(s); strategy = AgentTuneStrategy(s, llm, {}, lambda *a: None)
        strategy.initialize()
        strategy.beam = [record(s)]
        llm.complete = Mock(side_effect=BudgetExhausted("cap"))
        with self.assertRaises(BudgetExhausted):
            strategy.propose([record(s)], s.defaults)

    def test_native_acceptance_rules_differ(self):
        s = space(); before = record(s, value=100); after = record(s, value=100.5)
        self.assertTrue(AgentTuneStrategy(s, None, {}, None).accepts(after, before))
        self.assertFalse(AgenticDBStrategy(s, None, {}, None).accepts(after, before))

    def test_agenticdb_requires_explicit_db_action_and_preserves_plan_feedback(self):
        s = space(); llm = Mock()
        strategy = AgenticDBStrategy(s, llm, {}, lambda *a: None)
        llm.complete.return_value = {"candidate_config": {}}
        with self.assertRaises(ValueError):
            strategy.propose([record(s)], s.defaults)
        llm.complete.return_value = {"action_type": "db_config",
            "candidate_config": {"innodb_io_capacity": 300}, "global_config_plan": {"goal": "test"}}
        cfg, kind, _ = strategy.propose([record(s)], s.defaults)
        r = record(s, 2, 11, cfg, kind); r["metadata"]["accepted"] = True
        strategy.observe(r)
        self.assertEqual(strategy.rounds[0]["proposal"]["global_config_plan"], {"goal": "test"})
        self.assertFalse(strategy.snapshot()["gp_used"])
        self.assertFalse(strategy.snapshot()["reflexion_used"])
        self.assertNotIn("{{current_phase}}", llm.complete.call_args.args[2])

    def make_runner(self, root, method="Default", strategy=None):
        s = space(); raw = FixtureExecutor(s); raw.resize_checks = []
        ex = CompositeExecutor(raw); j = Journal(root/"records.sqlite3")
        ledger = FinalLedger(j, Limits(trials=8, wall_seconds=1500, llm_calls=20,
                                      llm_tokens=64000), 0)
        env = Environment("mysql", "fixture", "fixture", "fixture", "fixture", "fixture")
        return j, BaselineRun(method, j, ex, s, env, root, ledger, strategy)

    def test_default_three_real_slots_zero_llm_and_independent_verification(self):
        with tempfile.TemporaryDirectory() as tmp:
            j, r = self.make_runner(Path(tmp))
            out = r.run()
            self.assertTrue(out["baseline_ready"]); self.assertEqual(j.count, 3)
            self.assertEqual(out["budget"]["llm_calls"], 0)
            self.assertEqual([t.metadata["kind"] for t in j.store.trials()],
                             ["baseline", "default_repeat", "verification"])
            j.store.close()

    def test_eight_slot_cap_includes_baseline_final_and_raw_metrics(self):
        s = space(); strategy = Mock()
        strategy.snapshot.return_value = {}
        strategy.propose.return_value = ({**s.defaults, "innodb_io_capacity": 300}, "candidate", {})
        strategy.accepts.return_value = False
        with tempfile.TemporaryDirectory() as tmp:
            j, r = self.make_runner(Path(tmp), "AgenticDB", strategy)
            out = r.run()
            self.assertEqual(j.count, 8); self.assertEqual(out["valid_candidate_measurements"], 6)
            self.assertEqual(j.store.trials()[-1].metadata["kind"], "verification")
            self.assertNotEqual(j.store.trials()[0].metadata["raw_tps"], j.store.trials()[0].value)
            with self.assertRaises(BudgetExhausted):
                r.step(s.defaults, "candidate", {})
            j.store.close()

    def test_baseline_llm_no_cache_and_usage_charged_on_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            j, r = self.make_runner(Path(tmp)); client = Mock()
            client.dry_run = False; client.api_key = "fixture-secret"; client.model = "fixture"
            client.timeout_s = 90
            client.consume_last_usage.return_value = {"total_tokens": 12}
            client._call_openai_compatible.return_value = {"ok": True}
            llm = BaselineLLM(client, r.ledger, Path(tmp)/"llm_calls")
            for _ in range(2):
                llm.complete("same", {}, "JSON", temperature=1, top_p=.98)
            client._call_openai_compatible.side_effect = ValueError("invalid JSON")
            with self.assertRaises(ValueError):
                llm.complete("same", {}, "JSON")
            self.assertEqual(client._call_openai_compatible.call_count, 3)
            self.assertEqual(r.ledger.llm_calls, 3); self.assertEqual(r.ledger.llm_tokens, 36)
            self.assertEqual(len(list((Path(tmp)/"llm_calls").glob("*.json"))), 3)
            j.store.close()

    def test_transport_sampling_options_and_old_defaults(self):
        client = LLMClient("fixture-secret", "https://example.invalid", "fixture", response_retries=0)
        response = Mock()
        response.json.return_value = {"choices": [{"message": {"content": '{"ok":true}'}}]}
        with patch("requests.post", return_value=response) as post:
            client._call_openai_compatible("a", operation="test", system_prompt="JSON")
            self.assertEqual(post.call_args.kwargs["json"]["temperature"], .2)
            self.assertNotIn("top_p", post.call_args.kwargs["json"])
            client._call_openai_compatible("a", operation="test", system_prompt="JSON",
                                           temperature=1, top_p=.98)
            self.assertEqual(post.call_args.kwargs["json"]["top_p"], .98)
        for kwargs in ({"temperature": True}, {"temperature": float("nan")}, {"top_p": 0}):
            with patch("requests.post") as post, self.assertRaises(ValueError):
                client._call_openai_compatible("a", operation="test", system_prompt="JSON", **kwargs)
            post.assert_not_called()

    def test_plot_keeps_estimates_out_of_measured_table(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = {"simulated": False, "method": "Default", "repeat": 0,
                        "comparison_protocol": {"fixture": True}, "benchmark": {}, "space": []}
            source = root/"Default"; source.mkdir()
            (source/"manifest.json").write_text(json.dumps(manifest))
            (source/"summary.json").write_text(json.dumps({"status": "completed", "completed_trials": 3,
                "budget": {"llm_calls": 0, "llm_tokens_or_reserved": 0}}))
            rs = []
            for i, kind in enumerate(("baseline", "default_repeat", "verification"), 1):
                r = record(space(), i, kind=kind)
                r["metadata"].update(seq=i, active_elapsed_seconds=75*i)
                rs.append(r)
            (source/"trials.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rs))
            literature = Path(__file__).resolve().parents[1]/"docs/experiment_data/timed_static_100min_002/literature_curves.csv"
            rows = plot([source], root/"fig", literature)
            self.assertEqual([r["method"] for r in rows], ["Default"])
            estimates = (root/"fig/literature_ratio_curves.csv").read_text()
            self.assertIn("literature_ratio_estimate", estimates)
            self.assertTrue((root/"fig/measured_comparison.png").is_file())


if __name__ == "__main__":
    unittest.main()
