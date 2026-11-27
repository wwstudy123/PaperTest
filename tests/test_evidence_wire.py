from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from admtune.agents import HeuristicBackend
from admtune.contracts import Trial
from admtune.evidence_wire import EvidenceWireBackend
from admtune.memory import Store
from admtune.reflection import Reflector


class EvidenceWireTests(unittest.TestCase):
    def test_reflection_keeps_execution_failure_distinct_from_knob_violation(self):
        with tempfile.TemporaryDirectory() as folder:
            store = Store(Path(folder) / "evidence.sqlite3")
            fixture = HeuristicBackend()
            delegate = Mock()
            delegate.complete.side_effect = fixture.complete
            try:
                store.append(Trial("failed", "p1", "env", "p95_latency_ms:minimize", {"x": 1}, [0],
                                   "execution_failed", metadata={"reason": "Buffer-pool resize did not settle",
                                                                "kind": "startup"}))
                reflector = Reflector(store, delegate, old_summary_limit=0)
                self.assertTrue(reflector.trigger(["phase_end"], force=True))
                payload = delegate.complete.call_args.args[1]
                self.assertIn("Buffer-pool resize", payload["record_context"][0]["failure_reason"])
                self.assertFalse(payload["record_context"][0]["measurement_complete"])
                self.assertTrue(any("does not prove" in x for x in payload["interpretation_rules"]))
                self.assertIsNone(payload["facts"][0]["value"])
            finally:
                store.close()

    def payload(self):
        return {
            "facts": [{"trial_id": "long-real-id-1", "status": "measured", "value": 2.35},
                      {"trial_id": "long-real-id-2", "status": "execution_failed", "value": None}],
            "conditions": {"environment_key": "real-environment", "objective_key": "p95_latency_ms:minimize"},
            "old_summaries": [], "events": ["phase_end"],
            "record_context": [{"trial_id": "long-real-id-1", "config": {"x": 1}, "phase_id": "phase-1"},
                               {"trial_id": "long-real-id-2", "config": {"x": 2}, "phase_id": "phase-1"}],
        }

    def test_lossless_roundtrip_and_saved_mapping(self):
        payload = self.payload();original = deepcopy(payload)
        fixture = HeuristicBackend()
        with tempfile.TemporaryDirectory() as folder:
            backend = EvidenceWireBackend(fixture, audit_dir=Path(folder))
            result = backend.complete("reflection", payload)
            self.assertEqual(result["facts"], payload["facts"])
            self.assertEqual(result["conditions"], payload["conditions"])
            self.assertEqual(result["evidence_ids"], ["long-real-id-1", "long-real-id-2"])
            audit = json.loads((Path(folder) / "reflection-001.json").read_text())
            self.assertEqual(audit["alias_to_trial_id"]["t1"], "long-real-id-1")
            self.assertEqual(audit["decoded_response"], result)
            self.assertNotEqual(audit["original_payload_sha256"], audit["wire_payload_sha256"])
        self.assertEqual(payload, original)

    def test_unknown_alias_not_repaired_or_accepted(self):
        for field in ("facts", "evidence_ids"):
            fixture = Mock()
            result = {"facts": [{"trial_id": "t1", "status": "measured", "value": 2.35}],
                      "evidence_ids": ["t1"]}
            if field == "facts":
                result["facts"][0]["trial_id"] = "invented"
            else:
                result["evidence_ids"] = ["invented"]
            fixture.complete.return_value = result
            with self.assertRaisesRegex(ValueError, "Unknown"):
                EvidenceWireBackend(fixture).complete("reflection", self.payload())

    def test_changed_numeric_fact_is_not_silently_corrected(self):
        fixture = Mock()
        fixture.complete.return_value = {
            "facts": [{"trial_id": "t1", "status": "measured", "value": 999}],
            "evidence_ids": ["t1"],
        }
        result = EvidenceWireBackend(fixture).complete("reflection", self.payload())
        self.assertEqual(result["facts"][0]["value"], 999)  # store, not codec, must reject this
        result.update(hypotheses=[], actions=["verify"], conditions=self.payload()["conditions"])
        with tempfile.TemporaryDirectory() as folder:
            store = Store(Path(folder) / "evidence.sqlite3")
            try:
                store.append(Trial(
                    "long-real-id-1", "phase-1", "real-environment", "p95_latency_ms:minimize",
                    {"x": 1}, [0], "measured", 2.35, True, True, True,
                ))
                with self.assertRaisesRegex(ValueError, "Observed facts"):
                    store.commit_reflection(result, ["long-real-id-1"])
                self.assertEqual(store.reflections(), [])
                self.assertEqual(len(store.pending()), 1)
            finally:
                store.close()

    def test_inline_hypothesis_references_are_unambiguous_after_retrieval(self):
        fixture = HeuristicBackend()
        delegate = Mock()
        delegate.complete.side_effect = lambda role, payload: {
            **fixture.complete(role, payload), "hypotheses": ["t1的波动不证明t2变差；勿替换at1或t123。"],
        }
        result = EvidenceWireBackend(delegate).complete("reflection", self.payload())
        self.assertEqual(result["hypotheses"], [
            "long-real-id-1的波动不证明long-real-id-2变差；勿替换at1或t123。",
        ])

    def test_no_background_or_nonbijective_evidence(self):
        for alteration in ("old", "duplicate", "unknown_context"):
            fixture = Mock();payload = self.payload()
            if alteration == "old":
                payload["old_summaries"] = [{"evidence_ids": ["older"]}]
            elif alteration == "duplicate":
                payload["facts"][1]["trial_id"] = payload["facts"][0]["trial_id"]
            else:
                payload["record_context"][0]["trial_id"] = "unknown"
            with self.assertRaises(ValueError):
                EvidenceWireBackend(fixture).complete("reflection", payload)
            fixture.complete.assert_not_called()

    def test_other_roles_pass_through(self):
        fixture = Mock();payload = {"catalog": []}
        fixture.complete.return_value = {"selected": ["x"]}
        self.assertEqual(EvidenceWireBackend(fixture).complete("selector", payload), {"selected": ["x"]})
        fixture.complete.assert_called_once_with("selector", payload)

    def test_incremental_store_consumes_full_ids_once(self):
        with tempfile.TemporaryDirectory() as folder:
            store = Store(Path(folder) / "evidence.sqlite3")
            try:
                for fact in self.payload()["facts"]:
                    store.append(Trial(
                        fact["trial_id"], "phase-1", "real-environment", "p95_latency_ms:minimize",
                        {"x": 1}, [0], fact["status"], fact["value"], True, True, True,
                    ))
                reflector = Reflector(
                    store, EvidenceWireBackend(HeuristicBackend()), min_batch=1, old_summary_limit=0,
                )
                self.assertTrue(reflector.trigger(["phase_end"], force=True))
                self.assertEqual(store.pending(), [])
                self.assertEqual(store.reflections()[0]["facts"], self.payload()["facts"])
                self.assertFalse(reflector.trigger(["phase_end"], force=True))
                self.assertEqual(len(store.reflections()), 1)
            finally:
                store.close()
