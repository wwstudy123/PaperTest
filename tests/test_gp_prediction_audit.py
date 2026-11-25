from copy import deepcopy
import json
import math
from pathlib import Path
import tempfile
import unittest

from admtune.gp_prediction_audit import GPPredictionAudit
from admtune.space import Knob, Space


class PredictionAuditTests(unittest.TestCase):
    def rows(self):
        return [{"seq": i+1, "config": {"buffer": i % 3+1}, "valid": True,
                 "joint_value": math.exp(.1*(i % 3)+.02*(i//3)),
                 "measurement": {"value": math.exp(.1*(i % 3)+.02*(i//3))*10,
                    "metadata": {"p95_latency_ms": 10}, "status": "measured",
                    "complete": True, "effective": True, "constraints_ok": True}}
                for i in range(6)]

    def make(self, output):
        return GPPredictionAudit(Space([Knob("buffer", "integer", 1, 1, 3)]), "env", output)

    def test_bootstrap_has_no_forecasts_or_extra_measurement(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)/"audit"; audit = self.make(output)
            self.assertIsNone(audit.prepare(self.rows()[:5], {"buffer": 1}, 10))
            self.assertFalse(output.exists())
            self.assertFalse(audit.legacy.fitted)

    def test_sealed_prediction_only_uses_completed_past_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)/"audit"; audit = self.make(output)
            sealed = audit.prepare(self.rows(), {"buffer": 1}, 20)
            self.assertTrue((output/"before-000007.json").is_file())
            self.assertEqual(sealed, json.loads((output/"before-000007.json").read_text()))
            self.assertEqual(sealed["training_sequences"], list(range(1, 7)))
            self.assertEqual(sealed["next_seq"], 7)
            self.assertEqual(sealed["environment_key"], "env")
            self.assertNotIn("actual_objective", sealed)
            self.assertEqual(len(sealed["models"]["legacy"]["snapshot"]["target"]), 6)
            self.assertEqual(len(sealed["models"]["repeat_aware"]["snapshot"]["gp"]["target"]), 6)
            with self.assertRaises(ValueError):
                audit.prepare(self.rows(), {"buffer": 1}, 21)

    def test_previous_test_result_may_train_next_prediction_but_no_future_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            audit = self.make(Path(tmp))
            rows = self.rows()
            audit.prepare(rows, {"buffer": 1}, 20)
            seventh = deepcopy(rows[0])
            seventh.update(seq=7, joint_value=1.1)
            seventh["measurement"]["value"] = 11
            rows.append(seventh)
            sealed = audit.prepare(rows, {"buffer": 2}, 30)
            self.assertEqual(sealed["next_seq"], 8)
            self.assertEqual(sealed["training_sequences"], list(range(1, 8)))
            old = sealed["models"]["legacy"]["snapshot"]
            new = sealed["models"]["repeat_aware"]["snapshot"]["gp"]
            self.assertEqual(old["version"], 2)
            self.assertFalse(old["optimizer"]["optimized"])
            self.assertTrue(new["optimizer"]["optimized"])
            self.assertEqual(len(new["target"]), 7)
            self.assertNotIn("repeat-8", new["training_ids"])

    def test_bad_past_scores_or_measurements_are_not_silently_used(self):
        for change in ("invalid", "wrong_score", "gap"):
            with tempfile.TemporaryDirectory() as tmp:
                audit = self.make(Path(tmp)); rows = self.rows()
                if change == "invalid":
                    rows[0]["valid"] = False
                elif change == "wrong_score":
                    rows[0]["joint_value"] += 1
                else:
                    rows[0]["seq"] = 0
                with self.assertRaises(ValueError):
                    audit.prepare(rows, {"buffer": 1}, 20)
                self.assertFalse(list(Path(tmp).glob("*.json")))
