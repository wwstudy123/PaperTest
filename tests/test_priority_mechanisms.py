import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from tools.audit_priority_mechanisms import inspect_run


class PriorityMechanismTests(unittest.TestCase):
    def fixture(self, variant, events=(), reflections=0):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        (root / "manifest.json").write_text(json.dumps({
            "method": "ADMTune", "ablation": variant, "plan": {"single_slice": False},
        }))
        (root / "trials.jsonl").write_text(json.dumps({
            "trial_id": "old", "phase_id": "phase-1", "status": "measured",
            "complete": True, "effective": True, "value": 50,
        }) + "\n")
        with sqlite3.connect(root / "records.sqlite3") as db:
            db.execute("CREATE TABLE events(seq INTEGER PRIMARY KEY, kind TEXT, payload TEXT)")
            db.execute("CREATE TABLE reflections(seq INTEGER PRIMARY KEY, payload TEXT)")
            for kind, payload in events:
                db.execute("INSERT INTO events(kind,payload) VALUES (?,?)", (kind, json.dumps(payload)))
            for _ in range(reflections):
                db.execute("INSERT INTO reflections(payload) VALUES ('{}')")
        return root

    def test_detects_disabled_gate_still_running(self):
        p = self.fixture("no_gp", [("prescreen", {"config_ids": ["x"], "eligible": []})])
        with self.assertRaisesRegex(ValueError, "prescreening"):
            inspect_run(p)

    def test_detects_old_training_in_no_history(self):
        p = self.fixture("no_history", [("gp_fit", {
            "path": "phase-3-gp-0001.json", "training_ids": ["old"],
        })])
        with self.assertRaisesRegex(ValueError, "prior-phase"):
            inspect_run(p)

    def test_detects_reflection_not_actually_disabled(self):
        with self.assertRaisesRegex(ValueError, "reflection"):
            inspect_run(self.fixture("no_reflection", reflections=1))

    def test_detects_single_slice_not_enabled(self):
        with self.assertRaisesRegex(ValueError, "Single-slice"):
            inspect_run(self.fixture("single_slice"))

    def test_full_empty_audit_does_not_invent_predictions(self):
        row, predictions = inspect_run(self.fixture("full"))
        self.assertEqual(predictions, [])
        self.assertIsNone(row["empirical_interval_coverage"])


if __name__ == "__main__":
    unittest.main()
