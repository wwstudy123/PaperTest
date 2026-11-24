import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from admtune.paper_plot import load_run


class PlotEvidenceTests(unittest.TestCase):
    def make_run(self, root):
        manifest = dict(simulated=False, git_dirty=False, data_snapshot_restored_by_harness=True, method="Default")
        summary = dict(complete=True, initial_configuration_restored=True)
        for name, value in (("manifest.json", manifest), ("summary.json", summary)):
            (root / name).write_text(json.dumps(value))
        (root / "trials.jsonl").write_text("{}\n")
        (root / "curve.csv").write_text("method,valid,tps\nDefault,True,100\n")
        self.rehash(root)

    def rehash(self, root):
        hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir()
                  if p.name != "checksums.json"}
        (root / "checksums.json").write_text(json.dumps(hashes))

    def test_accepts_only_unchanged_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); self.make_run(root)
            self.assertEqual(load_run(root)[0]["method"], "Default")
            (root / "curve.csv").write_text("method,valid,tps\nDefault,True,999\n")
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_run(root)

    def test_no_simulation_or_baseline_renaming(self):
        for change in ({"simulated": True}, {"git_dirty": True}, {"method": "CDBTune"},
                       {"data_snapshot_restored_by_harness": False}):
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); self.make_run(root)
                m = json.loads((root / "manifest.json").read_text()); m.update(change)
                (root / "manifest.json").write_text(json.dumps(m)); self.rehash(root)
                with self.assertRaises(ValueError):
                    load_run(root)

    def test_verification_requires_completed_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); self.make_run(root)
            (root / "summary.json").write_text(json.dumps({"complete": False, "initial_configuration_restored": True}))
            self.rehash(root)
            with self.assertRaises(ValueError):
                load_run(root)
