from pathlib import Path
import csv
import hashlib
import json
import unittest
import zipfile


ROOT = Path(__file__).resolve().parents[1]


class PlotPackageTests(unittest.TestCase):
    def test_zip_manifest_and_no_private_archives_or_credentials_files(self):
        p = ROOT/"docs/exports/admtune_plot_pack_20261126.zip"
        with zipfile.ZipFile(p) as archive:
            self.assertIsNone(archive.testzip())
            manifest = json.loads(archive.read("manifest.json"))
            self.assertEqual(set(archive.namelist()), set(manifest)|{"README.md", "manifest.json"})
            for name, digest in manifest.items():
                self.assertFalse(Path(name).is_absolute())
                self.assertNotIn("..", Path(name).parts)
                self.assertFalse(name.endswith((".cnf", ".sqlite3", ".tar.gz", ".key", ".pem")))
                self.assertNotIn("/runs/", name)
                self.assertNotIn("/configs/local/", name)
                self.assertNotIn(".env", Path(name).name)
                self.assertEqual(hashlib.sha256(archive.read(name)).hexdigest(), digest)
                self.assertEqual(hashlib.sha256((ROOT/name).read_bytes()).hexdigest(), digest)

    def test_reference_endpoints_are_explicitly_not_one_protocol(self):
        with (ROOT/"docs/experiment_data/research_plot_pack_20261126/published_reference_endpoints.csv").open() as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 6)
        self.assertTrue(all(r["same_protocol_across_entire_table"] == "False" for r in rows))
        published = [r for r in rows if "published" in r["endpoint_kind"]]
        self.assertEqual(len(published), 4)
        self.assertTrue(all("AgenticDB Fig5" in r["source"] for r in published))

    def test_interrupted_slot_is_not_performance_or_training_data(self):
        folder = ROOT/"docs/experiment_data/composite_pair_repeat_001"
        with (folder/"observations.csv").open() as f:
            rows = list(csv.DictReader(f))
        interrupted = [r for r in rows if r["valid"] == "False"]
        self.assertEqual(len(interrupted), 1)
        self.assertEqual(interrupted[0]["trial"], "7")
        self.assertEqual(interrupted[0]["tps"], "")
        self.assertEqual(interrupted[0]["objective_value"], "")
        v = json.loads((folder/"verification.json").read_text())
        self.assertEqual(v["gp_training_count"], 6)
        self.assertFalse(v["recovery"]["recovered_final_used_for_training"])
        self.assertFalse(v["recovery"]["strict_uninterrupted_pair"])
        self.assertEqual(v["accepted_at"], [])

    def test_package_has_ten_figure_groups_and_explicit_remaining_experiments(self):
        folder = ROOT/"docs/experiment_data/research_plot_pack_20261126"
        with (folder/"figure_index.csv").open() as f:
            figures = list(csv.DictReader(f))
        self.assertEqual(len(figures), 10)
        for r in figures:
            for suffix in (".png", ".pdf"):
                self.assertTrue((ROOT/"docs/figures"/(r["figure"]+suffix)).is_file())
        with (folder/"requirements_status.csv").open() as f:
            requirements = list(csv.DictReader(f))
        self.assertTrue(any("六方法" in r["requirement"] and r["status"] == "尚未完成"
                            for r in requirements))
        self.assertTrue(any("消融" in r["requirement"] and r["status"] == "尚未完成"
                            for r in requirements))
