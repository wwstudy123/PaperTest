import ast
from pathlib import Path
import tempfile
import unittest

import numpy as np

from admtune.onlinetune_adapter import LinearCatalog, prepare_core
from test_calibration import space


class OnlineTuneAdapterTests(unittest.TestCase):
    def test_linear_not_log_coordinate_and_grid_mapping(self):
        s = space()
        codec = LinearCatalog(s)
        self.assertEqual(codec.decode(codec.encode(s.defaults)), s.defaults)
        for x in (np.zeros(len(s.names)), np.ones(len(s.names)), np.full(len(s.names), .43)):
            cfg = codec.decode(x)
            self.assertEqual(s.merge(cfg, {}), cfg)

    def test_pinned_source_patch_parses_without_running_upstream(self):
        source = Path(__file__).resolve().parents[1] / "runs/upstream_sources/OnlineTune-audit"
        if not source.exists():
            self.skipTest("Optional pinned upstream source not downloaded")
        with tempfile.TemporaryDirectory() as tmp:
            report = prepare_core(source, Path(tmp) / "core")
            self.assertEqual(len(report["patched_sha256"]), 10)
            for path in (Path(tmp) / "core").rglob("*.py"):
                ast.parse(path.read_text())
            text = (Path(tmp) / "core/onlinetune/safe/models/gp.py").read_text()
            self.assertNotIn(" ctive_dims=", text)
            self.assertIn(" active_dims=", text)

    def test_upstream_changes_are_not_silently_loaded(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source"
            (source / "onlinetune").mkdir(parents=True)
            (source / "onlinetune/knobs.py").write_text("# modified source")
            with self.assertRaisesRegex(ValueError, "Unpinned"):
                prepare_core(source, Path(tmp) / "patched")
