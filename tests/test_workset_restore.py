import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from tools.restore_workset_v2 import restore


class RestoreGuardTests(unittest.TestCase):
    def test_incomplete_preparation_refuses_without_touching_live_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "PaperTest"
            root.mkdir()
            (root / ".env").write_text("ADMTUNE_MYSQL_V2_PASSWORD=private-test-value\n")
            base = Path(tmp) / "base"
            (base / "snapshots/initial").mkdir(parents=True)
            (base / "data").mkdir()
            marker = base / "data/must-remain"
            marker.write_text("original")
            (base / "my.cnf").write_text("config")
            (base / "preparation.json").write_text(json.dumps({"completed": False}))
            (base / "snapshots/initial/manifest.json").write_text("{}")
            driver = Mock()
            with patch.dict("sys.modules", {"pymysql": driver}), patch("tools.restore_workset_v2.subprocess.run") as run:
                result = restore(root, base, Path(tmp) / "audit")
            self.assertFalse(result["completed"])
            self.assertEqual(marker.read_text(), "original")
            driver.connect.assert_not_called()
            run.assert_not_called()
            self.assertNotIn("private-test-value", json.dumps(result))

    def test_existing_audit_directory_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            audit = Path(tmp) / "audit"
            audit.mkdir()
            with patch.dict("sys.modules", {"pymysql": Mock()}), self.assertRaises(FileExistsError):
                restore(Path(tmp), Path(tmp) / "base", audit)
