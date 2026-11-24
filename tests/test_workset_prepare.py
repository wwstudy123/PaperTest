from contextlib import redirect_stderr
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.prepare_workset_v2 import cold_copy, tree_hashes, file_hash, validate_identity, main, prepare_command


class WorksetPrepareTests(unittest.TestCase):
    def test_lua_size_options_are_explicit_and_password_is_not_in_argv(self):
        command = prepare_command(Path("/private/sysbench.conf"))
        self.assertIn("--tables=4", command)
        self.assertIn("--table-size=2000000", command)
        self.assertIn("--rand-seed=42", command)
        self.assertFalse(any("password" in arg for arg in command))

    def test_cold_copy_verifies_nested_files_and_preserves_original(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder);source = root / "data";source.mkdir()
            (source / "sub").mkdir();(source / "sub/table.ibd").write_bytes(b"synthetic-pages")
            target = root / "snapshot"
            expected = tree_hashes(source)
            self.assertEqual(cold_copy(source, target, server_stopped=True), expected)
            self.assertEqual(tree_hashes(target), expected)
            self.assertEqual(tree_hashes(source), expected)
            self.assertEqual(len(file_hash(source / "sub/table.ibd")), 64)

    def test_never_copy_live_or_overwrite_existing_destination(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder);source = root / "data";source.mkdir();(source / "f").write_bytes(b"x")
            with self.assertRaises(ValueError):
                cold_copy(source, root / "snap", server_stopped=False)
            target = root / "snap";target.mkdir();(target / "keep").write_text("keep")
            with self.assertRaises(ValueError):
                cold_copy(source, target, server_stopped=True)
            self.assertEqual((target / "keep").read_text(), "keep")

    def test_snapshot_refuses_symlinks_and_empty_sources(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder);source = root / "data";source.mkdir()
            with self.assertRaises(ValueError):
                cold_copy(source, root / "snap", server_stopped=True)
            (source / "link").symlink_to(root)
            with self.assertRaises(ValueError):
                tree_hashes(source)

    def test_identity_handles_mysql_doublewrite_enum_without_weakening_durability(self):
        expected = ["8.0.36", 3308, "/tmp/d", "O_DIRECT", 1, 1, "ON"]
        for value in ("ON", 1, True):
            validate_identity([*expected[:-1], value], Path("/tmp/d"))
        for index, value in ((0, "8.4.0"), (1, 3307), (2, "/tmp/other"),
                             (3, "fsync"), (4, 2), (5, 0), (6, "OFF")):
            changed = expected[:];changed[index] = value
            with self.assertRaises(ValueError):
                validate_identity(changed, Path("/tmp/d"))

    def test_authorization_required_before_environment_access(self):
        with patch("sys.argv", ["prepare_workset_v2"]), redirect_stderr(io.StringIO()), patch(
            "tools.prepare_workset_v2.dotenv_values"
        ) as env:
            with self.assertRaises(SystemExit):
                main()
            env.assert_not_called()
