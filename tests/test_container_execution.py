from dataclasses import replace
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from config import MySQLSettings, SSHSettings
from connectors.local_connector import LocalConnector
from connectors.ssh_connector import SSHConnector
from llm_client import LLMClient
from tools.mysql_process import restart_mysql_process
from tools.restart_db import restart_db


def ssh_settings():
    return SSHSettings(False, None, None, None, None, "TEST_HOST", "TEST_PORT", "TEST_USER", "TEST_PASSWORD")


class ContainerExecutionTests(unittest.TestCase):
    def test_local_mode_does_not_need_ssh_credentials(self):
        connector = LocalConnector(ssh_settings(), "/tmp/unused.cnf")
        connector.connect()
        result = connector.run("printf 'local-ok'", check=True)
        self.assertEqual(result.stdout, "local-ok")
        connector.close()

    def test_root_does_not_invoke_sudo(self):
        connector = LocalConnector(ssh_settings(), "/tmp/unused.cnf")
        with patch("os.geteuid", return_value=0), patch.object(
            connector, "_wrap_sudo", side_effect=AssertionError("must not use sudo"),
        ):
            self.assertEqual(connector.run("printf ok", sudo=True, check=True).stdout, "ok")

    def test_atomic_file_restore_preserves_mode(self):
        connector = LocalConnector(ssh_settings(), "/tmp/unused.cnf")
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "mysql.cnf"
            connector.write_file(str(path), "before")
            path.chmod(0o600)
            connector.write_file(str(path), "after")
            self.assertEqual(connector.read_file(str(path)), "after")
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            connector.copy_file(str(path), str(path) + ".bak")
            self.assertEqual(connector.read_file(str(path) + ".bak"), "after")

    def test_local_timeout_is_counted_as_failure(self):
        connector = LocalConnector(ssh_settings(), "/tmp/unused.cnf")
        result = connector.run("sleep 3", timeout=0.05)
        self.assertEqual(result.exit_code, 124)

    def test_symbolic_configuration_write_rejected(self):
        connector = LocalConnector(ssh_settings(), "/tmp/unused.cnf")
        with tempfile.TemporaryDirectory() as folder:
            target, link = Path(folder) / "target", Path(folder) / "link"
            target.write_text("untouched")
            link.symlink_to(target)
            with self.assertRaises(ValueError):
                connector.write_file(str(link), "changed")
            self.assertEqual(target.read_text(), "untouched")

    def test_remote_root_does_not_require_sudo(self):
        connector = SSHConnector(ssh_settings(), "/tmp/unused.cnf")
        with patch.dict(os.environ, {"TEST_USER": "root"}):
            self.assertEqual(connector._wrap_sudo("printf ok"), "printf ok")

    def test_process_restart_is_scoped_to_config_and_socket(self):
        settings = MySQLSettings(
            "/tmp/experiment/my.cnf", "unused", "root", "TEST_DB_PASSWORD",
            service_manager="process", mysqld_path="/tmp/experiment/bin/mysqld",
            mysqladmin_path="/tmp/experiment/bin/mysqladmin",
            socket_path="/tmp/experiment/mysql.sock", pid_file="/tmp/experiment/mysql.pid",
        )
        connector = Mock()
        connector.dry_run = True
        restart_mysql_process(connector, settings)
        commands = "\n".join(c.args[0] for c in connector.run.call_args_list)
        self.assertIn("/proc/$pid/cmdline", commands)
        self.assertIn("--defaults-file=/tmp/experiment/my.cnf", commands)
        self.assertIn("--socket=/tmp/experiment/mysql.sock", commands)
        self.assertNotIn("pkill", commands)
        self.assertNotIn("systemctl", commands)

    def test_process_restart_requires_absolute_scoped_paths(self):
        settings = MySQLSettings(
            "/tmp/my.cnf", "unused", "root", "TEST_DB_PASSWORD", service_manager="process",
        )
        with self.assertRaises(ValueError):
            restart_mysql_process(Mock(), settings)

    def test_systemd_mode_remains_explicitly_available_for_db_service(self):
        connector = Mock()
        connector.run.return_value.stdout = "active"
        restart_db(connector, "mysql")
        self.assertIn("systemctl restart mysql", connector.run.call_args_list[0].args[0])

    def test_provider_thinking_option_is_explicit(self):
        with patch.dict(os.environ, {"OPENAI_ENABLE_THINKING": "false"}):
            self.assertFalse(LLMClient.from_env(dry_run=True).enable_thinking)
        client = LLMClient(None, "http://unused", "fixture")
        self.assertIsNone(client.enable_thinking)


if __name__ == "__main__":
    unittest.main()
