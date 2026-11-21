from dataclasses import fields
import importlib.util
from pathlib import Path
import unittest

from auditor import AuditorSettings, TuningAuditor, DB_PHASE, STOP_PHASE, allowed_actions_for_phase
from config import AppConfig, TargetConfig
from dbms_policy import enforce_dbms_only
from models import BenchmarkResult
from validators.safety_guard import SafetyGuard


ROOT = Path(__file__).resolve().parents[1]


class DBMSPolicyTests(unittest.TestCase):
    def test_semantics_cannot_be_relaxed(self):
        for key in ["sync_binlog", "innodb_doublewrite", "skip-log-bin", "vm.swappiness"]:
            with self.assertRaises(ValueError):
                enforce_dbms_only("db_config", {key: 0})
        enforce_dbms_only("db_config", {"innodb_io_capacity": 400})

    def test_only_database_actions_exist(self):
        self.assertEqual(allowed_actions_for_phase(DB_PHASE), {"db_config", "repeat_benchmark"})
        for phase in ["os_sysctl", "os_control", "unknown", STOP_PHASE]:
            self.assertEqual(allowed_actions_for_phase(phase), set())
        for action in ["os_config", "os_control", "shell"]:
            self.assertFalse(SafetyGuard().validate(action, {}).passed)

    def test_database_plateau_stops_without_layer_switch(self):
        auditor = TuningAuditor(AuditorSettings(min_db_rounds=2, db_plateau_patience=2))
        rounds = [
            {"proposal": {"action_type": "db_config"}, "metadata": {"phase": "db"}, "decision": "rollback"}
            for _ in range(2)
        ]
        decision = auditor.audit(rounds=rounds, current_phase=DB_PHASE, best_result=BenchmarkResult(score=1))
        self.assertTrue(decision.stop)
        self.assertTrue(decision.plateau_detected)
        self.assertEqual(decision.next_phase, STOP_PHASE)

    def test_legacy_layer_recommendation_cannot_change_capabilities(self):
        auditor = TuningAuditor(AuditorSettings(min_db_rounds=10, model_recommendation_min_rounds=1))
        result = auditor.audit(
            rounds=[{"proposal": {"action_type": "db_config", "auditor_recommendation": {"next_phase": "os_sysctl"}},
                     "decision": "accepted", "metadata": {"phase": "db"}}],
            current_phase=DB_PHASE, best_result=BenchmarkResult(score=1),
        )
        self.assertFalse(result.stop)
        self.assertEqual(result.next_phase, DB_PHASE)

    def test_disabled_auditor_cannot_bypass_scope(self):
        result = TuningAuditor(AuditorSettings(enabled=False)).audit(
            rounds=[], current_phase="os_control", best_result=BenchmarkResult(score=1),
        )
        self.assertTrue(result.stop)

    def test_removed_modules_and_configuration_fields(self):
        for name in [
            "tools.apply_os_config", "tools.apply_os_controls",
            "tools.collect_os_controls", "validators.os_control_validator",
        ]:
            self.assertIsNone(importlib.util.find_spec(name))
        for cls in [AppConfig, TargetConfig, AuditorSettings]:
            names = {f.name for f in fields(cls)}
            self.assertFalse(names & {"os", "os_knobs", "os_controls", "dbms_only", "min_os_rounds"})
        with self.assertRaises(ValueError):
            AuditorSettings(initial_phase="os_sysctl")

    def test_no_kernel_mutation_commands_in_runtime_sources(self):
        for directory in ["admtune", "connectors", "tools", "validators"]:
            for path in (ROOT / directory).rglob("*.py"):
                self.assertNotIn("sysctl -w", path.read_text(), str(path))
                self.assertNotIn("apply_os_config", path.read_text(), str(path))


if __name__ == "__main__":
    unittest.main()
