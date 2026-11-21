import unittest

from auditor import (
    AuditorSettings, AuditDecision, TuningAuditor, DB_PHASE, STOP_PHASE,
    OS_SYSCTL_PHASE, OS_CONTROL_PHASE, allowed_actions_for_phase,
)
from models import BenchmarkResult
from unittest.mock import patch
from dbms_policy import enforce_dbms_only


class DBMSPolicyTests(unittest.TestCase):
    def test_semantics_cannot_be_relaxed(self):
        for key in ["sync_binlog", "innodb_doublewrite", "skip-log-bin", "vm.swappiness"]:
            with self.assertRaises(ValueError):
                enforce_dbms_only("db_config", {key: 0})
        enforce_dbms_only("db_config", {"innodb_io_capacity": 400})
        with self.assertRaises(ValueError):
            enforce_dbms_only("os_config", {})

    def test_no_os_action_capability(self):
        self.assertEqual(
            allowed_actions_for_phase(DB_PHASE, dbms_only=True),
            {"db_config", "repeat_benchmark"},
        )
        for phase in [OS_SYSCTL_PHASE, OS_CONTROL_PHASE, "unknown", STOP_PHASE]:
            self.assertEqual(allowed_actions_for_phase(phase, dbms_only=True), set())

    def test_os_recommendation_becomes_stop(self):
        auditor = TuningAuditor(AuditorSettings(dbms_only=True))
        transition = AuditDecision(DB_PHASE, OS_SYSCTL_PHASE, "switch", "plateau")
        with patch.object(auditor, "_audit", return_value=transition):
            decision = auditor.audit(
                rounds=[], current_phase=DB_PHASE, best_result=BenchmarkResult(score=1),
            )
        self.assertTrue(decision.stop)
        self.assertEqual(decision.next_phase, STOP_PHASE)

    def test_disabled_auditor_cannot_bypass_scope(self):
        auditor = TuningAuditor(AuditorSettings(dbms_only=True, enabled=False))
        result = auditor.audit(
            rounds=[], current_phase=OS_CONTROL_PHASE, best_result=BenchmarkResult(score=1),
        )
        self.assertTrue(result.stop)

    def test_upstream_os_mode_still_available_explicitly(self):
        self.assertIn("os_config", allowed_actions_for_phase(OS_SYSCTL_PHASE))


if __name__ == "__main__":
    unittest.main()
