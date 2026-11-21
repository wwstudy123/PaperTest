from dataclasses import replace
import json
import logging
from pathlib import Path
import re
import tempfile
import unittest

import yaml

from auditor import AuditDecision, AuditorSettings, TuningAuditor
from config import load_app_config
from llm_client import LLMClient
from loop import build_prompt, run_agenticdb
from memory.workload_memory import WorkloadMemoryBook
from models import BenchmarkResult
from utils.logging_utils import TraceLogger


ROOT = Path(__file__).resolve().parents[1]


def load_config(target: Path | None = None):
    return load_app_config(
        project_root=ROOT, target_path=target or ROOT / "configs/examples/target.yaml",
        benchmark_path=ROOT / "configs/examples/benchmark.yaml",
        knobs_path=ROOT / "configs/examples/knobs.yaml", dry_run_override=True,
    )


class DatabaseOnlyIntegrationTests(unittest.TestCase):
    def test_removed_target_section_rejected_even_when_empty(self):
        with tempfile.TemporaryDirectory() as folder:
            payload = yaml.safe_load((ROOT / "configs/examples/target.yaml").read_text())
            payload["os"] = {}
            path = Path(folder) / "target.yaml"
            path.write_text(yaml.safe_dump(payload))
            with self.assertRaisesRegex(ValueError, "已删除"):
                load_config(path)

    def test_removed_auditor_options_are_not_silently_ignored(self):
        for key, value in [("dbms_only", False), ("min_os_rounds", 2), ("os_gray_zone_confirm", True)]:
            with self.subTest(key=key), tempfile.TemporaryDirectory() as folder:
                payload = yaml.safe_load((ROOT / "configs/examples/target.yaml").read_text())
                payload["auditor"][key] = value
                path = Path(folder) / "target.yaml"
                path.write_text(yaml.safe_dump(payload))
                with self.assertRaisesRegex(ValueError, "已移除"):
                    load_config(path)

    def test_legacy_initial_phase_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            payload = yaml.safe_load((ROOT / "configs/examples/target.yaml").read_text())
            payload["auditor"]["initial_phase"] = "os_control"
            path = Path(folder) / "target.yaml"
            path.write_text(yaml.safe_dump(payload))
            with self.assertRaisesRegex(ValueError, "只支持数据库"):
                load_config(path)

    def test_prompt_has_no_system_parameter_space(self):
        prompt = build_prompt(
            skill_text=(ROOT / "skills/agenticdb_skill.md").read_text(),
            os_metrics={"cpu_cores": 8}, db_metrics={}, current_config={"sync_binlog": 1, "innodb_io_capacity": 200},
            db_runtime={"dbms": "mysql", "db_version": "8.0.36", "objective": {"primary_metric": "tps"}},
            profile_context={"benchmark": {"objective": {"primary_metric": "tps", "formula": "tps", "direction": "maximize"}}},
            knob_space={"sync_binlog": {"type": "integer"}, "innodb_io_capacity": {"type": "integer"}},
            current_phase="db", round_id=1,
            audit_decision=AuditDecision("db", "db", "continue_db", "继续"),
            baseline_result=BenchmarkResult(1), best_result=BenchmarkResult(1),
            history=[], tuning_history_summary={}, workload_memory_summary={},
        )
        self.assertIn("只允许白名单内的数据库参数", prompt)
        self.assertNotIn("allowed_os_", prompt)
        self.assertNotIn("sysctl", prompt)
        self.assertNotIn("{{", prompt)
        self.assertIn('"cpu_cores": 8', prompt)
        context = json.loads(prompt.split("请依据以下上下文输出一个 JSON 对象：\n", 1)[1])
        self.assertEqual(set(context["allowed_knob_space"]), {"innodb_io_capacity"})

    def test_heuristic_cannot_propose_removed_layers(self):
        client = LLMClient(None, "http://unused", "fixture", dry_run=True)
        self.assertFalse(hasattr(client, "_heuristic_os_proposal"))
        with self.assertRaises(ValueError):
            client.generate_json("", {"current_phase": "os_sysctl"})

    def test_stop_recommendation_stays_in_database_control(self):
        auditor = TuningAuditor(AuditorSettings(model_recommendation_min_rounds=1))
        decision = auditor.audit(
            rounds=[{"metadata": {"phase": "db"}, "decision": "accepted", "proposal": {
                "action_type": "db_config", "auditor_recommendation": {"next_phase": "stop"},
            }}],
            current_phase="db", best_result=BenchmarkResult(1),
        )
        self.assertTrue(decision.stop)
        self.assertTrue(decision.model_recommendation_used)

    def test_markdown_headings_are_chinese(self):
        paths = [ROOT / "README.md", *(ROOT / "docs").glob("*.md"), *(ROOT / "skills").glob("*.md")]
        for path in paths:
            title = next(line for line in path.read_text().splitlines() if line.startswith("# "))
            self.assertRegex(title, r"[\u4e00-\u9fff]", str(path))

    def test_inherited_database_baseline_and_memory_roundtrip(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            app = replace(
                load_config(), history_path=base / "history.json",
                memory_book_path=base / "book.jsonl", runs_dir=base,
            )
            logger = logging.getLogger("test.database-only")
            logger.handlers = [logging.NullHandler()]
            # Two runs exercise both creation and comparison of compatible memory entries.
            for i in range(2):
                directory = base / f"run-{i}"
                directory.mkdir()
                summary = run_agenticdb(
                    app_config=app, max_rounds=2, run_dir=directory, logger=logger,
                    trace_logger=TraceLogger(directory), use_history=False, use_memory_book=True,
                )
                self.assertTrue(summary["dry_run"])
                self.assertIn(summary["final_phase"], {"db", "stop"})
                self.assertNotIn("best_os_config", summary)
                self.assertNotIn("os_knobs", summary["db_runtime"])
                report = Path(summary["report_paths"]["report_md"]).read_text()
                self.assertIn("# AgenticDB 数据库调优运行报告", report)
                self.assertIn("## 调优轮次", report)
                trace = (directory / "trace.jsonl").read_text()
                self.assertNotIn("sysctl -w", trace)
                self.assertNotIn("switch_to_os", trace)
                self.assertNotIn("allowed_os_", trace)
            entries = WorkloadMemoryBook(app.memory_book_path).load_entries("mysql")
            self.assertTrue(entries)
            self.assertTrue(all("best_os_config" not in entry for entry in entries))


if __name__ == "__main__":
    unittest.main()
