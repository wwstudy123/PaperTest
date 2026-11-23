from contextlib import redirect_stderr
from dataclasses import replace
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
from unittest.mock import Mock

from admtune.calibration import (
    baseline_plan, sensitivity_plan, measurement_stats, choose_rate, process_sample, main,
    completed_sensitivity_parameters, supplement_plan,
)
from admtune.contracts import fingerprint
from admtune.replay import ReplayPlan, make_schedule, monitor, request_channels
from admtune.space import Space, Knob


def space():
    return Space([
        Knob("innodb_buffer_pool_size", "integer", 1024**3, 1024**3, 16 * 1024**3, step=1024**3),
        Knob("innodb_io_capacity", "integer", 200, 100, 2000, step=100),
        Knob("innodb_io_capacity_max", "integer", 2000, 2000, 6000, step=500),
        Knob("sort_buffer_size", "integer", 262144, 262144, 8 * 1024**2, step=262144),
    ])


class CalibrationTests(unittest.TestCase):
    def sensitivity_records(self):
        s = space()
        return [{**p, "valid": True, "value_p95_ms": 2.0, "queue_p95_ms": .1,
                 "service_p95_ms": 2, "constraints_ok": True}
                for p in sensitivity_plan(s, s.defaults, 100)]

    def test_unbracketed_probe_is_not_counted_as_completed(self):
        records = self.sensitivity_records()
        self.assertEqual(completed_sensitivity_parameters(records[:2]), set())
        self.assertEqual(len(completed_sensitivity_parameters(records[:3])), 1)
        self.assertEqual(len(completed_sensitivity_parameters(records)), 4)
        records[2]["seed"] = 601
        with self.assertRaises(ValueError):
            completed_sensitivity_parameters(records[:3])

    def test_supplement_plan_checks_parent_and_selects_only_missing_parameters(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            preflight = {"space_key": "same", "current_config": space().defaults,
                         "config_sha256": "file", "readiness": {"live_schema_hash": "schema"}}
            hardware = {"cpu_model": "same", "cgroup_limits": {}}
            manifest = {"mode": "zero_llm_calibration", "git_dirty": False,
                        "preflight": preflight, "hardware_readonly": hardware}
            summary = {"completed": False, "initial_configuration_restored": True,
                       "error": "Calibration budget reached; no new trial",
                       "llm_calls": 0, "final_config_sha256": "file"}
            records = self.sensitivity_records()[:3]
            for name, value in (("manifest.json", manifest), ("summary.json", summary)):
                (root / name).write_text(json.dumps(value))
            (root / "trials.jsonl").write_text("\n".join(json.dumps(r) for r in records))
            missing, rate, audit = supplement_plan(root, preflight, hardware)
            self.assertEqual(len(missing), 3)
            self.assertEqual(rate, 100)
            self.assertEqual(len(audit["completed_parameters"]), 1)
            plan = sensitivity_plan(space(), space().defaults, rate, missing)
            self.assertEqual(len(plan), 7)
            changed = {
                next(k for k in space().names if p["config"][k] != space().defaults[k])
                for p in plan if p["kind"] == "probe"
            }
            self.assertEqual(changed, set(missing))
            for change in ({"completed": True}, {"initial_configuration_restored": False}, {"error": "SQL failed"}):
                (root / "summary.json").write_text(json.dumps({**summary, **change}))
                with self.assertRaises(ValueError):
                    supplement_plan(root, preflight, hardware)
            (root / "summary.json").write_text(json.dumps(summary))
            with self.assertRaises(ValueError):
                supplement_plan(root, preflight, {"cpu_model": "changed"})

    def test_sensitivity_subset_cannot_invent_or_duplicate_knobs(self):
        for params in ([], ["invented"], ["sort_buffer_size", "sort_buffer_size"]):
            with self.assertRaises(ValueError):
                sensitivity_plan(space(), space().defaults, 100, params)

    def test_channel_lookup_parses_only_predeclared_universe_not_request_count(self):
        from unittest.mock import Mock
        catalog = SimpleNamespace(channel=Mock(return_value=1))
        plan = ReplayPlan(scenario="steady", phase_seconds=28, base_rate=1000, high_rate=1010)
        mapping = request_channels(plan, catalog)
        self.assertEqual(len(mapping), 8)
        self.assertEqual(catalog.channel.call_count, 8)
        self.assertEqual({r.template_sql for r in make_schedule(plan)}, set(mapping))

    def test_cached_channel_map_preserves_original_classification(self):
        plan = ReplayPlan(scenario="steady", read_percent=90)
        catalog, _, _ = monitor(plan)
        cached = request_channels(plan, catalog)
        original = {r.template_sql: catalog.channel(r.template_sql) for r in make_schedule(plan)}
        self.assertEqual(cached, original)

    def test_steady_read_mix_is_exact_and_original_defaults_are_unchanged(self):
        for percent in (10, 50, 90):
            plan = ReplayPlan(scenario="steady", read_percent=percent)
            requests = make_schedule(plan)
            for second in range(plan.duration):
                cohort = [r for r in requests if second <= r.scheduled < second + 1]
                self.assertEqual(sum(r.operation == "read" for r in cohort), percent // 10)
        self.assertEqual(ReplayPlan().read_percent, 50)
        for kwargs in ({"read_percent": 90}, {"scenario": "steady", "read_percent": 15},
                       {"scenario": "steady", "read_percent": True}):
            with self.assertRaises(ValueError):
                ReplayPlan(**kwargs)

    def test_predeclared_plan_is_bounded_and_reproducible(self):
        specs = baseline_plan()
        self.assertEqual(len(specs), 12)
        self.assertEqual(sum(s["kind"] == "repeat" for s in specs), 6)
        self.assertEqual({s["seed"] for s in specs if s["kind"] == "repeat"}, {42, 43, 44})
        for spec in specs:
            plan = ReplayPlan(scenario="steady", phase_seconds=28, base_rate=spec["rate"],
                              high_rate=spec["rate"] + 10, read_percent=spec["read_percent"])
            self.assertEqual(plan.duration, 84)
            self.assertLessEqual(plan.duration * plan.high_rate, 100000)
        s = space()
        probes = sensitivity_plan(s, s.defaults, 1000)
        self.assertEqual(probes, sensitivity_plan(s, s.defaults, 1000))
        self.assertEqual(len(specs) + len(probes), 21)
        self.assertEqual([p["kind"] for p in probes], ["reference", "probe"] * 4 + ["reference"])
        self.assertTrue(all(p["seed"] == 600 for p in probes))
        for spec in probes:
            changes = sum(spec["config"][k] != s.defaults[k] for k in s.names)
            self.assertEqual(changes, 1 if spec["kind"] == "probe" else 0)

    def test_sensitivity_does_not_quietly_become_a_noop(self):
        s = space();initial = {**s.defaults, "innodb_buffer_pool_size": 2 * 1024**3}
        with self.assertRaises(ValueError):
            sensitivity_plan(s, initial, 10)

    def requests(self):
        config = space().defaults
        return config, [
            {"seq": i, "scheduled": 15 + i / 10, "emitted": 15 + i / 10,
             "started": 15 + i / 10 + .001, "completed": 15 + i / 10 + .003,
             "emit_lag_seconds": 0, "ok": True, "rows": 1, "config_generation": 0,
             "config_transition": False, "session_config_hash": fingerprint(config)}
            for i in range(20)
        ]

    def test_measurement_uses_only_declared_cohort(self):
        config, requests = self.requests()
        requests.append({**requests[0], "seq": 99, "scheduled": 1, "completed": 999})
        result = measurement_stats(requests, ReplayPlan(), config, measurement=2)
        self.assertTrue(result["complete_cohort"] and result["effective"])
        self.assertAlmostEqual(result["value_p95_ms"], 3)
        self.assertAlmostEqual(result["queue_p95_ms"], 1)
        self.assertAlmostEqual(result["service_p95_ms"], 2)
        self.assertEqual(result["expected_requests"], 20)

    def test_incomplete_failed_stale_session_transition_and_duplicate_refuse_label(self):
        config, original = self.requests()
        cases = [original[:-1]]
        for change in ({"ok": False}, {"rows": 0}, {"session_config_hash": "wrong"},
                       {"config_generation": 1}, {"config_transition": True},
                       {"emit_lag_seconds": 1}, {"seq": 1}):
            cases.append([{**original[0], **change}, *original[1:]])
        for requests in cases:
            with self.subTest(request=requests[0]):
                self.assertIsNone(measurement_stats(requests, ReplayPlan(), config, measurement=2)["value_p95_ms"])

    def test_rate_selection_excludes_dropped_sla_and_client_queue_dominated_probes(self):
        good = {"read_percent": 10, "valid": True, "constraints_ok": True,
                "queue_p95_ms": .1, "service_p95_ms": 2, "rate": 10}
        records = [good, {**good, "rate": 100}, {**good, "rate": 500, "valid": False},
                   {**good, "rate": 1000, "queue_p95_ms": 20},
                   {**good, "rate": 2000, "constraints_ok": False},
                   {**good, "rate": 3000, "read_percent": 90}]
        self.assertEqual(choose_rate(records, 10), 100)
        with self.assertRaises(ValueError):
            choose_rate([], 10)

    def test_process_stat_parsing_does_not_shift_fields_for_parentheses_in_name(self):
        # /proc stat fields 3..22; utime=14, stime=15, starttime=22.
        fields = ["S"] + [str(i) for i in range(4, 23)]
        def read(path):
            if str(path).endswith("/stat"):
                return "123 (mysql (worker)) " + " ".join(fields)
            return "read_bytes: 100\nwrite_bytes: 200\n"
        with patch.object(Path, "read_text", read), patch("admtune.calibration.os.sysconf", return_value=100):
            sample = process_sample(123)
        self.assertAlmostEqual(sample["cpu_seconds"], .29)
        self.assertEqual(sample["start_ticks"], 22)
        self.assertEqual(sample["read_bytes"], 100)

    def test_cli_refuses_without_dedicated_instance_before_reading_credentials(self):
        args = ["calibration", "--target", "/missing", "--benchmark", "/missing",
                "--knobs", "/missing", "--output", "/missing"]
        with patch("sys.argv", args), redirect_stderr(io.StringIO()), patch(
            "admtune.calibration.load_dotenv"
        ) as env:
            with self.assertRaises(SystemExit):
                main()
            env.assert_not_called()

    def test_failed_calibration_restores_initial_configuration(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder);target = root / "target.yaml";output = root / "run"
            target.write_text("mysql:\n  host: 127.0.0.1\n")
            app = SimpleNamespace(target=SimpleNamespace(
                mysql=SimpleNamespace(config_path=str(root / "my.cnf")),
                hardware_memory_bytes=90 * 1024**3,
            ))
            actuator = Mock()
            actuator.initial = SimpleNamespace(config=space().defaults, file_content="baseline")
            actuator.readback.return_value = space().defaults
            actuator.connector.read_file.return_value = "baseline"
            args = ["calibration", "--target", str(target), "--benchmark", str(root / "b"),
                    "--knobs", str(root / "k"), "--output", str(output), "--dedicated-instance"]
            with patch("sys.argv", args), patch("admtune.calibration.load_dotenv"), patch(
                "admtune.calibration.load_app_config", return_value=app
            ), patch("admtune.calibration.load_space", return_value=space()), patch(
                "admtune.calibration.AgenticDBExecutor", return_value=actuator
            ), patch("admtune.calibration.run_calibration", side_effect=RuntimeError("probe failed")), patch(
                "admtune.calibration.wait_buffer_pool_ready", return_value={"status_code": 0}
            ), patch("builtins.print"):
                self.assertEqual(main(), 1)
            actuator.restore_initial.assert_called_once()
            actuator.close.assert_called_once()
            result = json.loads((output / "summary.json").read_text())
            self.assertFalse(result["completed"])
            self.assertTrue(result["initial_configuration_restored"])
            self.assertEqual(result["llm_calls"], 0)

    def test_existing_output_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder);target = root / "target.yaml";output = root / "run"
            output.mkdir();sentinel = output / "summary.json";sentinel.write_text("original")
            target.write_text("mysql:\n  host: 127.0.0.1\n")
            app = SimpleNamespace(target=SimpleNamespace(
                mysql=SimpleNamespace(config_path=str(root / "my.cnf")),
                hardware_memory_bytes=90 * 1024**3,
            ))
            args = ["calibration", "--target", str(target), "--benchmark", str(root / "b"),
                    "--knobs", str(root / "k"), "--output", str(output), "--dedicated-instance"]
            with patch("sys.argv", args), patch("admtune.calibration.load_dotenv"), patch(
                "admtune.calibration.load_app_config", return_value=app
            ), patch("admtune.calibration.load_space", return_value=space()), patch(
                "admtune.calibration.AgenticDBExecutor"
            ) as create, patch("builtins.print"):
                self.assertEqual(main(), 1)
            create.assert_not_called()
            self.assertEqual(sentinel.read_text(), "original")
