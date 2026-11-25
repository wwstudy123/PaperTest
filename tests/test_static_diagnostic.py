import unittest
from unittest.mock import patch
from types import SimpleNamespace

from admtune.static_diagnostic import plan, configurations, counter_delta, DiagnosticConnector, summarize, rate_check


class StaticDiagnosticTests(unittest.TestCase):
    def test_balanced_predeclared_plan(self):
        rows = plan()
        self.assertEqual([r["seq"] for r in rows], list(range(1, 10)))
        for field in ("block", "position"):
            for n in range(1, 4):
                self.assertEqual({r["name"] for r in rows if r[field] == n},
                                 {"Default", "BufferOnly", "ADMTuneConfig"})

    def test_config_isolates_buffer_and_rejects_changed_parent(self):
        initial = dict(innodb_buffer_pool_size=1024**3, innodb_io_capacity=200,
                       innodb_io_capacity_max=2000, sort_buffer_size=262144)
        chosen = {**initial, "innodb_buffer_pool_size": 4*1024**3,
                  "innodb_io_capacity": 1200, "innodb_io_capacity_max": 4500}
        space = SimpleNamespace(merge=lambda base, delta: {**base, **delta})
        configs = configurations(initial, chosen, space)
        self.assertEqual([k for k in initial if initial[k] != configs["BufferOnly"][k]],
                         ["innodb_buffer_pool_size"])
        with self.assertRaises(ValueError):
            configurations(initial, {**chosen, "innodb_io_capacity": 800}, space)

    def samples(self):
        before = {"at": 1, "process": {"start_ticks": 10, "cpu_seconds": 2, "read_bytes": 100, "write_bytes": 0},
                  "cgroup": {"path": "fixture", "nr_periods": 10, "nr_throttled": 2,
                             "usage_seconds": 5, "throttled_seconds": 1},
                  "mysql": {"reads": 10}}
        after = {"at": 61, "process": {"start_ticks": 10, "cpu_seconds": 122, "read_bytes": 1124, "write_bytes": 0},
                 "cgroup": {"path": "fixture", "nr_periods": 610, "nr_throttled": 302,
                            "usage_seconds": 365, "throttled_seconds": 121},
                 "mysql": {"reads": 20}}
        return before, after

    def test_delta_units_and_no_fake_wall_percent(self):
        d = counter_delta(*self.samples())
        self.assertEqual(d["mysqld_cpu_cores"], 2)
        self.assertEqual(d["cgroup_cpu_cores"], 6)
        self.assertEqual(d["throttled_period_fraction"], .5)
        self.assertEqual(d["throttled_seconds"], 120)
        self.assertEqual(d["mysql"]["reads"], 10)

    def test_counter_reset_or_restart_rejected(self):
        before, after = self.samples()
        after["process"]["start_ticks"] = 11
        with self.assertRaises(ValueError):
            counter_delta(before, after)
        before, after = self.samples()
        after["mysql"]["reads"] = 0
        with self.assertRaises(ValueError):
            counter_delta(before, after)

    def test_hook_only_wraps_formal_measurement(self):
        connector = object.__new__(DiagnosticConnector)
        before, after = self.samples()
        with patch.object(DiagnosticConnector, "capture", side_effect=[before, after]) as capture, \
                patch("connectors.local_connector.LocalConnector.run", return_value=SimpleNamespace(ok=True)):
            connector.run("sysbench /script --time=15 run")
            capture.assert_not_called()
            connector.run("sysbench /script --time=60 run")
            self.assertEqual(capture.call_count, 2)
            self.assertEqual(connector.boundaries["delta"]["boundary_elapsed_seconds"], 60)

    def test_empty_aggregates_do_not_invent_values(self):
        self.assertTrue(all(r["n"] == 0 and r["median_tps"] is None for r in summarize([])))

    def test_failed_command_still_captures_boundaries_without_hiding_error(self):
        connector = object.__new__(DiagnosticConnector)
        with patch.object(DiagnosticConnector, "capture", side_effect=self.samples()), \
                patch("connectors.local_connector.LocalConnector.run", side_effect=RuntimeError("fixture")):
            with self.assertRaisesRegex(RuntimeError, "fixture"):
                connector.run("sysbench /script --time=60 run")
            self.assertEqual(connector.boundaries["delta"]["mysqld_read_bytes"], 1024)

    def test_rate_flag_reaches_warmup_and_measurement_without_password_evidence(self):
        connector = object.__new__(DiagnosticConnector)
        connector.fixed_rate = 1000
        connector.capture = None
        with patch("connectors.local_connector.LocalConnector.run", return_value=SimpleNamespace(ok=True)) as run:
            connector.run("sysbench /script --mysql-password='private fixture' --time=15 run")
            connector.run("sysbench /script --mysql-password='private fixture' --time=60 run")
            self.assertTrue(all("--rate=1000" in c.args[0] for c in run.call_args_list))
            self.assertEqual([c["duration"] for c in connector.command_evidence], [15, 60])
            self.assertNotIn("private fixture", str(connector.command_evidence))

    def test_rate_rejects_overrides_and_wrong_value(self):
        connector = object.__new__(DiagnosticConnector)
        connector.fixed_rate = 1000
        with self.assertRaises(ValueError):
            connector.run("sysbench /script --time=60 --rate=900 run")
        connector.fixed_rate = 800
        with self.assertRaises(ValueError):
            connector.run("sysbench /script --time=60 run")

    def test_rate_gate_requires_native_header_and_achieved_load(self):
        log = "Target transaction rate: 1000/sec\n[ 5s ] qlen: 2\n[ 10s ] qlen: 0"
        self.assertTrue(rate_check(log, 995)["passed"])
        self.assertEqual(rate_check(log, 995)["max_reported_queue"], 2)
        self.assertEqual(rate_check("Target transaction rate: 1000/sec\nqueue length: 3, concurrency: 2",
                                    1000)["max_reported_queue"], 3)
        self.assertFalse(rate_check(log, 900)["passed"])
        self.assertFalse(rate_check("fixture", 1000)["passed"])
        self.assertFalse(rate_check(log, None)["passed"])
