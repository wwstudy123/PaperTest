from contextlib import redirect_stderr
from dataclasses import asdict
from decimal import Decimal
import io
from pathlib import Path
from types import SimpleNamespace
import threading
import unittest
from unittest.mock import Mock, MagicMock, patch

from admtune.replay import ReplayPlan, Request, RangeRequest, make_schedule, monitor, request_channels, MySQLRequests
from admtune.dynamic_runtime import GenerationalRequests
from admtune.space import Space, Knob
from admtune.workset_probe import probe_plan, check_preparation, validate_analyze, main
from tools.prepare_workset_v2 import DB


def space():
    return Space([
        Knob("innodb_buffer_pool_size", "integer", 1024**3, 512 * 1024**2, 4 * 1024**3, step=128 * 1024**2),
        Knob("innodb_io_capacity", "integer", 200, 100, 2000, step=100),
        Knob("innodb_io_capacity_max", "integer", 2000, 2000, 6000, step=500),
        Knob("sort_buffer_size", "integer", 262144, 262144, 8 * 1024**2, step=262144),
    ])


class WorksetProbeTests(unittest.TestCase):
    def test_statistics_refresh_covers_exactly_the_dedicated_tables(self):
        rows = [[f"{DB}.sbtest{i}", "analyze", "status", "OK"] for i in range(1, 5)]
        validate_analyze(rows)
        for changed in (rows[:-1], [*rows[:-1], rows[0]],
                        [*rows[:-1], [f"{DB}.sbtest4", "analyze", "status", "Failed"]]):
            with self.assertRaises(ValueError):
                validate_analyze(changed)

    def test_point_request_metadata_and_bindings_are_unchanged(self):
        r = Request(0, 1, "A", "read", 1, 3)
        self.assertEqual(asdict(r), {"seq": 0, "scheduled": 1, "phase": "A",
                                    "operation": "read", "table": 1, "row": 3})
        self.assertEqual(r.parameters, (3,))
        self.assertEqual(r.sql, "SELECT c FROM sbtest1 WHERE id=%s")

    def test_ranges_never_run_off_the_dataset_and_all_are_read_only(self):
        for profile, top_k in (("range-scan", 0), ("range-sort", 8192)):
            p = ReplayPlan(scenario="steady", read_percent=100, query_profile=profile,
                           range_rows=32768, top_k=top_k, table_size=2_000_000)
            requests = make_schedule(p)
            self.assertEqual(requests, make_schedule(p))
            self.assertTrue(all(isinstance(r, RangeRequest) and r.operation == "read" for r in requests))
            self.assertTrue(all(1 <= r.parameters[0] < r.parameters[1] <= p.table_size for r in requests))
            catalog, _, _ = monitor(p);channels = request_channels(p, catalog)
            self.assertTrue(all(channels[r.template_sql] != catalog.unknown_channel for r in requests))

    def test_profiles_are_explicit_not_silent_changes_to_point_or_dynamic_mix(self):
        for args in [
            {"query_profile": "invented"}, {"range_rows": 1}, {"range_rows": False},
            {"query_profile": "range-scan", "range_rows": 100},
            {"scenario": "steady", "read_percent": 90, "query_profile": "range-scan", "range_rows": 100},
            {"scenario": "steady", "read_percent": 100, "query_profile": "range-sort", "range_rows": 100, "top_k": 0},
            {"scenario": "steady", "read_percent": 100, "query_profile": "range-sort", "range_rows": 100, "top_k": 100},
        ]:
            with self.assertRaises(ValueError):
                ReplayPlan(**args)

    def test_sort_result_depends_on_top_k_order_and_is_small(self):
        r = RangeRequest(0, 0, "A", "read", 2, 5, 32768, 8192)
        self.assertIn("ORDER BY c,id LIMIT 8192", r.sql)
        self.assertIn("SUM(k)", r.sql)
        self.assertEqual(r.parameters, (5, 32772))
        r.validate_result([(8192, Decimal("123456789"))])
        for result in ([], [(8191, 1)], [(8192, None)], [(8192, 1, 2)]):
            with self.assertRaises(ValueError):
                r.validate_result(result)

    def test_scan_reads_uncovered_payload_and_validates_whole_range_count(self):
        r = RangeRequest(0, 0, "A", "read", 1, 5, 32768)
        self.assertIn("SUM(OCTET_LENGTH(c))", r.sql)
        r.validate_result([(32768, Decimal("12"), Decimal("3899392"))])
        with self.assertRaises(ValueError):
            r.validate_result([(32767, 12, 23)])
        with self.assertRaises(ValueError):
            RangeRequest(0, 0, "A", "write", 1, 1, 10)

    def test_mysql_client_checks_aggregate_and_returns_result_fingerprint(self):
        client = object.__new__(MySQLRequests)
        cursor = MagicMock();cursor.fetchall.return_value = [(8192, Decimal("987654"))]
        connection = MagicMock();connection.open = True
        connection.cursor.return_value.__enter__.return_value = cursor
        client.local = SimpleNamespace(connection=connection)
        r = RangeRequest(0, 0, "A", "read", 1, 5, 32768, 8192)
        result = client(r)
        self.assertEqual(result["rows"], 1)
        self.assertEqual(result["aggregate_count"], 8192)
        self.assertEqual(len(result["result_fingerprint"]), 64)
        cursor.execute.assert_called_once_with(r.sql, r.parameters)
        cursor.fetchall.return_value = [(10, Decimal("9"))]
        with self.assertRaises(ValueError):
            client(r)

    def test_generation_metadata_preserves_verified_query_result_fingerprint(self):
        client = object.__new__(GenerationalRequests)
        client.lock = threading.Lock();client.generation = 1;client.changing = False
        client.current = {"sort_buffer_size": 262144};client.local = SimpleNamespace(generation=1)
        with patch.object(MySQLRequests, "__call__", return_value={"rows": 1, "result_fingerprint": "checked"}):
            result = client(Request(0, 0, "A", "read", 1, 1))
        self.assertEqual(result["result_fingerprint"], "checked")
        self.assertEqual(result["config_generation"], 1)
        self.assertFalse(result["config_transition"])

    def test_frozen_plan_has_two_repeated_single_parameter_probes_per_block(self):
        s = space();plan = probe_plan(s, s.defaults)
        self.assertEqual(len(plan), 10)
        for start in (0, 5):
            block = plan[start:start+5]
            self.assertEqual([p["kind"] for p in block], ["reference", "probe", "reference", "probe", "reference"])
            self.assertEqual(len({p["seed"] for p in block}), 1)
            self.assertTrue(all(p["read_percent"] == 100 and p["rate"] == 10 for p in block))
            for p in block:
                self.assertEqual(sum(p["config"][k] != s.defaults[k] for k in s.names),
                                 int(p["kind"] == "probe"))
        with self.assertRaises(ValueError):
            probe_plan(s, {**s.defaults, "innodb_buffer_pool_size": 2 * 1024**3})

    def test_refuse_unrestored_or_wrong_dataset(self):
        p = {"completed": True, "snapshot_restore_verified": True, "port": 3308,
             "database": DB}
        s = {"port": 3308, "database": DB}
        w = SimpleNamespace(tables=4, table_size=2_000_000)
        check_preparation(p, s, w)
        for change in ({"completed": False}, {"snapshot_restore_verified": False}, {"port": 3307}):
            with self.assertRaises(ValueError):
                check_preparation({**p, **change}, s, w)

    def test_cli_authorization_precedes_credentials_and_file_access(self):
        argv = ["probe", "--target", "/missing", "--benchmark", "/missing", "--knobs", "/missing", "--output", "/missing"]
        with patch("sys.argv", argv), redirect_stderr(io.StringIO()), patch("admtune.workset_probe.load_dotenv") as env:
            with self.assertRaises(SystemExit):
                main()
            env.assert_not_called()
