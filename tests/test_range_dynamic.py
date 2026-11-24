"""统一扫描/排序模板空间的零模型回归；不连接数据库。"""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from admtune.replay import ReplayPlan, make_schedule, monitor, request_channels, query_semantics
import test_replay
from admtune.phase_safety import GuardAcceptance
from admtune.replay import replay


def mixed_plan(**kwargs):
    return ReplayPlan(query_profile="range-mixed", read_percent=100,
                      range_rows=32768, top_k=8192, table_size=2_000_000, **kwargs)


class RangeDynamicTests(unittest.TestCase):
    def test_frozen_schedule_is_read_only_and_has_declared_mix(self):
        p = mixed_plan()
        schedule = make_schedule(p)
        self.assertEqual(schedule, make_schedule(p))
        self.assertNotEqual(schedule, make_schedule(replace(p, seed=43)))
        for index, scans in enumerate((216, 24, 216)):
            rows = [r for r in schedule if index * 24 <= r.scheduled < (index + 1) * 24]
            self.assertEqual(len(rows), 240)
            self.assertEqual(sum(r.top_k == 0 for r in rows), scans)
            self.assertTrue(all(r.operation == "read" and r.parameters[1] <= p.table_size for r in rows))

    def test_catalog_is_independent_of_future_schedule_and_family_has_distinct_channel(self):
        p = mixed_plan()
        with patch("admtune.replay.make_schedule", side_effect=AssertionError("future leakage")):
            catalog, _, metadata = monitor(p)
        self.assertEqual(catalog.cluster_count, 2)
        self.assertEqual(len(catalog.mapping), 8)
        requests = make_schedule(p)
        mapping = request_channels(p, catalog)
        channels = [{mapping[r.template_sql] for r in requests if bool(r.top_k) == family}
                    for family in (False, True)]
        self.assertEqual([len(x) for x in channels], [1, 1])
        self.assertFalse(channels[0] & channels[1])
        self.assertNotIn(catalog.unknown_channel, set(mapping.values()))
        self.assertEqual(metadata, monitor(replace(p, scenario="steady", seed=99))[2])

    def test_mix_switch_and_steady_control(self):
        for scenario, ends in (("composition", [8, 40, 64]), ("steady", [8])):
            with tempfile.TemporaryDirectory() as tmp:
                result = test_replay.ReplayTests().run_fake(Path(tmp), plan=mixed_plan(scenario=scenario))
                self.assertTrue(result["complete"])
                self.assertTrue(result["detector_matches_plan"])
                self.assertEqual([r["end"] for r in result["phase_ready"]], ends)

    def test_invalid_mixed_modes_and_semantics(self):
        for overrides in (dict(scenario="intensity"), dict(scenario="transient"),
                          dict(read_percent=50), dict(top_k=0)):
            kwargs = dict(query_profile="range-mixed", read_percent=100, range_rows=100, top_k=20)
            kwargs.update(overrides)
            with self.assertRaises(ValueError):
                ReplayPlan(**kwargs)
        self.assertIn("Read-only", query_semantics(mixed_plan()))
        self.assertIn("ORDER BY", query_semantics(mixed_plan()))

    def test_mixed_guard_accepts_same_phase_and_rejects_cross_phase(self):
        plan = mixed_plan(phase_seconds=96)
        clock = test_replay.Clock()
        with tempfile.TemporaryDirectory() as tmp:
            with patch("admtune.replay.time.monotonic", clock.monotonic), \
                    patch("admtune.replay.time.sleep", clock.sleep), \
                    patch("admtune.replay.ThreadPoolExecutor", test_replay.Pool):
                result = replay(plan, lambda r: 1, Path(tmp), guard=GuardAcceptance(plan))
            self.assertTrue(result["phase_guard_passed"])
            self.assertTrue(result["detector_matches_plan"])
