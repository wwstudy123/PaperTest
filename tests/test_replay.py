import csv
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from concurrent.futures import Future

from admtune.replay import ReplayPlan, make_schedule, monitor, replay
from admtune.replay_report import export_replay
from admtune.workload import Window


class Clock:
    def __init__(self):
        self.now = 100.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += max(0, seconds)


class Pool:
    defer = False

    def __init__(self, max_workers):
        self.pending = []

    def __enter__(self):
        return self

    def submit(self, fn, *args):
        future = Future()
        if self.defer:
            self.pending.append((future, fn, args))
        else:
            future.set_result(fn(*args))
        return future

    def __exit__(self, *args):
        for future, fn, args in self.pending:
            future.set_result(fn(*args))


class DeferredPool(Pool):
    defer = True


class ReplayTests(unittest.TestCase):
    def test_schedule_is_frozen_and_reproducible(self):
        p = ReplayPlan()
        requests = make_schedule(p)
        self.assertEqual(requests, make_schedule(p))
        self.assertNotEqual(requests, make_schedule(replace(p, seed=43)))
        self.assertEqual(len(requests), 720)
        for index, expected_reads in enumerate([216, 24, 216]):
            phase = [r for r in requests if index * 24 <= r.scheduled < (index + 1) * 24]
            self.assertEqual(len(phase), 240)
            self.assertEqual(sum(r.operation == 'read' for r in phase), expected_reads)
            self.assertTrue(all(1 <= r.table <= 4 and 1 <= r.row <= 100000 for r in phase))

    def test_intensity_only_preserves_composition(self):
        p = ReplayPlan(scenario='intensity')
        r = make_schedule(p)
        self.assertEqual(len(r), 1200)
        for index, count in enumerate([240, 720, 240]):
            phase = [x for x in r if index * 24 <= x.scheduled < (index + 1) * 24]
            self.assertEqual(len(phase), count)
            self.assertEqual(sum(x.operation == 'read' for x in phase), count // 2)

    def test_monitor_frozen_before_test_and_detects_both_scenarios(self):
        for scenario in ['composition', 'intensity']:
            p = ReplayPlan(scenario=scenario)
            catalog, detector, metadata = monitor(p)
            self.assertEqual(catalog.cluster_count, 2)
            self.assertEqual(len(catalog.mapping), 8)
            events = []
            for start in range(0, p.duration, p.window_seconds):
                counts = [[0] * 3 for _ in range(p.slices)]
                for r in make_schedule(p):
                    if start <= r.scheduled < start + p.window_seconds:
                        counts[int((r.scheduled - start) / 2)][catalog.channel(r.template_sql)] += 1
                d = detector.observe(Window(start, start + 4, counts))
                if d.event == 'phase_ready':
                    events.append(start + 4)
            self.assertEqual(events, [8, 40, 64])
            self.assertEqual(metadata['encoder']['source'], 'external_scheduled')

    def test_invalid_plans(self):
        for kwargs in [dict(scenario='bad'), dict(phase_seconds=4), dict(base_rate=3),
                       dict(slices=3), dict(workers=0), dict(base_rate=True),
                       dict(max_emit_lag_seconds=float('nan')), dict(high_rate=10)]:
            with self.assertRaises(ValueError):
                ReplayPlan(**kwargs)

    def run_fake(self, output, execute=lambda r: 1, pool=Pool, plan=None):
        clock = Clock()
        with patch('admtune.replay.time.monotonic', clock.monotonic), patch('admtune.replay.time.sleep', clock.sleep), patch('admtune.replay.ThreadPoolExecutor', pool):
            return replay(plan or ReplayPlan(), execute, output)

    def test_steady_plan_has_no_actual_load_switch(self):
        p = ReplayPlan(scenario='steady')
        requests = make_schedule(p)
        self.assertEqual(len(requests), 720)
        self.assertEqual({r.phase for r in requests}, {'A'})
        for second in range(p.duration):
            rows = [r for r in requests if second <= r.scheduled < second + 1]
            self.assertEqual(len(rows), 10)
            self.assertEqual(sum(r.operation == 'read' for r in rows), 5)

    def test_transient_plan_is_two_one_second_pulses(self):
        p = ReplayPlan(scenario='transient')
        requests = make_schedule(p)
        self.assertEqual(len(requests), 760)
        for second in range(p.duration):
            rows = [r for r in requests if second <= r.scheduled < second + 1]
            self.assertEqual(len(rows), 30 if second in {24, 48} else 10)
            self.assertEqual(sum(r.operation == 'read' for r in rows), len(rows) // 2)

    def test_negative_controls_suppress_persistent_changes(self):
        for scenario in ['steady', 'transient']:
            with tempfile.TemporaryDirectory() as t:
                summary = self.run_fake(Path(t), plan=ReplayPlan(scenario=scenario))
                self.assertTrue(summary['complete'])
                self.assertTrue(summary['detector_matches_plan'])
                self.assertEqual(summary['phase_ready'], [{'end': 8, 'phase_id': 'phase-1'}])
                self.assertEqual(summary['detector_event_counts'].get('change_confirmed', 0), 0)
                self.assertEqual(summary['detector_event_counts'].get('change_suspected', 0),
                                 2 if scenario == 'transient' else 0)
                self.assertFalse(summary['three_phases_detected'])
                self.assertEqual(export_replay(Path(t))['windows'], 18)

    def test_execution_export_and_checksums(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            summary = self.run_fake(root)
            self.assertTrue(summary['simulated'])
            self.assertTrue(summary['complete'])
            self.assertTrue(summary['detector_matches_plan'])
            self.assertEqual(summary['counts']['succeeded'], 720)
            report = export_replay(root)
            self.assertEqual(report['requests'], 720)
            self.assertEqual(report['windows'], 18)
            with (root / 'windows.csv').open() as f:
                windows = list(csv.DictReader(f))
            self.assertEqual(sum(int(r['scheduled_offered']) for r in windows), 720)
            self.assertTrue(all(r['cohort_failed'] == '0' for r in windows))
            hashes = json.loads((root / 'checksums.json').read_text())
            self.assertEqual(hashes['requests.csv'], hashlib.sha256((root / 'requests.csv').read_bytes()).hexdigest())
            self.assertEqual((root / 'requests.csv').read_text().count('\n'), 721)

    def test_slow_execution_drops_without_changing_offered_plan(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            summary = self.run_fake(root, pool=DeferredPool)
            self.assertFalse(summary['complete'])
            self.assertEqual(summary['counts']['offered'], 720)
            self.assertEqual(summary['counts']['submitted'], 36)
            self.assertEqual(summary['counts']['dropped'], 684)
            self.assertTrue(summary['three_phases_detected'])
            self.assertEqual(export_replay(root)['requests'], 720)

    def test_query_errors_are_not_success_labels_or_secret_logs(self):
        def failure(request):
            raise ValueError('secret must not be logged')
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            summary = self.run_fake(root, execute=failure)
            self.assertFalse(summary['complete'])
            self.assertEqual(summary['counts']['failed'], 720)
            self.assertNotIn('secret', (root / 'requests.jsonl').read_text())
            export_replay(root)
            with (root / 'windows.csv').open() as f:
                self.assertTrue(all(r['cohort_e2e_p95_ms'] == '' for r in csv.DictReader(f)))

    def test_duplicate_evidence_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            self.run_fake(root)
            first = (root / 'requests.jsonl').read_text().splitlines()[0]
            with (root / 'requests.jsonl').open('a') as f:
                f.write(first + '\n')
            with self.assertRaises(ValueError):
                export_replay(root)

    def test_completed_rate_uses_completion_time_not_scheduled_time(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            self.run_fake(root)
            records = [json.loads(x) for x in (root / 'requests.jsonl').read_text().splitlines()]
            records[0]['completed'] = 4.1
            (root / 'requests.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in records))
            export_replay(root)
            with (root / 'windows.csv').open() as f:
                windows = list(csv.DictReader(f))
            self.assertEqual(int(windows[0]['completed_success']), 39)
            self.assertEqual(int(windows[1]['completed_success']), 41)


if __name__ == '__main__':
    unittest.main()
