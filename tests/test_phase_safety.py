from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from admtune.execution import Measurement
from admtune.phase_safety import PhaseTiming, PhaseFence, PhaseAdmissionError, PhaseGuardedExecutor, GuardAcceptance
from admtune.replay import ReplayPlan, replay, main
from admtune.workload import Window, Detection
from test_replay import Clock, Pool


def observe(fence, start=4, end=8, event='phase_ready', state='stable', phase='phase-1'):
    fence.observe(Window(start, end, [[5, 5, 0], [5, 5, 0]]), Detection(state, event, phase, [0.0]))


def short_timing():
    return PhaseTiming(warmup_seconds=0, measurement_seconds=4, apply_seconds=0,
                       recovery_seconds=0, proposal_seconds_per_phase=0, trials_per_phase=1)


class Inner:
    measurement_protocol = 'scheduled_single_sql'
    estimated_trial_seconds = 4
    simulated = True

    def __init__(self, action):
        self.action = action
        self.calls = self.restores = 0

    def measure(self, config, context):
        self.calls += 1
        self.action()
        return Measurement(3)

    def readback(self):
        return {'x': 1}

    def restore(self, config):
        self.restores += 1


class PhaseSafetyTests(unittest.TestCase):
    def test_plan_includes_all_reserves_and_rejects_short_phase(self):
        timing = PhaseTiming()
        self.assertEqual(timing.trial_reserve, 109)
        self.assertEqual(timing.minimum_phase_seconds, 592)
        with self.assertRaises(ValueError):
            timing.validate_phase(24)
        timing.validate_phase(600)
        with self.assertRaises(ValueError):
            timing.validate_phase(601)

    def test_plan_rejects_invalid_numbers(self):
        for kwargs in [dict(measurement_seconds=0), dict(warmup_seconds=-1), dict(apply_seconds=True),
                       dict(confirm_windows=1), dict(trials_per_phase=0)]:
            with self.assertRaises(ValueError):
                PhaseTiming(**kwargs)

    def test_cli_uses_benchmark_duration_instead_of_hardcoded_75_seconds(self):
        from admtune.phase_safety import main as plan_main
        with tempfile.TemporaryDirectory() as t:
            path = Path(t) / 'benchmark.yaml'
            for warmup, duration, minimum in [(15, 60, 592), (30, 120, 968)]:
                path.write_text(json.dumps({'workload': {'warmup_time': warmup, 'duration': duration}}))
                stdout = io.StringIO()
                with patch('sys.argv', ['phase-plan', '--benchmark', str(path)]), redirect_stdout(stdout):
                    plan_main()
                self.assertEqual(json.loads(stdout.getvalue())['minimum_phase_seconds'], minimum)

    def test_cli_rejects_short_guard_before_files_or_database(self):
        with tempfile.TemporaryDirectory() as t, redirect_stderr(io.StringIO()):
            output = Path(t) / 'must_not_exist'
            with patch('sys.argv', ['replay', '--target', '/nonexistent', '--scenario', 'composition',
                                    '--output', str(output), '--dedicated-instance', '--guard-acceptance']):
                with self.assertRaises(SystemExit) as exc:
                    main()
            self.assertEqual(exc.exception.code, 2)
            self.assertFalse(output.exists())

    def test_full_acceptance_timing_and_no_silent_shortening(self):
        with self.assertRaises(ValueError):
            GuardAcceptance(ReplayPlan())
        guard = GuardAcceptance(ReplayPlan(phase_seconds=96))
        self.assertEqual(guard.timing.warmup_seconds, 15)
        self.assertEqual(guard.timing.measurement_seconds, 60)
        self.assertEqual(guard.timing.minimum_phase_seconds, 96)

    def test_no_admission_until_stable_or_when_stale(self):
        fence = PhaseFence(window_seconds=4)
        with self.assertRaises(PhaseAdmissionError):
            fence.start(lease_id='a', now=8, timing=short_timing())
        observe(fence)
        for kwargs in [dict(now=13), dict(now=7), dict(now=8, expected_phase_id='phase-2'),
                       dict(now=8, expected_context=[1.0])]:
            with self.assertRaises(PhaseAdmissionError):
                fence.start(lease_id='a', timing=short_timing(), **kwargs)

    def test_budget_tail_admission_refused(self):
        fence = PhaseFence(window_seconds=4);observe(fence)
        with self.assertRaises(PhaseAdmissionError):
            fence.start(lease_id='tail', now=8, timing=short_timing(), announced_deadline=12)
        self.assertEqual(fence.leases, [])

    def test_seal_waits_for_complete_covering_window(self):
        fence = PhaseFence(window_seconds=4);observe(fence)
        lease = fence.start(lease_id='good', now=8, timing=short_timing())
        self.assertEqual(lease.status, 'provisional')
        observe(fence, 8, 12, None)
        self.assertEqual(lease.status, 'sealed')
        fence.close()
        self.assertEqual(lease.status, 'sealed')

    def test_suspected_change_precedes_seal_and_rejects(self):
        fence = PhaseFence(window_seconds=4);observe(fence)
        lease = fence.start(lease_id='mixed', now=8, timing=short_timing())
        observe(fence, 8, 12, 'change_suspected')
        self.assertEqual(lease.status, 'rejected')
        self.assertFalse(fence.usable)
        self.assertIsNone(lease.sealed_at_window_end)

    def test_gap_or_wrong_window_rejects(self):
        for start, end in [(12, 16), (8, 10)]:
            fence = PhaseFence(window_seconds=4);observe(fence)
            lease = fence.start(lease_id='gap', now=8, timing=short_timing())
            observe(fence, start, end, None)
            self.assertEqual(lease.reason, 'monitor_gap')

    def test_execution_overrun_requires_later_window(self):
        fence = PhaseFence(window_seconds=4);observe(fence)
        lease = fence.start(lease_id='slow', now=8, timing=short_timing(), needs_completion=True)
        observe(fence, 8, 12, None)
        self.assertEqual(lease.status, 'provisional')
        fence.complete(lease, 13)
        self.assertEqual(lease.status, 'provisional')
        observe(fence, 12, 16, None)
        self.assertEqual(lease.status, 'sealed')
        self.assertEqual(lease.sealed_at_window_end, 16)

    def test_announced_deadline_overrun_is_not_a_valid_label(self):
        fence = PhaseFence(window_seconds=4);observe(fence)
        lease = fence.start(lease_id='overrun', now=8, timing=short_timing(),
                            announced_deadline=16, needs_completion=True)
        observe(fence, 8, 12, None)
        observe(fence, 12, 16, None)
        fence.complete(lease, 15)
        self.assertEqual(lease.status, 'rejected')
        self.assertEqual(lease.reason, 'announced_deadline_overrun')
        with self.assertRaises(ValueError):
            fence.complete(lease, 16)

    def test_per_phase_slots_cannot_be_silently_reused(self):
        fence = PhaseFence(window_seconds=4);observe(fence)
        fence.start(lease_id='first', now=8, timing=short_timing())
        observe(fence, 8, 12, None)
        with self.assertRaises(PhaseAdmissionError):
            fence.start(lease_id='second', now=12, timing=short_timing())

    def wrapper(self, action, *, deadline=None):
        fence = PhaseFence(window_seconds=4);observe(fence)
        now = [8]
        inner = Inner(lambda: action(fence, now))
        wrapped = PhaseGuardedExecutor(inner, fence, short_timing(), clock=lambda: now[0],
                                       phase_id=lambda: 'phase-1', workload_protocol='scheduled_single_sql',
                                       deadline=deadline, seal_timeout=.001)
        return wrapped, inner, fence

    def test_wrapper_accepts_only_sealed_value(self):
        def work(fence, now):
            now[0] = 12
            observe(fence, 8, 12, None)
        wrapped, inner, fence = self.wrapper(work)
        result = wrapped.measure({'x': 1}, [0.0])
        self.assertEqual(result.value, 3)
        self.assertEqual(result.metadata['phase_guard']['status'], 'sealed')
        wrapped.restore({'x': 1})
        self.assertEqual(inner.restores, 1)

    def test_wrapper_invalidates_mixed_labels(self):
        def work(fence, now):
            now[0] = 12
            observe(fence, 8, 12, 'change_suspected')
        wrapped, inner, fence = self.wrapper(work)
        result = wrapped.measure({'x': 1}, [0.0])
        self.assertIsNone(result.value)
        self.assertFalse(result.complete)
        self.assertEqual(result.status, 'phase_invalidated')

    def test_refused_trial_does_not_restore_missing_snapshot(self):
        wrapped, inner, fence = self.wrapper(lambda f, n: None, deadline=lambda: 12)
        result = wrapped.measure({'x': 1}, [0.0])
        self.assertEqual(result.status, 'phase_admission_refused')
        self.assertEqual(inner.calls, 0)
        wrapped.restore({'x': 1})
        self.assertEqual(inner.restores, 0)

    def test_monitor_timeout_fails_closed(self):
        def work(fence, now):
            now[0] = 12
        wrapped, inner, fence = self.wrapper(work)
        result = wrapped.measure({'x': 1}, [0.0])
        self.assertIsNone(result.value)
        self.assertEqual(result.metadata['phase_guard']['reason'], 'monitor_seal_timeout')

    def test_sysbench_cannot_masquerade_as_monitored_single_sql(self):
        inner = Inner(lambda: None);inner.measurement_protocol = 'sysbench_oltp'
        with self.assertRaises(ValueError):
            PhaseGuardedExecutor(inner, PhaseFence(window_seconds=4), short_timing(), clock=lambda: 0,
                                 phase_id=lambda: 'phase-1', workload_protocol='scheduled_single_sql')

    def test_phase_budget_must_cover_executor_estimate(self):
        inner = Inner(lambda: None);inner.estimated_trial_seconds = 20
        with self.assertRaises(ValueError):
            PhaseGuardedExecutor(inner, PhaseFence(window_seconds=4), short_timing(), clock=lambda: 0,
                                 phase_id=lambda: 'phase-1', workload_protocol='scheduled_single_sql')

    def test_replay_acceptance_drives_gate_without_blocking_scheduler(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t);clock = Clock();plan = ReplayPlan(phase_seconds=96)
            with patch('admtune.replay.time.monotonic', clock.monotonic), patch('admtune.replay.time.sleep', clock.sleep), patch('admtune.replay.ThreadPoolExecutor', Pool):
                summary = replay(plan, lambda request: 1, root, guard=GuardAcceptance(plan))
            self.assertTrue(summary['phase_guard_passed'])
            self.assertTrue(summary['complete'])
            evidence = json.loads((root / 'phase_guard.json').read_text())
            good, mixed = evidence['observations']
            self.assertTrue(good['accepted_observation'])
            self.assertEqual(good['measurement_start'], 23)
            self.assertEqual(good['measurement_end'], 83)
            self.assertEqual(good['sealed_at_window_end'], 84)
            self.assertFalse(mixed['accepted_observation'])
            self.assertIsNone(mixed['value'])
            self.assertEqual(evidence['gp_labels_written'], 0)
            self.assertEqual(summary['counts']['offered'], 2880)


if __name__ == '__main__':
    unittest.main()
