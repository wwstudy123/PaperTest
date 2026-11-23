from contextlib import redirect_stderr
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np

from admtune.agents import HeuristicBackend
from admtune.agents import Team
from admtune.cli import demo_space
from admtune.contracts import Environment, Objective, fingerprint
from admtune.dynamic import DynamicSession, PooledTeam, main, instance_lock
from admtune.dynamic_runtime import DynamicExecutor, LiveTelemetry, GenerationalRequests
from admtune.replay import MySQLRequests, Request
from admtune.execution import SimulatedExecutor, Measurement
from admtune.controller import Controller, Policy
from admtune.budget import Ledger, Limits
from admtune.memory import Store
from admtune.reflection import Reflector
from admtune.phase_safety import PhaseFence, PhaseTiming
from admtune.replay import ReplayPlan
from admtune.surrogate import GaussianProcess
from admtune.workload import Window, Detection


class DynamicTests(unittest.TestCase):
    def test_failed_new_baseline_does_not_reuse_old_phase_values(self):
        with tempfile.TemporaryDirectory() as t:
            space = demo_space();backend = HeuristicBackend(0)
            store = Store(Path(t) / 'records.sqlite3')
            executor = SimulatedExecutor(space)
            controller = Controller(
                space=space, environment=Environment('synthetic', 'v', 'h', 'd', 's', 'p'),
                objective=Objective(),
                detector=SimpleNamespace(state='stable', changed_count=0), executor=executor,
                team=Team(space, backend), reflector=Reflector(store, backend), store=store,
                ledger=Ledger(Limits(wall_seconds=300)), policy=Policy(),
            )
            try:
                controller.begin_phase('phase-1', [0.0])
                self.assertIsNotNone(controller.baseline)
                controller.gp.fitted = True
                with patch.object(executor, 'measure', return_value=Measurement(None, 'execution_failed', False, False, False)):
                    controller.begin_phase('phase-2', [1.0])
                self.assertIsNone(controller.baseline)
                self.assertIsNone(controller.best_value)
                self.assertFalse(controller.active)
                self.assertFalse(controller.gp.fitted)
            finally:
                store.close()

    def test_instance_lock_rejects_concurrent_mutators(self):
        with tempfile.TemporaryDirectory() as t:
            config = Path(t) / 'mysql.cnf'
            first = instance_lock(config)
            try:
                with self.assertRaises(BlockingIOError):
                    instance_lock(config)
            finally:
                first.close()
            instance_lock(config).close()

    def generational_client(self, *, wrong_readback=False):
        class Connection:
            def __init__(self):
                self.open = True
                self.value = 1
                self.rowcount = 1
            def cursor(self):
                return self
            def __enter__(self):
                return self
            def __exit__(self, *a):
                return False
            def execute(self, sql, args=None):
                if sql.startswith('SET SESSION'):
                    self.value = int(sql.split('=')[1])
            def fetchone(self):
                return (0 if wrong_readback else self.value,)
            def fetchall(self):
                return [('row',)]
            def close(self):
                self.open = False
        def init(client, settings):
            client.settings = settings
            client.local = threading.local()
            client.lock = threading.Lock()
            client.connections = []
        with patch.object(MySQLRequests, '__init__', init):
            client = GenerationalRequests({}, {'sort_buffer_size': 262144})
        def connect():
            connection = Connection()
            client.connections.append(connection)
            return connection
        client.connect = connect
        return client

    def test_config_generation_refreshes_sessions_and_marks_transitions(self):
        client = self.generational_client()
        request = Request(0, 0, 'A', 'read', 1, 1)
        first = client(request)
        self.assertEqual(first['config_generation'], 0)
        self.assertEqual(client.connections[0].value, 262144)
        client.begin_change()
        self.assertTrue(client(request)['config_transition'])
        client.publish({'sort_buffer_size': 524288})
        second = client(request)
        self.assertEqual(second['config_generation'], 1)
        self.assertFalse(second['config_transition'])
        self.assertFalse(client.connections[0].open)
        self.assertEqual(client.connections[1].value, 524288)
        client.publish({'sort_buffer_size': 524288})
        client(request)
        self.assertEqual(len(client.connections), 2)
        client.close()

    def test_session_readback_failure_is_not_silently_accepted(self):
        client = self.generational_client(wrong_readback=True)
        with self.assertRaises(ValueError):
            client(Request(0, 0, 'A', 'read', 1, 1))
        client.close()

    def test_gp_snapshot_round_trip_preserves_predictions(self):
        model = GaussianProcess(config_dimensions=1)
        model.fit([[0, 0], [.5, 0], [1, 0]], [1, 2, 1.5])
        snapshot = model.snapshot(['a', 'b', 'c'])
        loaded = GaussianProcess.from_snapshot(json.loads(json.dumps(snapshot)))
        self.assertEqual(model.predict([[.25, 0]]), loaded.predict([[.25, 0]]))
        self.assertEqual(loaded.version, model.version)
        with self.assertRaises(ValueError):
            model.snapshot(['a'])
        snapshot['target_scale'] += 1
        with self.assertRaises(ValueError):
            GaussianProcess.from_snapshot(snapshot)

    def test_candidate_pool_does_not_repeat_paid_recommendations(self):
        space = demo_space();backend = HeuristicBackend(0)
        team = PooledTeam(space, backend, batch_size=6, max_selected=4)
        team.initialize([0], [], diagnostics={'objective_key': 'p95_latency_ms:minimize'})
        first = team.recommend(space.defaults, 'phase-1', [0], [])
        for _ in range(5):
            self.assertEqual(team.recommend(space.defaults, 'phase-1', [0], []), first)
        self.assertEqual(backend.calls.count('recommender'), 1)
        self.assertGreaterEqual(len(first), 4)

    def test_dynamic_cli_requires_paid_authorization_and_smoke_label(self):
        base = ['dynamic', '--target', '/missing', '--benchmark', '/missing', '--knobs', '/missing',
                '--output', '/missing', '--dedicated-instance']
        for extra in [['--backend', 'llm'], ['--measurement-seconds', '4'], ['--llm-calls', '13']]:
            with patch('sys.argv', base + extra), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as exc:
                    main()
                self.assertEqual(exc.exception.code, 2)

    def runtime(self, *, transition=False, wrong_generation=False):
        space = demo_space();config = space.defaults
        timing = PhaseTiming(warmup_seconds=0, measurement_seconds=4, apply_seconds=0,
                             recovery_seconds=0, proposal_seconds_per_phase=0, trials_per_phase=6)
        fence = PhaseFence(window_seconds=4)
        telemetry = LiveTelemetry(fence);now = [8.0]
        telemetry.elapsed = lambda: now[0]
        fence.observe(Window(4, 8, [[5, 5, 0], [5, 5, 0]]), Detection('stable', 'phase_ready', 'phase-1', [0.0]))
        def wait_until(end, lease):
            now[0] = end
        telemetry.wait_until = wait_until
        def cohort(start, end, lease):
            offers = [{'seq': 0, 'scheduled': 9, 'submitted': True, 'emit_lag_seconds': .001}]
            results = [{'seq': 0, 'rows': 1, 'ok': True, 'config_generation': 0 if wrong_generation else 1,
                        'config_transition': transition, 'session_config_hash': fingerprint(config),
                        'emitted': 9, 'completed': 9.002}]
            # 模拟监测独立推进到完整覆盖窗口，执行器仍必须回报完成才能封存。
            fence.observe(Window(8, 12, [[5, 5, 0], [5, 5, 0]]), Detection('stable', None, 'phase-1', [0.0]))
            return offers, results
        telemetry.cohort = cohort
        actuator = SimpleNamespace(space=space, simulated=True, max_p95_ms=1000, _verify=lambda *a: None)
        executor = DynamicExecutor(actuator, SimpleNamespace(), telemetry, timing,
                                   ReplayPlan(phase_seconds=80), phase_id=lambda: 'phase-1')
        executor._prepare = lambda conf: (SimpleNamespace(invariants={}), 1)
        return executor, config

    def test_dynamic_measurement_uses_real_cohort_semantics(self):
        executor, config = self.runtime()
        result = executor.measure(config, [0.0])
        self.assertEqual(result.status, 'measured')
        self.assertTrue(result.complete and result.effective)
        self.assertAlmostEqual(result.value, 2)
        self.assertEqual(result.metadata['request_count'], 1)
        self.assertEqual(result.metadata['phase_guard']['status'], 'sealed')

    def test_transitional_or_old_session_results_never_become_labels(self):
        for kwargs in [dict(transition=True), dict(wrong_generation=True)]:
            executor, config = self.runtime(**kwargs)
            result = executor.measure(config, [0.0])
            self.assertIsNone(result.value)
            self.assertFalse(result.complete)
            self.assertEqual(result.status, 'phase_invalidated')

    def test_control_worker_full_aba_with_synthetic_executor(self):
        with tempfile.TemporaryDirectory() as t:
            output = Path(t);space = demo_space()
            environment = Environment('synthetic', 'fixture', 'test', 'test', 'test', 'test')
            actuator = SimpleNamespace(space=space, simulated=True)
            timing = PhaseTiming(warmup_seconds=0, measurement_seconds=4, apply_seconds=1,
                                 recovery_seconds=1, proposal_seconds_per_phase=0, trials_per_phase=6)
            plan = ReplayPlan(phase_seconds=80)
            fake = SimulatedExecutor(space, seed=0, objective='latency')
            session = DynamicSession(actuator, None, space, environment, output, plan, timing,
                                     backend_kind='heuristic', seed=0)
            virtual = [8]
            session.telemetry.elapsed = lambda: virtual[0]
            with patch('admtune.dynamic.DynamicExecutor', return_value=fake):
                session.on_start(time.monotonic())
                previous_context = [0.0, 0.0]
                try:
                    for index, context in enumerate([[0.0, 0.0], [6.0, 6.0], [0.0, 0.0]], 1):
                        end = 8 if index == 1 else (index - 1) * 80 + 16
                        last = session.telemetry.fence.last_end
                        if last is not None:
                            for left in range(int(last), end - 4, 4):
                                virtual[0] = left + 4
                                session.on_window(Window(left, left + 4, [[5, 5, 0], [5, 5, 0]]),
                                                  Detection('stable', None, f'phase-{index - 1}', previous_context))
                        virtual[0] = end
                        session.on_window(Window(end - 4, end, [[5, 5, 0], [5, 5, 0]]),
                                          Detection('stable', 'phase_ready', f'phase-{index}', context))
                        expiry = time.monotonic() + 12
                        while time.monotonic() < expiry:
                            session.check_health()
                            path = output / 'progress.json'
                            if path.exists() and json.loads(path.read_text())['finished_phases'] >= index:
                                break
                            time.sleep(.01)
                        else:
                            self.fail('Control worker did not finish phase')
                        previous_context = context
                finally:
                    session.stop()
            self.assertTrue(session.result['simulated'])
            self.assertTrue(session.result['closed_loop_completed'])
            self.assertEqual(session.result['reflection_commits'], 3)
            self.assertEqual(session.result['budget']['llm_calls'], 0)
            self.assertLessEqual(session.result['budget']['trial_slots_used'], 18)
            self.assertGreaterEqual(session.result['budget']['trial_slots_used'], 12)
            self.assertGreater(session.result['prediction_observations'], 0)
            snapshots = list((output / 'gp_models').glob('*.json'))
            self.assertTrue(snapshots)
            for path in snapshots:
                snap = json.loads(path.read_text())
                loaded = GaussianProcess.from_snapshot(snap)
                self.assertEqual(loaded.x.shape[1], 6)
            summaries = json.loads((output / 'dynamic_summary.json').read_text())
            self.assertFalse(summaries['is_thesis_evidence'])


if __name__ == '__main__':
    unittest.main()
