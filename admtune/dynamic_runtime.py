"""同一外生请求流上的动态配置应用和 P95 测量；不启动 Sysbench。"""
from __future__ import annotations

from dataclasses import asdict
import math
import threading
import time

import numpy as np

from .contracts import fingerprint
from .execution import Measurement, RecoveryFailed
from .phase_safety import PhaseAdmissionError
from .replay import MySQLRequests, query_semantics
from .security import redact
from tools.apply_db_config import apply_db_config, _format_mysql_sql_value
from tools.rollback import create_config_backup
from tools.mysql_resize import wait_buffer_pool_ready


class GenerationalRequests(MySQLRequests):
    SESSION_KNOBS = {'sort_buffer_size', 'join_buffer_size', 'read_buffer_size', 'read_rnd_buffer_size'}

    def __init__(self, settings, initial):
        super().__init__(settings)
        self.generation = 0
        self.current = dict(initial)
        self.changing = False

    def begin_change(self):
        with self.lock:
            self.changing = True

    def publish(self, config):
        with self.lock:
            if config != self.current:
                self.generation += 1
            self.current = dict(config)
            self.changing = False
            return self.generation

    def __call__(self, request):
        with self.lock:
            generation, config, transition = self.generation, dict(self.current), self.changing
        if getattr(self.local, 'generation', None) != generation:
            if hasattr(self.local, 'connection') and self.local.connection.open:
                self.local.connection.close()
            conn = self.connect()
            with conn.cursor() as cursor:
                for name in sorted(self.SESSION_KNOBS & set(config)):
                    cursor.execute(f'SET SESSION {name}={_format_mysql_sql_value(config[name])}')
                    cursor.execute(f'SELECT @@SESSION.{name}')
                    if int(cursor.fetchone()[0]) != int(config[name]):
                        raise ValueError('Session parameter readback mismatch')
            self.local.connection, self.local.generation = conn, generation
        result = super().__call__(request)
        details = result if isinstance(result, dict) else {"rows": result}
        with self.lock:
            transition = transition or self.changing or generation != self.generation
        return {**details, 'config_generation': generation, 'config_transition': transition,
                'session_config_hash': fingerprint(config)}


class LiveTelemetry:
    """调度线程只写短记录；调优线程等待条件，不阻塞到达时间表。"""
    def __init__(self, fence):
        self.fence = fence
        self.condition = threading.Condition()
        self.origin = None
        self.stopped = False
        self.offered = {}
        self.results = {}
        self.latest = None
        self.ready = []
        self.windows = []

    def elapsed(self):
        if self.origin is None:
            raise RuntimeError('Replay has not started')
        return time.monotonic() - self.origin

    def offer(self, record):
        with self.condition:
            self.offered[record['seq']] = record
            self.condition.notify_all()

    def result(self, record):
        with self.condition:
            self.results[record['seq']] = record
            self.condition.notify_all()

    def window(self, window, detection):
        self.fence.observe(window, detection)
        with self.condition:
            self.latest = detection
            self.windows.append({'window_end': window.end, **asdict(detection)})
            if detection.event == 'phase_ready':
                self.ready.append((window.end, detection))
            self.condition.notify_all()

    def stop(self):
        self.fence.close()
        with self.condition:
            self.stopped = True
            self.condition.notify_all()

    def wait_until(self, deadline, lease):
        with self.condition:
            while self.elapsed() < deadline:
                if self.stopped or self.fence.snapshot(lease)['status'] == 'rejected':
                    raise PhaseAdmissionError('Phase or stream changed during measurement')
                self.condition.wait(timeout=min(.2, max(.001, deadline - self.elapsed())))
            if self.stopped or self.fence.snapshot(lease)['status'] == 'rejected':
                raise PhaseAdmissionError('Phase or stream ended')

    def cohort(self, start, end, lease, timeout=6):
        expiry = time.monotonic() + timeout
        with self.condition:
            while True:
                offers = [r for r in self.offered.values() if start <= r['scheduled'] < end]
                ready = offers and all(not r['submitted'] or r['seq'] in self.results for r in offers)
                # 主调度必须已越过区间终点，防止尾部尚未发出的请求被漏算。
                emitted_after_end = any(r['scheduled'] >= end for r in self.offered.values())
                if ready and emitted_after_end:
                    return offers, [self.results.get(r['seq']) for r in offers]
                if self.stopped or self.fence.snapshot(lease)['status'] == 'rejected':
                    raise PhaseAdmissionError('Phase changed while draining requests')
                if time.monotonic() >= expiry:
                    raise TimeoutError('Request cohort did not drain completely')
                self.condition.wait(timeout=.1)


class DetectorView:
    def __init__(self, telemetry):
        self.telemetry = telemetry

    @property
    def state(self):
        return 'stable' if self.telemetry.fence.usable else 'waiting_stable'

    @property
    def changed_count(self):
        return 0 if self.telemetry.fence.usable else 1


class DynamicExecutor:
    measurement_protocol = 'scheduled_single_sql'

    def __init__(self, actuator, client, telemetry, timing, plan, *, phase_id):
        if any(k.restart_required for k in actuator.space.knobs.values()):
            raise ValueError('Initial dynamic adapter excludes restart-required knobs')
        self.actuator, self.client, self.telemetry = actuator, client, telemetry
        self.timing, self.plan, self.phase_id = timing, plan, phase_id
        self.simulated = actuator.simulated
        self.estimated_trial_seconds = timing.trial_reserve
        self.last_admitted = False
        self.counter = 0
        self.resize_checks = []

    def readback(self):
        return self.actuator.readback()

    def diagnose(self):
        with self.telemetry.condition:
            end = self.telemetry.windows[-1]['window_end'] if self.telemetry.windows else 0
            recent = [r for r in self.telemetry.offered.values()
                      if end - self.plan.window_seconds <= r['scheduled'] < end]
        return {'objective_key': 'p95_latency_ms:minimize', 'request_protocol': self.measurement_protocol,
                'notes': 'Only scheduled SQL stream; lower measured latency is better.',
                'sql_semantics': query_semantics(self.plan),
                'observed_offered_qps': len(recent) / self.plan.window_seconds,
                'observed_read_share': sum(r['operation'] == 'read' for r in recent) / max(1, len(recent)),
                'observed_sort_share': sum(bool(r.get('top_k')) for r in recent) / max(1, len(recent)),
                'runtime': self.actuator.diagnose().get('runtime_metrics', {}),
                'declared_hardware': getattr(getattr(getattr(self.actuator, "app", None),
                                                     "target", None), "hardware", {}),
                'config_generation': self.client.generation}

    def _prepare(self, config):
        a = self.actuator
        config = a.space.merge(config, {})
        self.client.begin_change()
        before = a.snapshot()
        a.pending = before
        self.last_admitted = True
        self.resize_checks = [wait_buffer_pool_ready(a.db, before.config['innodb_buffer_pool_size'], timeout=60)]
        a.counter += 1
        create_config_backup(a.connector, config_path=a.settings.config_path,
                             round_id=a.counter, run_dir=a.run_dir)
        delta = {name: value for name, value in config.items() if before.config[name] != value}
        if delta:
            apply_db_config(a.connector, a.db, config_path=a.settings.config_path,
                            candidate_config=delta, knob_specs=a.specs)
        a._verify(config, before.invariants)
        self.resize_checks.append(wait_buffer_pool_ready(a.db, config['innodb_buffer_pool_size'], timeout=60))
        generation = self.client.publish(config)
        return before, generation

    def restore(self, config):
        if not self.last_admitted:
            if self.readback() != config:
                raise RecoveryFailed('Configuration changed after refused admission')
            self.client.publish(config)
            return
        self.client.begin_change()
        wait_buffer_pool_ready(self.actuator.db, self.readback()['innodb_buffer_pool_size'], timeout=60)
        self.actuator.restore(config)
        wait_buffer_pool_ready(self.actuator.db, config['innodb_buffer_pool_size'], timeout=60)
        self.client.publish(config)

    def measure(self, config, context):
        self.counter += 1
        self.last_admitted = False
        now = self.telemetry.elapsed()
        deadline = min(self.plan.duration, (math.floor(now / self.plan.phase_seconds) + 1) * self.plan.phase_seconds)
        try:
            lease = self.telemetry.fence.start(
                lease_id=f'dynamic-{self.counter}', now=now, timing=self.timing,
                announced_deadline=deadline, needs_completion=True,
                expected_phase_id=self.phase_id(), expected_context=context,
            )
        except PhaseAdmissionError as exc:
            return Measurement(None, 'phase_admission_refused', False, False, False, {'reason': str(exc)})
        metadata = {'request_protocol': self.measurement_protocol, 'objective_key': 'p95_latency_ms:minimize',
                    'phase_guard': self.telemetry.fence.snapshot(lease)}
        try:
            before, generation = self._prepare(config)
            metadata.update(config_generation=generation, config_hash=fingerprint(config),
                            buffer_pool_ready_checks=self.resize_checks)
            self.telemetry.wait_until(self.telemetry.elapsed() + self.timing.warmup_seconds, lease)
            start = self.telemetry.elapsed()
            end = start + self.timing.measurement_seconds
            self.telemetry.wait_until(end, lease)
            offers, results = self.telemetry.cohort(start, end, lease)
            effective = all(
                r and r['ok'] and r.get('rows') == 1 and r.get('config_generation') == generation
                and not r.get('config_transition', True) and r.get('session_config_hash') == fingerprint(config)
                for r in results
            )
            complete = all(r['submitted'] and r['emit_lag_seconds'] <= self.plan.max_emit_lag_seconds for r in offers)
            metadata.update(measurement_start=start, measurement_end=end, request_count=len(offers),
                            warmup_seconds=self.timing.warmup_seconds, measurement_seconds=self.timing.measurement_seconds)
            self.actuator._verify(config, before.invariants)
            actual_end = self.telemetry.elapsed()
            self.telemetry.fence.complete(lease, actual_end)
            state = self.telemetry.fence.await_seal(lease, timeout=12)
            metadata['phase_guard'] = state
            crosses = lease.start < self.plan.phase_seconds * math.ceil((lease.start + 1e-8) / self.plan.phase_seconds) < end
            if not complete or not effective or state['status'] != 'sealed' or crosses:
                return Measurement(None, 'phase_invalidated', False, effective, False, metadata)
            raw_value = float(np.percentile([1000 * (r['completed'] - r['emitted']) for r in results], 95))
            # 声明的报告精度：1e-6 ms；保留原始值，避免模型抄录浮点尾数造成伪造事实。
            value = round(raw_value, 6)
            metadata['raw_p95_latency_ms'] = raw_value
            return Measurement(value, constraints_ok=value <= self.actuator.max_p95_ms, metadata=metadata)
        except RecoveryFailed:
            raise
        except Exception as exc:
            metadata['reason'] = redact(str(exc))
            metadata['phase_guard'] = self.telemetry.fence.snapshot(lease)
            return Measurement(None, 'phase_invalidated' if isinstance(exc, PhaseAdmissionError) else 'execution_failed',
                               False, False, False, metadata)
