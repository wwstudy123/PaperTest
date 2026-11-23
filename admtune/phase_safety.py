"""阶段时长规划与线程安全观测栅栏；不负责推断未来负载或调参。"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import math
from pathlib import Path
import threading

import yaml

from .workload import Detection, Window


@dataclass(frozen=True)
class PhaseTiming:
    window_seconds: int = 4
    confirm_windows: int = 2
    stable_windows: int = 2
    warmup_seconds: int = 15
    measurement_seconds: int = 60
    apply_seconds: int = 15
    recovery_seconds: int = 15
    proposal_seconds_per_phase: int = 30
    trials_per_phase: int = 5

    def __post_init__(self):
        for name in ['window_seconds', 'confirm_windows', 'stable_windows', 'measurement_seconds', 'trials_per_phase']:
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f'Positive integer required: {name}')
        for name in ['warmup_seconds', 'apply_seconds', 'recovery_seconds', 'proposal_seconds_per_phase']:
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f'Nonnegative integer required: {name}')
        if min(self.confirm_windows, self.stable_windows) < 2:
            raise ValueError('At least two confirmation and stability windows required')

    @property
    def detection_reserve(self):
        # 有限计划假设：窗口对齐、切换后立即平稳；不声称噪声下必在此时间稳定。
        return (self.confirm_windows + self.stable_windows) * self.window_seconds

    @property
    def trial_reserve(self):
        # 额外一个完整窗口用于结果封存；不能测完就立即当作有效标签。
        return self.apply_seconds + self.warmup_seconds + self.measurement_seconds + self.window_seconds + self.recovery_seconds

    @property
    def minimum_phase_seconds(self):
        seconds = self.detection_reserve + self.proposal_seconds_per_phase + self.trials_per_phase * self.trial_reserve
        return math.ceil(seconds / self.window_seconds) * self.window_seconds

    def validate_phase(self, phase_seconds):
        if type(phase_seconds) is not int or phase_seconds % self.window_seconds:
            raise ValueError('Phase duration must be integral complete windows')
        if phase_seconds < self.minimum_phase_seconds:
            raise ValueError(f'Phase too short: {phase_seconds}s; need at least {self.minimum_phase_seconds}s for this declared budget')

    def report(self):
        return {**asdict(self), 'detection_reserve_seconds': self.detection_reserve,
                'trial_reserve_seconds': self.trial_reserve, 'minimum_phase_seconds': self.minimum_phase_seconds,
                'calibrated': False,
                'assumption': 'Window-aligned externally scheduled stable phases; late readiness reduces trial capacity, never extends a phase.'}


class PhaseAdmissionError(RuntimeError):
    pass


@dataclass
class Lease:
    lease_id: str
    phase_id: str
    epoch: int
    start: float
    measurement_start: float
    measurement_end: float
    context: list[float]
    status: str = 'provisional'
    reason: str | None = None
    sealed_at_window_end: float | None = None
    needs_completion: bool = False
    execution_finished: bool = True
    announced_deadline: float | None = None
    post_execution_reserve: float = 0
    safety_probe: bool = False


class PhaseFence:
    """只由完整外生窗口推进；疑似变化也作废在途观测，防止延迟确认污染。"""

    def __init__(self, *, window_seconds: float):
        if not math.isfinite(window_seconds) or window_seconds <= 0:
            raise ValueError('Positive window duration required')
        self.window_seconds = window_seconds
        self.lock = threading.RLock()
        self.condition = threading.Condition(self.lock)
        self.epoch = 0
        self.phase_id = None
        self.state = 'initializing'
        self.last_end = None
        self.context = []
        self.usable = False
        self.leases: list[Lease] = []
        self.admitted_per_phase = {}

    def invalidate(self, reason: str):
        with self.lock:
            self.epoch += 1
            self.usable = False
            for lease in self.leases:
                if lease.status == 'provisional':
                    lease.status, lease.reason = 'rejected', reason
            self.condition.notify_all()

    def _seal(self, lease):
        if (lease.status != 'provisional' or not lease.execution_finished
                or self.last_end is None or self.last_end < lease.measurement_end):
            return
        if not self.usable or lease.epoch != self.epoch or lease.phase_id != self.phase_id:
            lease.status, lease.reason = 'rejected', 'phase_epoch_changed'
        else:
            lease.status = 'sealed'
            lease.sealed_at_window_end = self.last_end

    def observe(self, window: Window, detection: Detection):
        with self.lock:
            window.features()
            if self.last_end is not None and window.start < self.last_end:
                raise ValueError('Fence requires chronological nonoverlapping complete windows')
            gap = self.last_end is not None and not math.isclose(window.start, self.last_end, abs_tol=1e-8)
            wrong_width = not math.isclose(window.end - window.start, self.window_seconds, abs_tol=1e-8)
            changed = self.phase_id is not None and detection.phase_id != self.phase_id
            disturbed = detection.event in {'change_suspected', 'change_confirmed'} or detection.state != 'stable'
            if gap or wrong_width or changed or disturbed:
                reason = 'monitor_gap' if gap or wrong_width else 'workload_change_or_unstable'
                self.invalidate(reason)
            self.state, self.phase_id = detection.state, detection.phase_id
            self.context = list(detection.context)
            self.last_end = window.end
            self.usable = not (gap or wrong_width or disturbed) and self.phase_id is not None
            # 只有覆盖观测终点的完整窗口已到达，才允许封存。变化检查必须先执行。
            for lease in self.leases:
                self._seal(lease)
            self.condition.notify_all()

    def start(self, *, lease_id: str, now: float, timing: PhaseTiming,
              announced_deadline: float | None = None, safety_probe: bool = False,
              needs_completion: bool = False, expected_phase_id=None, expected_context=None):
        """deadline 仅支持明确公告的实验计划；未知线上边界不用未来信息。"""
        with self.lock:
            if any(lease.lease_id == lease_id for lease in self.leases):
                raise ValueError('Duplicate lease ID')
            if not math.isfinite(now) or not self.usable or self.last_end is None:
                raise PhaseAdmissionError('No current stable monitored phase')
            if expected_phase_id is not None and expected_phase_id != self.phase_id:
                raise PhaseAdmissionError('Stale phase ID')
            if expected_context is not None and list(expected_context) != self.context:
                raise PhaseAdmissionError('Stale workload context')
            if now < self.last_end or now - self.last_end >= self.window_seconds:
                raise PhaseAdmissionError('Monitor stale or request predates current observation')
            if not math.isclose(timing.window_seconds, self.window_seconds):
                raise ValueError('Timing and monitor windows differ')
            if announced_deadline is not None:
                if not math.isfinite(announced_deadline) or announced_deadline <= now:
                    raise PhaseAdmissionError('Invalid or expired announced phase deadline')
                if not safety_probe and now + timing.trial_reserve > announced_deadline:
                    raise PhaseAdmissionError('Insufficient announced phase time including seal and recovery reserves')
            if safety_probe and announced_deadline is None:
                raise ValueError('An adversarial probe must be explicitly bounded and labelled')
            if not safety_probe and self.admitted_per_phase.get(self.phase_id, 0) >= timing.trials_per_phase:
                raise PhaseAdmissionError('Per-phase trial slots exhausted; rejected executions are not refunded')
            begin = now + timing.apply_seconds + timing.warmup_seconds
            lease = Lease(lease_id, self.phase_id, self.epoch, now, begin,
                          begin + timing.measurement_seconds, list(self.context),
                          needs_completion=needs_completion, execution_finished=not needs_completion,
                          announced_deadline=announced_deadline,
                          post_execution_reserve=timing.window_seconds + timing.recovery_seconds,
                          safety_probe=safety_probe)
            self.leases.append(lease)
            if not safety_probe:
                self.admitted_per_phase[self.phase_id] = self.admitted_per_phase.get(self.phase_id, 0) + 1
            return lease

    def complete(self, lease, actual_end):
        with self.lock:
            if (lease not in self.leases or not lease.needs_completion or lease.execution_finished
                    or not math.isfinite(actual_end) or actual_end < lease.start):
                raise ValueError('Invalid completion for guarded measurement')
            lease.measurement_end = max(lease.measurement_end, actual_end)
            lease.execution_finished = True
            if (lease.announced_deadline is not None
                    and actual_end + lease.post_execution_reserve > lease.announced_deadline):
                lease.status, lease.reason = 'rejected', 'announced_deadline_overrun'
            self._seal(lease)
            self.condition.notify_all()

    def await_seal(self, lease, timeout):
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError('Positive finite seal timeout required')
        with self.condition:
            ready = self.condition.wait_for(lambda: lease.status != 'provisional', timeout=timeout)
            if not ready:
                lease.status, lease.reason = 'rejected', 'monitor_seal_timeout'
            return self.snapshot(lease)

    def close(self):
        self.invalidate('monitor_ended_before_seal')

    def snapshot(self, lease: Lease):
        with self.lock:
            return asdict(lease)


class PhaseGuardedExecutor:
    """在独立调优线程使用；完整覆盖窗口封存前不得把测量交给 Controller。"""
    def __init__(self, executor, fence, timing, *, clock, phase_id, workload_protocol, deadline=None, seal_timeout=15):
        if not workload_protocol or getattr(executor, 'measurement_protocol', None) != workload_protocol:
            raise ValueError('Measurement protocol differs from monitored offered workload; never inject Sysbench into single-SQL replay')
        if timing.window_seconds != fence.window_seconds:
            raise ValueError('Fence and timing windows differ')
        if timing.trial_reserve < executor.estimated_trial_seconds:
            raise ValueError('Phase timing must cover the executor estimate, including restart or other slow operations')
        if not math.isfinite(seal_timeout) or seal_timeout <= 0:
            raise ValueError('Positive seal timeout required')
        self.executor, self.fence, self.timing = executor, fence, timing
        self.clock, self.phase_id, self.deadline, self.seal_timeout = clock, phase_id, deadline, seal_timeout
        self.estimated_trial_seconds = max(executor.estimated_trial_seconds, timing.trial_reserve)
        self.counter = 0
        self.last_admitted = False

    def __getattr__(self, name):
        return getattr(self.executor, name)

    def restore(self, config):
        if self.last_admitted:
            return self.executor.restore(config)
        if self.executor.readback() != config:
            from .execution import RecoveryFailed
            raise RecoveryFailed('Admission refused but underlying configuration unexpectedly changed')

    def measure(self, config, context):
        from .execution import Measurement
        self.counter += 1
        self.last_admitted = False
        try:
            lease = self.fence.start(lease_id=f'guarded-{self.counter}', now=self.clock(), timing=self.timing,
                                     announced_deadline=self.deadline() if self.deadline else None,
                                     needs_completion=True, expected_phase_id=self.phase_id(),
                                     expected_context=context)
        except PhaseAdmissionError as exc:
            return Measurement(None, 'phase_admission_refused', False, False, False, {'reason': str(exc)})
        try:
            self.last_admitted = True
            measured = self.executor.measure(config, context)
        except BaseException:
            self.fence.invalidate('executor_failed_before_seal')
            raise
        actual_end = self.clock()
        self.fence.complete(lease, actual_end)
        state = self.fence.await_seal(lease, self.seal_timeout)
        if state['status'] != 'sealed':
            return Measurement(None, 'phase_invalidated', False, measured.effective, False,
                               {**measured.metadata, 'phase_guard': state})
        return Measurement(measured.value, measured.status, measured.complete, measured.effective,
                           measured.constraints_ok, {**measured.metadata, 'phase_guard': state})


class GuardAcceptance:
    """零调参验收：75 秒真实请求观测、尾部预算拒绝和故意跨阶段探针。"""
    def __init__(self, plan):
        if plan.scenario not in {'composition', 'intensity'}:
            raise ValueError('Acceptance needs a sustained workload switch')
        self.timing = PhaseTiming(window_seconds=plan.window_seconds, warmup_seconds=15,
                                  measurement_seconds=60, apply_seconds=0, recovery_seconds=0,
                                  proposal_seconds_per_phase=0, trials_per_phase=1)
        self.timing.validate_phase(plan.phase_seconds)
        self.plan = plan
        self.fence = PhaseFence(window_seconds=plan.window_seconds)
        self.started = False
        self.probed = False
        self.refusals = []

    def observe(self, window, detection):
        self.fence.observe(window, detection)
        if not self.started and detection.event == 'phase_ready':
            self.fence.start(lease_id='same_phase_75s', now=window.end, timing=self.timing,
                             announced_deadline=self.plan.phase_seconds)
            self.started = True
        # 窗口对齐的公告边界前一窗：正常试验必须拒绝，故意探针只测试安全性。
        if not self.probed and window.end == self.plan.phase_seconds - self.plan.window_seconds:
            try:
                self.fence.start(lease_id='tail_trial_must_not_start', now=window.end, timing=self.timing,
                                 announced_deadline=self.plan.phase_seconds)
            except PhaseAdmissionError as exc:
                self.refusals.append({'at': window.end, 'reason': str(exc)})
            else:
                raise RuntimeError('Tail admission unexpectedly passed')
            probe = PhaseTiming(window_seconds=self.plan.window_seconds, warmup_seconds=0,
                                measurement_seconds=2 * self.plan.window_seconds, apply_seconds=0,
                                recovery_seconds=0, proposal_seconds_per_phase=0, trials_per_phase=1)
            self.fence.start(lease_id='forced_cross_phase_probe', now=window.end, timing=probe,
                             announced_deadline=self.plan.phase_seconds, safety_probe=True)
            self.probed = True

    def finish(self, output: Path, replay_complete: bool):
        import json
        import numpy as np
        from .replay_report import read_jsonl, write_csv
        self.fence.close()
        offered = read_jsonl(output / 'offered.jsonl')
        completed = {r['seq']: r for r in read_jsonl(output / 'requests.jsonl')}
        rows = []
        switches = [self.plan.phase_seconds, 2 * self.plan.phase_seconds]
        for lease in self.fence.leases:
            row = self.fence.snapshot(lease)
            cohort = [r for r in offered if lease.measurement_start <= r['scheduled'] < lease.measurement_end]
            done = [completed.get(r['seq']) for r in cohort]
            good = bool(cohort) and all(r['submitted'] for r in cohort) and all(r and r['ok'] and r.get('rows') == 1 for r in done)
            # 公告边界只作此验收的明确预算与事后审计，不输入检测特征。
            crosses = any(lease.start <= boundary < lease.measurement_end for boundary in switches)
            sealed = row['sealed_at_window_end']
            drained = good and sealed is not None and all(r['completed'] <= sealed for r in done)
            accepted = row['status'] == 'sealed' and replay_complete and drained and not crosses
            latency = [1000 * (r['completed'] - r['emitted']) for r in done if r and r['ok']]
            rows.append({**row, 'request_count': len(cohort), 'all_requests_complete': good,
                         'completed_before_seal': drained, 'crosses_announced_boundary': crosses,
                         'accepted_observation': accepted, 'value': float(np.percentile(latency, 95)) if accepted else None,
                         'objective': 'p95_latency_ms:minimize',
                         'purpose': 'phase_fence_acceptance_only_not_tuning', 'gp_label_written': False})
        by_id = {row['lease_id']: row for row in rows}
        passed = (by_id.get('same_phase_75s', {}).get('accepted_observation') is True
                  and by_id.get('forced_cross_phase_probe', {}).get('accepted_observation') is False
                  and by_id.get('forced_cross_phase_probe', {}).get('status') == 'rejected'
                  and len(self.refusals) == 1)
        result = {'passed': passed, 'timing': self.timing.report(), 'observations': rows,
                  'admission_refusals': self.refusals, 'gp_labels_written': 0,
                  'warning': '真实请求 P95 观测验收，不是 Sysbench TPS、参数优化或正式模型标签。'}
        (output / 'phase_guard.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
        flat = [{k: v for k, v in row.items() if k != 'context'} for row in rows]
        write_csv(output / 'phase_guard.csv', flat, list(flat[0]) if flat else ['lease_id'])
        return result


def main():
    parser = argparse.ArgumentParser(description='离线规划阶段预算，不连接数据库或模型')
    parser.add_argument('--benchmark', type=Path, required=True)
    parser.add_argument('--phase-seconds', type=int)
    parser.add_argument('--trials-per-phase', type=int, default=5)
    parser.add_argument('--apply-seconds', type=int, default=15)
    parser.add_argument('--recovery-seconds', type=int, default=15)
    parser.add_argument('--proposal-seconds', type=int, default=30)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    try:
        workload = yaml.safe_load(args.benchmark.read_text())['workload']
        timing = PhaseTiming(warmup_seconds=workload['warmup_time'], measurement_seconds=workload['duration'],
                             trials_per_phase=args.trials_per_phase, apply_seconds=args.apply_seconds,
                             recovery_seconds=args.recovery_seconds, proposal_seconds_per_phase=args.proposal_seconds)
        if args.phase_seconds is not None:
            timing.validate_phase(args.phase_seconds)
    except (ValueError, OSError, KeyError, TypeError, yaml.YAMLError) as exc:
        parser.error(str(exc))
    import json
    result = {**timing.report(), 'announced_phase_seconds': args.phase_seconds,
              'benchmark_path': str(args.benchmark.resolve())}
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x') as handle:
            handle.write(text + '\n')
    print(text)


if __name__ == '__main__':
    main()
