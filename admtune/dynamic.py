"""真实动态调优试运行：固定到达、独立控制线程、同流 P95、严格预算。"""
from __future__ import annotations

import argparse
import fcntl
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
import hashlib
import os
from pathlib import Path
import socket
import threading
import time

from dotenv import load_dotenv
import yaml

from config import load_app_config
from llm_client import LLMClient
from .agents import AgenticDBBackend, HeuristicBackend, Team
from .budget import BudgetExhausted, Ledger, Limits
from .contracts import Environment, Objective, fingerprint
from .controller import Controller, Policy
from .dynamic_runtime import LiveTelemetry, DetectorView, DynamicExecutor, GenerationalRequests
from .execution import RecoveryFailed
from .live import load_space
from .memory import Store
from .phase_safety import PhaseFence, PhaseTiming
from .provenance import source_provenance
from .real_execution import AgenticDBExecutor
from .reflection import Reflector
from .replay import ReplayPlan, replay
from .replay_report import export_replay, write_csv
from .security import redact


def instance_lock(config_path):
    if not Path(config_path).is_absolute():
        raise ValueError('An absolute dedicated MySQL config path is required')
    fd = os.open(str(config_path) + '.admtune.lock',
                 os.O_CREAT | os.O_RDWR | getattr(os, 'O_NOFOLLOW', 0), 0o600)
    handle = os.fdopen(fd, 'w')
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BaseException:
        handle.close()
        raise
    return handle


class PooledTeam(Team):
    """每阶段一批候选，实际逐个测量和更新 GP；不是复用预先给定最优答案。"""
    def initialize(self, context, experiences, *, diagnostics=None):
        self.pool = None
        measured = [x for x in experiences if x.get('type') == 'measurement'][:1]
        summaries = [
            {k: x[k] for k in ['type', 'conditions', 'evidence_ids', 'hypotheses'] if k in x}
            for x in experiences if x.get('type') == 'reflection'
        ][-1:]
        super().initialize(context, measured + summaries, diagnostics=diagnostics)

    def recommend(self, parent, phase_id, context, experiences):
        if self.pool is None:
            self.pool = super().recommend(parent, phase_id, context, experiences[:1])
        return list(self.pool)


class BatchedController(Controller):
    def reflect(self, events, *, force=False):
        if force:
            return super().reflect(events, force=True)
        if events:
            self.store.event('reflection_batched', {'events': events, 'phase_id': self.phase_id})


class DynamicSession:
    def __init__(self, actuator, client, space, environment, output, plan, timing,
                 *, backend_kind, seed, llm_calls=12, llm_tokens=20000):
        self.actuator, self.client, self.space, self.environment = actuator, client, space, environment
        self.output, self.plan, self.timing = output, plan, timing
        self.backend_kind, self.seed = backend_kind, seed
        self.call_limit, self.token_limit = llm_calls, llm_tokens
        self.telemetry = LiveTelemetry(PhaseFence(window_seconds=plan.window_seconds))
        self.thread = None
        self.error = None
        self.result = None
        self.control = None
        self.last_progress_at = -1.0

    def on_start(self, origin):
        self.telemetry.origin = origin
        self.thread = threading.Thread(target=self._worker_entry, name='admtune-control', daemon=False)
        self.thread.start()

    def on_offered(self, record):
        self.telemetry.offer(record)

    def on_result(self, record):
        self.telemetry.result(record)

    def on_window(self, window, detection):
        self.telemetry.window(window, detection)

    def check_health(self):
        if self.error:
            raise RuntimeError('Dynamic control worker failed: ' + self.error)

    def stop(self):
        self.telemetry.stop()
        if self.thread:
            self.thread.join(timeout=150)
            if self.thread.is_alive():
                raise RecoveryFailed('Controller did not stop; do not restore concurrently with an active mutator')

    def _worker_entry(self):
        try:
            self._worker()
        except BaseException as exc:
            self.error = redact(str(exc))
            self.result = {**(self.result or {}), 'error': self.error, 'closed_loop_completed': False}
            try:
                (self.output / 'dynamic_summary.json').write_text(json.dumps(self.result, indent=2))
            except Exception:
                pass
            if getattr(self, 'store', None):
                try:
                    self.store.close()
                except Exception:
                    pass

    def _worker(self):
        store = None
        llm = None
        backend = None
        ledger = Ledger(Limits(trials=self.timing.trials_per_phase * 3,
                               wall_seconds=self.plan.duration + 180,
                               recovery_reserve_seconds=30, llm_calls=self.call_limit, llm_tokens=self.token_limit))
        phases = []
        done = set()
        verification = {}
        handled = set()
        self.ledger = ledger
        try:
            store = Store(self.output / 'records.sqlite3')
            self.store = store
            if self.backend_kind == 'llm':
                llm = LLMClient.from_env(dry_run=False)
                llm.allow_fallback = False
                backend = AgenticDBBackend(llm, ledger, max_output_tokens=1024,
                                           trace_dir=self.output / 'llm_calls')
            else:
                backend = HeuristicBackend(self.seed)
            executor = DynamicExecutor(self.actuator, self.client, self.telemetry, self.timing, self.plan,
                                       phase_id=lambda: self.control.phase_id if self.control else None)
            controller = BatchedController(
                space=self.space, environment=self.environment, objective=Objective('p95_latency_ms', 'minimize'),
                detector=DetectorView(self.telemetry), executor=executor,
                team=PooledTeam(self.space, backend, batch_size=6, max_selected=4),
                reflector=Reflector(store, backend, max_batch=12, old_summary_limit=0), store=store,
                ledger=ledger, policy=Policy(max_batches=1, max_replans=0, tolerated_drop=.2,
                                            final_repeats=1), seed=self.seed,
                snapshot_dir=self.output / 'gp_models',
            )
            self.control = controller
            current_ready = None
            while not self.telemetry.stopped:
                self.check_health()
                with self.telemetry.condition:
                    pending = [(end, d) for end, d in self.telemetry.ready if d.phase_id not in handled]
                    latest = self.telemetry.latest
                if pending:
                    ready_end, detection = pending[-1]
                    # 跳过已经过期的阶段；只调优当下实际稳定的阶段。
                    handled.update(d.phase_id for _, d in pending)
                    if not self.telemetry.fence.usable or latest.phase_id != detection.phase_id:
                        continue
                    controller.active = False
                    current_ready = ready_end
                    store.event('phase_start', {'phase_id': detection.phase_id, 'ready_at': ready_end,
                                               'compatible_history': len(controller.histories(detection.context))})
                    controller.begin_phase(detection.phase_id, detection.context)
                    self._progress(store, ledger, phases)
                    continue
                if (controller.phase_id is not None and controller.phase_id not in done
                        and self.telemetry.fence.usable and latest and latest.phase_id == controller.phase_id):
                    phase_records = [t for t in store.trials() if t.phase_id == controller.phase_id]
                    # 每阶段最后一个槽用于独立复测，前五槽包括新阶段基准。
                    search_limit = self.timing.trials_per_phase - 1
                    if controller.active and len(phase_records) < search_limit:
                        controller.last_context = list(latest.context)
                        controller.step()
                    else:
                        controller.active = False
                        values = []
                        if controller.baseline is not None and len(phase_records) < self.timing.trials_per_phase:
                            controller.last_context = list(latest.context)
                            controller.fit(controller.last_context)
                            prediction = controller.gp.predict([
                                self.space.encode(controller.best_config) + controller.last_context
                            ])[0] if controller.gp.fitted else None
                            record = controller.trial(controller.best_config, controller.last_context,
                                                      kind='verification', prediction=prediction)
                            if record.trainable and record.constraints_ok:
                                values.append(record.value)
                            if record.trainable and prediction:
                                residual = (controller.objective.target(record.value) - prediction.mean) / max(prediction.observation_std, 1e-9)
                                store.event('verification_prediction_error', {
                                    'trial_id': record.trial_id, 'residual': residual, 'covered': abs(residual) <= 2,
                                    'scope': 'repeat_of_selected_config_not_independent_heldout',
                                })
                            controller.fit(controller.last_context)
                        verification[controller.phase_id] = values
                        reflection_count = len(store.reflections())
                        controller.reflect(['phase_end'], force=True)
                        if len(store.reflections()) != reflection_count + 1:
                            raise RuntimeError('Phase reflection did not commit; inspect reflection_deferred, no automatic paid retry')
                        valid_candidates = sum(
                            t.trainable and t.constraints_ok and t.metadata.get('kind') in {'startup', 'candidate'}
                            for t in store.trials() if t.phase_id == controller.phase_id
                        )
                        phases.append({'phase_id': controller.phase_id, 'ready_at': current_ready,
                                       'finished_at': self.telemetry.elapsed(), 'baseline_value': controller.baseline,
                                       'search_best_value': controller.best_value, 'best_config': dict(controller.best_config),
                                       'verification_values': values, 'valid_candidates': valid_candidates})
                        done.add(controller.phase_id)
                        store.event('phase_finished', phases[-1])
                        self._progress(store, ledger, phases, force=True)
                else:
                    with self.telemetry.condition:
                        self.telemetry.condition.wait(timeout=.1)
                self._progress(store, ledger, phases)
        except BudgetExhausted as exc:
            self.error = 'Budget exhausted: ' + redact(str(exc))
            if store:
                store.event('budget_stop', {'reason': str(exc)})
        except Exception as exc:
            self.error = redact(str(exc))
        finally:
            if store:
                records = store.trials()
                with (self.output / 'trials.jsonl').open('w') as handle:
                    for record in records:
                        handle.write(json.dumps(record.to_dict(), allow_nan=False) + '\n')
                events = [(k, json.loads(v)) for k, v in store.db.execute('SELECT kind,payload FROM events ORDER BY seq')]
                # 监测事件由控制线程写入 SQLite，避免跨线程使用连接。
                for window in self.telemetry.windows:
                    store.event('workload', window)
                self.result = {
                    'simulated': self.actuator.simulated, 'objective': 'p95_latency_ms:minimize', 'backend': self.backend_kind,
                    'mode': 'dynamic_tuning_pilot', 'llm_calls': ledger.llm_calls,
                    'phases': phases, 'budget': ledger.snapshot(), 'error': self.error,
                    'trial_statuses': {status: sum(t.status == status for t in records) for status in {t.status for t in records}},
                    'valid_observations': sum(t.trainable for t in records),
                    'prediction_observations': sum(t.trainable and t.prediction is not None for t in records),
                    'predicted_search_observations': sum(t.trainable and t.prediction is not None and t.metadata.get('kind') != 'verification' for t in records),
                    'predicted_verification_observations': sum(t.trainable and t.prediction is not None and t.metadata.get('kind') == 'verification' for t in records),
                    'gp_fit_events': sum(k == 'gp_fit' for k, _ in events),
                    'reflection_commits': sum(k == 'reflection_committed' for k, _ in events),
                    'llm_api_attempts': ledger.llm_calls,
                    'heuristic_role_invocations': len(backend.calls) if isinstance(backend, HeuristicBackend) else 0,
                    'closed_loop_completed': len(phases) == 3 and all(
                        p['baseline_value'] is not None and p['verification_values'] and p['valid_candidates'] > 0
                        for p in phases
                    ) and any(t.trainable and t.prediction for t in records) and any(k == 'prescreen' for k, _ in events)
                    and sum(k == 'reflection_committed' for k, _ in events) == 3,
                    'is_thesis_evidence': False,
                    'note': '真实动态闭环试运行；未经正式校准、快照复位或方法对比，不能声称性能优势。',
                }
                (self.output / 'dynamic_summary.json').write_text(json.dumps(self.result, ensure_ascii=False, indent=2))
                self._export_trials(records)
                store.close()
            if llm:
                usage = llm.usage_summary()
                (self.output / 'llm_usage.json').write_text(json.dumps({**usage, 'budget': ledger.snapshot()}, indent=2))
                columns = ['timestamp', 'operation', 'model', 'duration_s', 'input_tokens', 'output_tokens', 'total_tokens', 'cached_tokens']
                write_csv(self.output / 'llm_usage.csv',
                          [{k: r.get(k) for k in columns} for r in usage.get('records', [])], columns)
            if self.result is None:
                self.result = {'error': self.error or 'Worker stopped before initialization', 'closed_loop_completed': False}
                (self.output / 'dynamic_summary.json').write_text(json.dumps(self.result, indent=2))

    def _progress(self, store, ledger, phases, *, force=False):
        now = self.telemetry.elapsed()
        if not force and now - self.last_progress_at < 1:
            return
        self.last_progress_at = now
        records = store.trials()
        payload = {'elapsed': self.telemetry.elapsed(), 'phase_id': self.control.phase_id,
                   'trials': len(records), 'budget': ledger.snapshot(), 'finished_phases': len(phases),
                   'last_trial': records[-1].to_dict() if records else None}
        temp = self.output / 'progress.tmp'
        temp.write_text(json.dumps(payload, indent=2))
        temp.replace(self.output / 'progress.json')

    def _export_trials(self, records):
        rows = []
        for index, t in enumerate(records, 1):
            rows.append({'trial_index': index, 'trial_id': t.trial_id, 'phase_id': t.phase_id,
                         'kind': t.metadata.get('kind'), 'status': t.status, 'value_p95_ms': t.value,
                         'complete': t.complete, 'effective': t.effective,
                         'accepted': t.metadata.get('accepted'), 'prediction_log_target': (t.prediction or {}).get('mean'),
                         'prediction_std': (t.prediction or {}).get('observation_std'),
                         'measurement_start': t.metadata.get('measurement_start'),
                         'measurement_end': t.metadata.get('measurement_end'),
                         'request_count': t.metadata.get('request_count'),
                         'config_generation': t.metadata.get('config_generation'),
                         'phase_guard_epoch': t.metadata.get('phase_guard', {}).get('epoch'),
                         'phase_guard_status': t.metadata.get('phase_guard', {}).get('status'),
                         'parent_config_json': json.dumps(t.metadata.get('parent_config'), sort_keys=True),
                         'config_json': json.dumps(t.config, sort_keys=True)})
        write_csv(self.output / 'dynamic_trials.csv', rows, list(rows[0]) if rows else ['trial_index'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target', type=Path, required=True)
    parser.add_argument('--benchmark', type=Path, required=True)
    parser.add_argument('--knobs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--env-file', type=Path, default=Path('.env'))
    parser.add_argument('--dedicated-instance', action='store_true')
    parser.add_argument('--allow-paid-llm', action='store_true')
    parser.add_argument('--backend', choices=['heuristic', 'llm'], default='heuristic')
    parser.add_argument('--smoke', action='store_true', help='短时软件链路验收，不是正式协议')
    parser.add_argument('--phase-seconds', type=int, default=600)
    parser.add_argument('--warmup-seconds', type=int, default=15)
    parser.add_argument('--measurement-seconds', type=int, default=60)
    parser.add_argument('--llm-calls', type=int, default=12)
    parser.add_argument('--llm-tokens', type=int, default=20000)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    if not args.dedicated_instance:
        parser.error('Dedicated recoverable experimental instance required')
    if args.backend == 'llm' and not args.allow_paid_llm:
        parser.error('Explicit paid LLM authorization required')
    if not 0 <= args.llm_calls <= 12 or not 0 <= args.llm_tokens <= 20000:
        parser.error('Pilot is capped at 12 API calls and 20000 tokens')
    if not args.smoke and (args.warmup_seconds < 15 or args.measurement_seconds < 60):
        parser.error('Shorter timing requires --smoke and must not be reported as calibrated or formal')
    try:
        timing = PhaseTiming(warmup_seconds=args.warmup_seconds, measurement_seconds=args.measurement_seconds,
                             apply_seconds=1 if args.smoke else 5, recovery_seconds=1 if args.smoke else 5,
                             proposal_seconds_per_phase=0 if args.backend == 'heuristic' else 30, trials_per_phase=6)
        timing.validate_phase(args.phase_seconds)
        plan = ReplayPlan(phase_seconds=args.phase_seconds, seed=42 + args.seed)
    except ValueError as exc:
        parser.error(str(exc))
    load_dotenv(args.env_file, override=True)
    if args.backend == 'llm' and not all(os.getenv(k) for k in ['OPENAI_API_KEY', 'OPENAI_BASE_URL', 'OPENAI_MODEL']):
        parser.error('Explicit key, endpoint and model required')
    root = Path(__file__).resolve().parent.parent
    app = load_app_config(project_root=root, target_path=args.target.resolve(), benchmark_path=args.benchmark.resolve(),
                          knobs_path=args.knobs.resolve(), dry_run_override=False)
    if app.target.dbms != 'mysql' or app.target.mysql is None:
        parser.error('Initial dynamic adapter supports MySQL only')
    plan = replace(plan, tables=app.benchmark.workload.tables, table_size=app.benchmark.workload.table_size)
    space = load_space(args.knobs, memory_bytes=app.target.hardware_memory_bytes)
    extra = (yaml.safe_load(args.target.read_text()) or {}).get('admtune', {})
    if any(not extra.get(k) or str(extra[k]).startswith('REPLACE') for k in ['hardware_id', 'data_id', 'schema_id']):
        parser.error('Explicit hardware, data and schema identities required')
    if app.target.mysql.host not in {'127.0.0.1', 'localhost'}:
        parser.error('Dynamic pilot is local-only')
    try:
        lock = instance_lock(app.target.mysql.config_path)
    except (OSError, ValueError) as exc:
        parser.error('Cannot acquire dedicated instance lock: ' + str(exc))
    args.output.mkdir(parents=True, exist_ok=False)
    actuator = AgenticDBExecutor(app, space, Objective(), args.output, allow_mutations=True, local=True,
                                max_p95_ms=float(extra.get('max_p95_ms', 1000)))
    client = session = None
    result = {'complete': False, 'closed_loop_completed': False}
    try:
        preflight = actuator.preflight()
        environment = Environment('mysql', preflight['db_version'], socket.gethostname(), extra.get('data_id', 'unverified'),
                                  preflight['readiness']['live_schema_hash'], fingerprint({'plan': asdict(plan), 'timing': timing.report(),
                                  'objective': 'p95_latency_ms:minimize'}), space.key + ':dynamic-context-v1:' + fingerprint({
                                      'slices': plan.slices, 'window_seconds': plan.window_seconds, 'base_rate': plan.base_rate,
                                  }))
        client = GenerationalRequests(asdict(actuator.settings), preflight['current_config'])
        client.preflight(plan)
        manifest = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'simulated': False,
                    'is_thesis_evidence': False, 'objective': 'p95_latency_ms:minimize',
                    'request_protocol': 'scheduled_single_sql', 'backend': args.backend, 'smoke': args.smoke,
                    'environment': asdict(environment), 'seed': args.seed, 'plan': asdict(plan),
                    'phase_timing': timing.report(), 'space': space.describe(), 'preflight': preflight,
                    'constraints_p95_ms': actuator.max_p95_ms, 'actuator_benchmark_not_executed': True,
                    'data_snapshot_restored': False, **source_provenance()}
        (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
        session = DynamicSession(actuator, client, space, environment, args.output, plan, timing,
                                 backend_kind=args.backend, seed=args.seed, llm_calls=args.llm_calls, llm_tokens=args.llm_tokens)
        # 即使主线程无法及时停止控制线程，也不能提前释放实例锁。
        session.instance_lock = lock
        result = replay(plan, client, args.output, simulated=False, observer=session)
    except Exception as exc:
        result['execution_error'] = redact(str(exc))
    finally:
        worker_stopped = True
        if session:
            try:
                session.stop()
                result.update(session.result or {})
            except Exception as exc:
                worker_stopped = False
                result['controller_stop_error'] = redact(str(exc))
        if worker_stopped:
            try:
                actuator.restore_initial()
                result['initial_configuration_restored'] = actuator.initial is not None
                if client:
                    client.publish(actuator.initial.config)
                    result['configuration_mutations'] = client.generation
                    result['configuration_mutation_definition'] = 'logical_config_changes_including_recovery'
                result['final_config_sha256'] = hashlib.sha256(
                    actuator.connector.read_file(actuator.settings.config_path, sudo=True).encode()
                ).hexdigest() if actuator.initial else None
            except Exception as exc:
                result['recovery_error'] = redact(str(exc))
                result['initial_configuration_restored'] = False
            if client:
                client.close()
            actuator.close()
        else:
            result['initial_configuration_restored'] = False
        if worker_stopped:
            lock.close()
        (args.output / 'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
        if all((args.output / f).exists() for f in ['offered.jsonl', 'requests.jsonl', 'windows.jsonl']):
            try:
                export_replay(args.output)
            except Exception as exc:
                result['complete'] = False
                result['export_error'] = redact(str(exc))
                (args.output / 'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get('complete') and result.get('closed_loop_completed') and result.get('initial_configuration_restored') else 1


if __name__ == '__main__':
    raise SystemExit(main())
