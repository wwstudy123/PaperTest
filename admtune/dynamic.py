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
from .evidence_wire import EvidenceWireBackend
from .live import load_space
from .memory import Store
from .phase_safety import PhaseFence, PhaseTiming
from .provenance import source_provenance
from .real_execution import AgenticDBExecutor
from .reflection import Reflector
from .replay import ReplayPlan, replay, monitor, query_semantics
from .replay_report import export_replay, write_csv
from .security import redact
from tools.mysql_resize import wait_buffer_pool_ready


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


def completion_checks(phases, records, events, *, require_gp=True, require_reflection=True, require_search=True):
    """保守不试也是合法决策；但不能把它等同于新配置预测效果已验证。"""
    ready_phases = len(phases) == 3 and all(
        p['baseline_value'] is not None and p['verification_values'] for p in phases
    )
    refused = {
        p['phase_id'] for kind, p in events
        if kind == 'prescreen' and p.get('config_ids') and not p.get('eligible')
    }
    decisions = ready_phases and (not require_search or all(
        p['valid_candidates'] > 0 or p['phase_id'] in refused for p in phases
    ))
    has_prediction = any(t.trainable and t.prediction for t in records)
    has_screen = any(k == 'prescreen' for k, _ in events)
    reflections = not require_reflection or sum(k == 'reflection_committed' for k, _ in events) == 3
    common = decisions and (not require_gp or (has_prediction and has_screen)) and reflections
    return {
        'acceptance_contract': 'control_flow_v2_resize_stable',
        'control_flow_completed': bool(common and (not require_search or any(p['valid_candidates'] > 0 for p in phases))),
        'closed_loop_completed': bool(common and (not require_search or all(p['valid_candidates'] > 0 for p in phases))),
        'gp_selected_execution_exercised': any(
            t.trainable and t.prediction and t.metadata.get('kind') in {'startup', 'candidate'} for t in records
        ),
        'conservative_no_trial_phases': sorted(
            p['phase_id'] for p in phases if p['valid_candidates'] == 0 and p['phase_id'] in refused
        ),
    }


class PooledTeam(Team):
    """每阶段一批候选，实际逐个测量和更新 GP；不是复用预先给定最优答案。"""
    def initialize(self, context, experiences, *, diagnostics=None):
        self.pool = None
        # GP refusal thresholds belong to the preceding phase/model. Compatible
        # observations remain available through retrieval, not stale feedback.
        self.feedback = {}
        measured = [x for x in experiences if x.get('type') == 'measurement'][:1]
        summaries = [
            {
                **{k: x[k] for k in ['type', 'conditions', 'evidence_ids'] if k in x},
                'hypothesis_previews': [
                    {'text': str(h)[:160], 'truncated': len(str(h)) > 160}
                    for h in x.get('hypotheses', [])[:2]
                ],
            }
            for x in experiences if x.get('type') == 'reflection'
        ][-1:]
        super().initialize(context, measured + summaries, diagnostics=diagnostics)

    def recommend(self, parent, phase_id, context, experiences):
        if self.pool is None:
            self.pool = super().recommend(parent, phase_id, context, experiences[:1])
        return list(self.pool)


class BatchedController(Controller):
    def reflect(self, events, *, force=False):
        if not getattr(self, "reflection_enabled", True):
            if force:
                self.store.event("reflection_disabled", {"phase_id": self.phase_id})
            return
        if force:
            return super().reflect(events, force=True)
        if events:
            self.store.event('reflection_batched', {'events': events, 'phase_id': self.phase_id})

    def begin_phase(self, phase_id, context):
        if not self.policy.cross_phase_memory:
            from .surrogate import GaussianProcess
            self.gp = GaussianProcess(kernel=self.policy.kernel,
                config_dimensions=len(self.space.encode(self.current)), seed=self.seed)
        return super().begin_phase(phase_id, context)


class DynamicSession:
    def __init__(self, actuator, client, space, environment, output, plan, timing,
                 *, backend_kind, seed, llm_calls=12, llm_tokens=20000, fixed_noise_log_std=0.0,
                 max_output_tokens=1024):
        self.actuator, self.client, self.space, self.environment = actuator, client, space, environment
        self.output, self.plan, self.timing = output, plan, timing
        self.backend_kind, self.seed = backend_kind, seed
        self.call_limit, self.token_limit = llm_calls, llm_tokens
        self.fixed_noise_log_std = fixed_noise_log_std
        self.max_output_tokens = max_output_tokens
        self.method, self.ablation = "ADMTune", "full"
        self.onlinetune_source = None
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
            self.result = {**(self.result or {}), 'error': self.error,
                           'closed_loop_completed': False, 'control_flow_completed': False}
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
            if self.method == "OnlineTune":
                backend = None
            elif self.backend_kind == 'llm':
                llm = LLMClient.from_env(dry_run=False)
                llm.allow_fallback = False
                backend = AgenticDBBackend(llm, ledger, max_output_tokens=self.max_output_tokens,
                                           trace_dir=self.output / 'llm_calls')
            else:
                backend = HeuristicBackend(self.seed)
            executor = DynamicExecutor(self.actuator, self.client, self.telemetry, self.timing, self.plan,
                                       phase_id=lambda: self.control.phase_id if self.control else None)
            controller_kwargs = dict(
                space=self.space, environment=self.environment, objective=Objective('p95_latency_ms', 'minimize'),
                detector=DetectorView(self.telemetry), executor=executor,
                team=None if self.method == "OnlineTune" else PooledTeam(self.space, backend, batch_size=6, max_selected=4),
                reflector=None if self.method == "OnlineTune" else Reflector(store, EvidenceWireBackend(
                    backend, audit_dir=self.output / 'reflection_maps',
                ), max_batch=12, old_summary_limit=0), store=store,
                ledger=ledger, policy=Policy(max_batches=1, max_replans=0, tolerated_drop=.2,
                    final_repeats=1, fixed_noise_log_std=self.fixed_noise_log_std,
                    cross_phase_memory=self.ablation != "no_history",
                    surrogate_gate=self.ablation != "no_gp"), seed=self.seed,
                snapshot_dir=self.output / 'gp_models',
            )
            if self.method == "ADMTune":
                controller = BatchedController(**controller_kwargs)
                controller.reflection_enabled = self.ablation != "no_reflection"
            elif self.method == "OnlineTune":
                from .onlinetune_dynamic import OnlineTuneDynamicController
                controller = OnlineTuneDynamicController(telemetry=self.telemetry,
                    upstream=self.onlinetune_source, output=self.output,
                    table_size=self.plan.table_size, **controller_kwargs)
            else:
                from .native_dynamic import NativeDynamicController, PhaseBudgetLLM
                def available():
                    now = self.telemetry.elapsed()
                    deadline = min(self.plan.duration, (int(now//self.plan.phase_seconds)+1)*self.plan.phase_seconds)
                    return deadline-now-2*self.timing.trial_reserve
                native_llm = PhaseBudgetLLM(llm, ledger, self.output/"llm_calls", available=available,
                                           max_output_tokens=self.max_output_tokens)
                controller = NativeDynamicController(method=self.method, llm=native_llm,
                    workload={"query_semantics": query_semantics(self.plan),
                        "tables": self.plan.tables, "rows_per_table": self.plan.table_size,
                        "offered_rate_source": "observed windows", "db_version": self.environment.version},
                    **controller_kwargs)
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
                        if (self.method == "ADMTune" and self.ablation != "no_reflection"
                                and len(store.reflections()) != reflection_count + 1):
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
                    "method": self.method, "ablation": self.ablation,
                    "shared_change_trigger": self.method != "ADMTune",
                    **completion_checks(phases, records, events,
                        require_gp=self.method == "ADMTune" and self.ablation != "no_gp",
                        require_reflection=self.method == "ADMTune" and self.ablation != "no_reflection",
                        require_search=self.method == "ADMTune"),
                    'is_thesis_evidence': False,
                    'note': '真实动态闭环试运行；未经正式校准、快照复位或方法对比，不能声称性能优势。',
                }
                if self.error:
                    self.result.update(closed_loop_completed=False, control_flow_completed=False)
                if self.method == "OnlineTune" and getattr(self, "control", None):
                    strategy = self.control.strategy
                    self.result["onlinetune_dynamic"] = {
                        "regions": len(strategy.bank), "svm_fits": strategy.router.svm_fits,
                        "model_reuse_events": sum(k == "onlinetune_region_activated"
                                                 and p.get("reused_model", False) for k, p in events),
                        "active_context": strategy.context.tolist(),
                        "actual_encoder": "observed SQL structure, not pretrained LSTM",
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
            elif self.method == "OnlineTune":
                (self.output / "llm_usage.json").write_text(json.dumps({
                    "model": None, "calls": 0, "total_tokens": 0, "records": [], "budget": ledger.snapshot()}))
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
    parser.add_argument('--backend', choices=['heuristic', 'llm', 'onlinetune'], default='heuristic')
    parser.add_argument('--smoke', action='store_true', help='短时软件链路验收，不是正式协议')
    parser.add_argument('--phase-seconds', type=int, default=600)
    parser.add_argument('--warmup-seconds', type=int, default=15)
    parser.add_argument('--measurement-seconds', type=int, default=60)
    parser.add_argument('--llm-calls', type=int, default=12)
    parser.add_argument('--llm-tokens', type=int, default=20000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--query-profile', choices=['point', 'range-mixed'], default='point')
    parser.add_argument('--range-rows', type=int, default=0)
    parser.add_argument('--top-k', type=int, default=0)
    parser.add_argument('--fixed-noise-log-std', type=float, default=0.0)
    parser.add_argument('--method', choices=["ADMTune", "AgenticDB", "AgentTune", "OnlineTune"], default="ADMTune")
    parser.add_argument('--onlinetune-source', type=Path)
    parser.add_argument('--ablation', choices=["full", "single_slice", "no_history", "no_gp", "no_reflection"],
                        default="full")
    parser.add_argument('--formal', action='store_true', help='冻结协议矩阵：共同上限36次API/120000token')
    parser.add_argument('--max-output-tokens', type=int, default=1024)
    args = parser.parse_args()
    try:
        Policy(fixed_noise_log_std=args.fixed_noise_log_std)
    except ValueError as exc:
        parser.error(str(exc))
    if not args.dedicated_instance:
        parser.error('Dedicated recoverable experimental instance required')
    if args.backend == 'llm' and not args.allow_paid_llm:
        parser.error('Explicit paid LLM authorization required')
    if args.method == "OnlineTune":
        if (args.backend != "onlinetune" or args.ablation != "full"
                or args.onlinetune_source is None or not args.onlinetune_source.is_dir()):
            parser.error("OnlineTune requires its dedicated backend, pinned source and no ADMTune ablations")
        args.llm_calls = args.llm_tokens = 0
    elif args.backend == "onlinetune":
        parser.error("OnlineTune backend cannot be relabelled as another method")
    call_cap, token_cap = (36, 120000) if args.formal else (12, 20000)
    if not 0 <= args.llm_calls <= call_cap or not 0 <= args.llm_tokens <= token_cap:
        parser.error(f'Cap: {call_cap} API calls and {token_cap} tokens')
    if args.formal and (args.backend not in {"llm", "onlinetune"} or args.smoke):
        parser.error("Formal matrix requires an actual method backend and full measurement timing")
    if not 1 <= args.max_output_tokens <= (1500 if args.formal else 1024):
        parser.error("Output cap: pilot 1024, formal 1500")
    if args.method in {"AgenticDB", "AgentTune"} and (args.backend != "llm" or args.ablation != "full"):
        parser.error("Native baselines require actual LLM and cannot be relabelled ablations")
    if not args.smoke and (args.warmup_seconds < 15 or args.measurement_seconds < 60):
        parser.error('Shorter timing requires --smoke and must not be reported as calibrated or formal')
    try:
        timing = PhaseTiming(warmup_seconds=args.warmup_seconds, measurement_seconds=args.measurement_seconds,
                             apply_seconds=5, recovery_seconds=5,
                             proposal_seconds_per_phase=0 if args.backend == 'heuristic' else 30, trials_per_phase=6)
        timing.validate_phase(args.phase_seconds)
        plan = ReplayPlan(phase_seconds=args.phase_seconds, seed=42 + args.seed,
                          query_profile=args.query_profile, range_rows=args.range_rows, top_k=args.top_k,
                          read_percent=100 if args.query_profile == 'range-mixed' else 50,
                          single_slice=args.ablation == "single_slice")
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
                                      'query_profile': plan.query_profile, 'range_rows': plan.range_rows, 'top_k': plan.top_k,
                                      'monitor': monitor(plan)[2],
                                  }))
        client = GenerationalRequests(asdict(actuator.settings), preflight['current_config'])
        client.preflight(plan)
        manifest = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'simulated': False,
                    'is_thesis_evidence': False, 'objective': 'p95_latency_ms:minimize',
                    'request_protocol': 'scheduled_single_sql', 'backend': args.backend, 'smoke': args.smoke,
                    'environment': asdict(environment), 'seed': args.seed, 'plan': asdict(plan),
                    'phase_timing': timing.report(), 'space': space.describe(), 'preflight': preflight,
                    'constraints_p95_ms': actuator.max_p95_ms, 'actuator_benchmark_not_executed': True,
                    'data_snapshot_restored': False,
                    'reflection_evidence_encoding': 'reflection_evidence_ids_v1',
                    'fixed_noise_log_std': args.fixed_noise_log_std,
                    'method': args.method, 'ablation': args.ablation, 'provider_seed_controlled': False,
                    'shared_change_trigger': args.method != "ADMTune",
                    'formal_protocol': args.formal, 'llm_call_limit': args.llm_calls,
                    'llm_token_limit': args.llm_tokens,
                    'max_output_tokens': args.max_output_tokens,
                    **source_provenance()}
        if args.method == "OnlineTune":
            from .onlinetune_dynamic import DYNAMIC_ADAPTATION
            manifest["onlinetune_adapter"] = DYNAMIC_ADAPTATION
        (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
        session = DynamicSession(actuator, client, space, environment, args.output, plan, timing,
                                 backend_kind=args.backend, seed=args.seed, llm_calls=args.llm_calls,
                                 llm_tokens=args.llm_tokens, fixed_noise_log_std=args.fixed_noise_log_std)
        session.max_output_tokens = args.max_output_tokens
        session.method, session.ablation = args.method, args.ablation
        session.onlinetune_source = args.onlinetune_source
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
                wait_buffer_pool_ready(actuator.db, actuator.readback()['innodb_buffer_pool_size'], timeout=60)
                actuator.restore_initial()
                result['final_buffer_pool_ready'] = wait_buffer_pool_ready(
                    actuator.db, actuator.initial.config['innodb_buffer_pool_size'], timeout=60
                )
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
    return 0 if result.get('complete') and result.get('control_flow_completed') and result.get('initial_configuration_restored') else 1


if __name__ == '__main__':
    raise SystemExit(main())
