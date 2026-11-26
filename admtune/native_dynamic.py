"""原方法搜索机制的动态适配：共享变化触发、连续负载，非ADMTune消融替身。"""
from .baseline_strategies import AgenticDBStrategy, AgentTuneStrategy, StrategyStopped, context_record
from .budget import BudgetExhausted
from .controller import Controller, Policy
from .formal_baselines import BaselineLLM
from .security import redact

GOAL = ("minimize measured end-to-end P95 latency in milliseconds under a fixed offered request stream. "
        "Internal objective_value / score = 1 / P95(ms), higher is better; it is NOT TPS or TPS/P95.")


def native_record(trial):
    r = trial.to_dict()
    r["value"] = 1/trial.value if trial.trainable and trial.constraints_ok else None
    r["metadata"] = {**r["metadata"], "p95_latency_ms": trial.value, "raw_tps": None,
                     "internal_score_metric": "inverse_p95"}
    return r


class PhaseBudgetLLM(BaselineLLM):
    def __init__(self, *args, available, **kwargs):
        super().__init__(*args, **kwargs)
        self.available = available

    def complete(self, *args, **kwargs):
        seconds = self.available()
        if seconds < 1:
            raise BudgetExhausted("Phase API time exhausted, final measurement retained")
        original = self.client.timeout_s
        self.client.timeout_s = max(1, min(original, int(seconds)))
        try:
            return super().complete(*args, **kwargs)
        finally:
            self.client.timeout_s = original


class NativeDynamicController(Controller):
    def __init__(self, *, method, llm, workload, **kwargs):
        kwargs["policy"] = Policy(surrogate_gate=False, min_gain=0 if method == "AgentTune" else .01,
                                  final_repeats=1)
        super().__init__(**kwargs)
        self.method, self.native_llm, self.workload = method, llm, workload
        self.strategy = None

    def begin_phase(self, phase_id, context):
        prior_rounds = list(self.strategy.rounds) if self.method == "AgenticDB" and self.strategy else []
        self.phase_id, self.last_context, self.active = phase_id, list(context), False
        self.baseline = self.best_value = None
        self.best_config = dict(self.current)
        r = self.trial(self.current, context, kind="baseline")
        if not (r.trainable and r.constraints_ok):
            self.store.event("stage_stopped", {"reason": "Invalid native baseline", "phase_id": phase_id})
            return
        self.baseline = self.best_value = r.value
        # 输入仅取当前观察到的SQL特征和已执行记录，不提供未来A/B标签。
        workload = {**self.workload, "observed": self.diagnostics(), "objective": GOAL}
        cls = AgenticDBStrategy if self.method == "AgenticDB" else AgentTuneStrategy
        self.strategy = cls(self.space, self.native_llm, workload, self.store.event, objective_text=GOAL)
        transformed = native_record(r)
        if self.method == "AgenticDB":
            self.strategy.rounds = prior_rounds
            self.strategy.current_workload_phase = phase_id
            self.strategy.physical_execution = ("Continuous external stream; last accepted config remains active "
                "between trials. Rejected trials restore the incumbent, not the original startup config.")
            self.strategy.best_record = transformed
            self.strategy.earlier_phase_measurements = [
                context_record(native_record(t)) for t in self.store.trials()
                if t.phase_id != phase_id and t.trainable and t.constraints_ok][-5:]
        else:
            self.strategy.initial_record = transformed
        self.active = True
        self.store.event("native_phase_restart", {"method": self.method, "phase_id": phase_id,
            "search_state_reinitialized": True, "measured_history_retained": True,
            "shared_change_trigger": True, "gp_used": False, "reflexion_used": False})

    def fit(self, context):
        self.gp.fitted = False

    def reflect(self, events, *, force=False):
        if force:
            self.store.event("native_no_added_reflexion", {"phase_id": self.phase_id, "method": self.method})

    def step(self):
        if not self.active or self.detector.state != "stable" or self.detector.changed_count:
            return None
        try:
            self.ledger.ensure_time(self.executor.estimated_trial_seconds)
            records = [native_record(t) for t in self.store.trials()]
            config, kind, decision = self.strategy.propose(records, self.current)
        except (BudgetExhausted, StrategyStopped) as exc:
            self.active = False
            self.store.event("native_search_stopped", {"phase_id": self.phase_id, "reason": str(exc)})
            return None
        except (ValueError, KeyError, TypeError) as exc:
            self.active = False
            self.store.event("native_proposal_invalid", {"phase_id": self.phase_id, "reason": redact(str(exc))})
            return None
        self.store.event("native_selection", {"phase_id": self.phase_id, "method": self.method,
                                             "decision": decision, "config": config})
        r = self.trial(config, self.last_context, kind=kind)
        if r.metadata.get("accepted"):
            self.best_value, self.best_config = r.value, dict(r.config)
        self.strategy.observe(native_record(r))
        if not (r.trainable and r.constraints_ok):
            self.active = False
        return r
