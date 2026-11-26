from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
import json
import uuid

import numpy as np

from .agents import Team
from .budget import BudgetExhausted, Ledger
from .contracts import Environment, Objective, Trial, fingerprint
from .execution import Executor, Measurement, RecoveryFailed
from .memory import Store
from .reflection import Reflector
from .space import Space
from .surrogate import GaussianProcess, Prediction
from .workload import PhaseDetector, Window


@dataclass(frozen=True)
class Policy:
    beta: float = 2.0
    tolerated_drop: float = 0.1
    min_gain: float = 0.01
    bootstrap: int = 4
    max_batches: int = 3
    max_replans: int = 1
    max_training: int = 64
    history_distance: float = 4.0
    surrogate_gate: bool = True
    cross_phase_memory: bool = True
    kernel: str = "joint"
    residual_threshold: float = 3
    final_repeats: int = 3
    fixed_noise_log_std: float = 0.0

    def __post_init__(self):
        if (
            self.beta <= 0 or not 0 <= self.tolerated_drop < 1 or self.min_gain < 0
            or self.bootstrap < 2 or min(self.max_batches, self.max_training, self.final_repeats) < 1
            or self.max_replans < 0 or self.history_distance <= 0
            or not math.isfinite(self.fixed_noise_log_std) or self.fixed_noise_log_std < 0
        ):
            raise ValueError("Invalid tuning policy")


class Controller:
    def __init__(
        self, *, space: Space, environment: Environment, objective: Objective,
        detector: PhaseDetector, executor: Executor, team: Team, reflector: Reflector,
        store: Store, ledger: Ledger, policy: Policy, seed: int = 0,
        snapshot_dir: Path | None = None,
    ):
        self.space, self.environment, self.objective = space, environment, objective
        self.seed = seed
        self.detector, self.executor, self.team = detector, executor, team
        self.reflector, self.store, self.ledger, self.policy = reflector, store, ledger, policy
        self.current = space.merge(executor.readback(), {})
        self.best_config = dict(self.current)
        self.baseline = self.best_value = None
        self.phase_id = None
        self.last_context: list[float] = []
        self.active = False
        self.seen: set[tuple[str, str]] = set()
        self.replans = self.no_gain = 0
        self.snapshot_dir = snapshot_dir
        self.gp = GaussianProcess(
            kernel=policy.kernel, config_dimensions=len(space.encode(self.current)), seed=seed,
        )

    def histories(self, context: list[float]) -> list[Trial]:
        return self.store.compatible(
            self.environment.key, self.objective.key, context,
            phase_id=None if self.policy.cross_phase_memory else self.phase_id,
            max_distance=self.policy.history_distance, limit=self.policy.max_training,
        )

    def experiences(self, context: list[float]) -> list[dict]:
        numerical = [
            {"type": "measurement", "trial_id": t.trial_id, "config": t.config, "value": t.value, "context": t.context}
            for t in self.histories(context)[:3]
        ]
        compatible_ids = {
            t.trial_id for t in self.store.trials()
            if t.environment_key == self.environment.key and t.objective_key == self.objective.key
            and len(t.context) == len(context)
            and (self.policy.cross_phase_memory or t.phase_id == self.phase_id)
            and float(np.sqrt(np.mean((np.asarray(t.context) - context)**2))) <= self.policy.history_distance
        }
        summaries = [
            {"type": "reflection", **summary}
            for summary in self.store.reflections()
            if set(summary["evidence_ids"]) <= compatible_ids
        ]
        return numerical + summaries[-2:]

    def reflect(self, events: list[str], *, force: bool = False):
        try:
            self.reflector.trigger(
                events, force=force,
                phase_id=None if self.policy.cross_phase_memory else self.phase_id,
            )
        except (ValueError, BudgetExhausted, RuntimeError) as exc:
            self.store.event("reflection_deferred", {"reason": str(exc)})

    def diagnostics(self) -> dict:
        diagnose = getattr(self.executor, "diagnose", None)
        if diagnose is None:
            return {}
        try:
            return diagnose()
        except Exception as exc:
            self.store.event("diagnostics_unavailable", {"reason": str(exc)})
            return {"available": False}

    def observe(self, window: Window):
        state = self.detector.observe(window)
        self.last_context = state.context
        self.store.event("workload", {**asdict(state), "window_end": window.end})
        if state.event == "change_confirmed":
            self.reflect(["phase_end"], force=True)
            self.active = False
            self.baseline = None
            self.gp.fitted = False
        if state.event != "phase_ready":
            return state
        self.begin_phase(state.phase_id, state.context)
        return state

    def begin_phase(self, phase_id: str, context: list[float]):
        """Initialize a measured phase; static mode explicitly supplies constant context."""
        self.phase_id = phase_id
        self.last_context = list(context)
        self.active = False
        self.baseline = self.best_value = None
        self.best_config = dict(self.current)
        self.gp.fitted = False
        self.replans = self.no_gain = 0
        baseline = self.trial(self.current, context, kind="baseline")
        if not baseline.trainable or not baseline.constraints_ok:
            self.active = False
            self.store.event("stage_stopped", {"reason": "Baseline invalid or violates constraints"})
            return
        self.baseline = self.best_value = baseline.value
        self.best_config = dict(self.current)
        self.team.initialize(context, self.experiences(context), diagnostics=self.diagnostics())
        self.active = True
        self.gp.fitted = False

    def trial(self, config: dict, context: list[float], *, kind: str, prediction: Prediction | None = None) -> Trial:
        self.ledger.charge_trial(kind, self.executor.estimated_trial_seconds)
        trial_id = uuid.uuid4().hex
        prediction_dict = asdict(prediction) if prediction else None
        # Persist before any execution or model update (also for nonpredicted startup).
        self.store.event("trial_submitted", {
            "trial_id": trial_id, "phase_id": self.phase_id, "kind": kind,
            "config": config, "context": context, "prediction": prediction_dict,
        })
        before = dict(self.current)
        try:
            measured = self.executor.measure(config, context)
        except KeyboardInterrupt:
            self.store.append(Trial(
                trial_id, self.phase_id, self.environment.key, self.objective.key,
                config, context, "interrupted", prediction=prediction_dict,
                metadata={"kind": kind, "simulated": self.executor.simulated},
            ))
            self.active = False
            self.executor.restore(before)
            raise
        except RecoveryFailed as exc:
            record = Trial(
                trial_id, self.phase_id, self.environment.key, self.objective.key,
                config, context, "recovery_failed", prediction=prediction_dict,
                metadata={"kind": kind, "reason": str(exc), "simulated": self.executor.simulated},
            )
            self.store.append(record)
            self.active = False
            raise
        except Exception as exc:
            measured = Measurement(None, "execution_failed", False, False, False, {"reason": str(exc)})
        if measured.value is not None and (
            isinstance(measured.value, bool) or not isinstance(measured.value, (float, int))
            or not math.isfinite(measured.value) or measured.value <= 0
        ):
            measured = Measurement(None, "invalid_measurement", False, measured.effective, False, {
                **measured.metadata, "invalid_value": repr(measured.value),
            })
        record = Trial(
            trial_id, self.phase_id, self.environment.key, self.objective.key, config,
            list(context), measured.status, measured.value, measured.complete,
            measured.effective, measured.constraints_ok, prediction_dict,
            {**measured.metadata, "kind": kind, "simulated": self.executor.simulated,
             "parent_config": before},
        )
        accepted = (
            record.trainable and record.constraints_ok
            and (kind == "baseline" or self.objective.utility(record.value, self.best_value) > 1 + self.policy.min_gain)
        )
        record.metadata["accepted"] = accepted and kind != "verification"
        self.store.append(record)
        self.seen.add((self.phase_id, fingerprint(config)))
        if accepted and kind != "verification":
            self.current = dict(config)
            self.store.event("config_accepted", {"trial_id": trial_id, "phase_id": self.phase_id,
                                                "config": config, "value": record.value})
        else:
            try:
                self.executor.restore(before)
                self.store.event("config_restored", {"trial_id": trial_id, "phase_id": self.phase_id,
                                                    "target_config": before})
            except Exception as exc:
                self.active = False
                self.store.event("recovery_failed", {"trial_id": trial_id, "reason": str(exc)})
                raise RecoveryFailed(str(exc)) from exc
        return record

    def fit(self, context: list[float]):
        if not self.policy.surrogate_gate:
            self.gp.fitted = False
            return
        history = self.histories(context)
        if len(history) < self.policy.bootstrap:
            self.gp.fitted = False
            return
        x = [self.space.encode(t.config) + t.context for t in history]
        y = [self.objective.target(t.value) for t in history]
        self.gp.fit(x, y, optimize=self.gp.version % 3 == 0 or self.policy.fixed_noise_log_std > 0,
                    fixed_noise_log_std=self.policy.fixed_noise_log_std)
        if self.snapshot_dir is not None:
            self.snapshot_dir.mkdir(parents=True, exist_ok=True)
            payload = {**self.gp.snapshot([t.trial_id for t in history]),
                       "phase_id": self.phase_id, "objective": self.objective.key}
            path = self.snapshot_dir / f"{self.phase_id}-gp-{self.gp.version:04d}.json"
            path.write_text(json.dumps(payload, indent=2))
            self.store.event("gp_fit", {"path": path.name, "version": self.gp.version,
                                       "training_ids": payload["training_ids"]})

    def step(self) -> Trial | None:
        if not self.active or self.detector.state != "stable" or self.detector.changed_count:
            return None
        if self.ledger.trials >= self.ledger.limits.trials - self.policy.final_repeats:
            self.active = False
            return None
        self.ledger.ensure_time()
        z = self.last_context
        self.team.diagnostics = self.diagnostics()
        self.fit(z)
        for _ in range(self.policy.max_batches):
            candidates = self.team.recommend(self.best_config, self.phase_id, z, self.experiences(z))
            candidates = [c for c in candidates if (self.phase_id, c.key) not in self.seen]
            if not candidates:
                self.team.feedback = {"reason": "Candidates were invalid, duplicate, or unchanged; resample."}
                continue
            predictions = None
            # Startup neighbors are permitted only in the dedicated test executor.
            if self.policy.surrogate_gate and self.gp.fitted:
                predictions = self.gp.predict([self.space.encode(c.config) + z for c in candidates])
                floor = self.objective.target(self.baseline) + math.log1p(-self.policy.tolerated_drop)
                gain_floor = self.objective.target(self.best_value) + math.log1p(self.policy.min_gain)
                eligible = [
                    i for i, p in enumerate(predictions)
                    if p.lower(self.policy.beta) >= floor and p.upper(self.policy.beta) > gain_floor
                ]
                self.store.event("prescreen", {
                    "phase_id": self.phase_id, "config_ids": [c.key for c in candidates],
                    "predictions": [asdict(p) for p in predictions], "eligible": eligible,
                    "configs": [c.config for c in candidates], "context": z,
                    "log_admission_floor": floor, "log_improvement_floor": gain_floor,
                    "best_config": self.best_config, "baseline_value": self.baseline,
                    "best_value": self.best_value, "beta": self.policy.beta,
                })
                if not eligible:
                    self.team.feedback = {
                        "reason": "No candidate met LCB admission and UCB improvement conditions.",
                        "log_admission_floor": floor, "log_improvement_floor": gain_floor,
                        "evaluations": [
                            {"config": c.config, **asdict(p)}
                            for c, p in zip(candidates[:4], predictions[:4])
                        ],
                    }
                    continue
                index = max(eligible, key=lambda i: predictions[i].upper(self.policy.beta))
            else:
                index = 0  # Explicit non-GP ablation, not a disguised AgentTune baseline.
            prediction = predictions[index] if predictions else None
            kind = "startup" if self.policy.surrogate_gate and not self.gp.fitted else "candidate"
            record = self.trial(candidates[index].config, z, kind=kind, prediction=prediction)
            self.team.feedback = {
                "reason": "Measured feedback, not a causal explanation.",
                "trial_id": record.trial_id, "status": record.status, "value": record.value,
                "accepted": record.metadata.get("accepted", False),
            }
            events = []
            if record.trainable:
                if prediction:
                    residual = (self.objective.target(record.value) - prediction.mean) / max(prediction.observation_std, 1e-9)
                    self.store.event("prediction_error", {
                        "trial_id": record.trial_id, "residual": residual,
                        "absolute_log_error": abs(self.objective.target(record.value) - prediction.mean),
                        "covered": abs(residual) <= self.policy.beta,
                    })
                    if abs(residual) > self.policy.residual_threshold:
                        events.append("prediction_error")
                if record.metadata["accepted"]:
                    self.best_value, self.best_config = record.value, dict(record.config)
                    self.no_gain = 0
                    events.append("new_best")
                else:
                    self.no_gain += 1
                if not record.constraints_ok or self.objective.utility(record.value, self.baseline) < 1 - self.policy.tolerated_drop:
                    self.team.relax()
                    events.append("range_fallback")
            else:
                self.team.relax()
                events.append("execution_anomaly")
            if self.no_gain >= 3:
                self.team.relax()
                events.append("range_fallback")
            # Numerical update is independent of reflection success.
            self.fit(z)
            self.reflect(events)
            return record
        self.replans += 1
        self.team.relax()
        if self.replans <= self.policy.max_replans:
            self.team.initialize(z, self.experiences(z))
        else:
            self.active = False
            self.store.event("stage_stopped", {"reason": "No admissible new candidates after bounded replanning"})
        return None

    def finalize(self) -> dict:
        verification = []
        if self.baseline is not None:
            for _ in range(self.policy.final_repeats):
                try:
                    record = self.trial(self.best_config, self.last_context, kind="verification")
                    if record.trainable and record.constraints_ok:
                        verification.append(record.value)
                except BudgetExhausted:
                    break
        self.reflect(["phase_end"], force=True)
        return {
            "simulated": self.executor.simulated, "objective": self.objective.key,
            "phase_id": self.phase_id, "best_config": self.best_config,
            "search_best_value": self.best_value, "baseline_value": self.baseline,
            "independent_verification_values": verification,
            "independent_verification_median": float(np.median(verification)) if verification else None,
            "budget": self.ledger.snapshot(),
            "warning": "Synthetic plumbing only; not database or LLM performance evidence." if self.executor.simulated else None,
        }
