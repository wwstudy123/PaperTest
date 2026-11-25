"""候选GP新版本：重复证据估计训练噪声，每批新增观测重估超参数。

独立、显式调用，不接入旧在线控制器，不读取后续检验点。
"""
from collections import defaultdict
from copy import deepcopy
import math
import re
import statistics

from .contracts import fingerprint
from .surrogate import GaussianProcess

POLICY = {
    "version": "repeat_aware_gp_v1", "bootstrap": 6, "min_repeat_degrees_of_freedom": 3,
    "objective_key": "tps_per_p95:maximize",
    "optimize_on_each_history_update": True,
    "grouping": ["environment_key", "objective_key", "phase_id", "config", "context"],
    "noise_parameterization": "residual_plus_fixed_variance_v1",
    "no_automatic_repeat_measurements": True,
}


def record_fingerprint(trial):
    return fingerprint({k: getattr(trial, k) for k in (
        "trial_id", "environment_key", "objective_key", "phase_id", "config", "context",
        "value", "status", "complete", "effective", "constraints_ok")})


def repeat_noise(trials):
    if not trials or len({t.trial_id for t in trials}) != len(trials):
        raise ValueError("Nonempty unique observation IDs required")
    if len({(t.environment_key, t.objective_key) for t in trials}) != 1:
        raise ValueError("Do not mix objective or environment")
    if trials[0].objective_key != POLICY["objective_key"]:
        raise ValueError("This policy requires the composite maximization objective")
    groups = defaultdict(list)
    for t in trials:
        if (isinstance(t.value, bool) or not isinstance(t.value, (float, int))
                or not (t.trainable and t.constraints_ok)):
            raise ValueError("Noise estimator accepts only valid, constraint-compatible observations")
        if any(isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v)
               for v in t.context):
            raise ValueError("Finite context required")
        key = fingerprint({k: getattr(t, k) for k in POLICY["grouping"]})
        groups[key].append(math.log(t.value))
    repeated = [ys for ys in groups.values() if len(ys) >= 2]
    df = sum(len(ys)-1 for ys in repeated)
    sse = sum(sum((y-statistics.mean(ys))**2 for y in ys) for ys in repeated)
    estimate = math.sqrt(sse/df) if df else None
    active = df >= POLICY["min_repeat_degrees_of_freedom"]
    return {"training_ids": [t.trial_id for t in trials],
            "repeat_groups": len(repeated), "degrees_of_freedom": df, "sse": sse,
            "estimated_log_std": estimate, "fixed_noise_log_std": estimate if active else 0.0,
            "floor_eligible": active}


class RepeatAwareGP:
    def __init__(self, space, *, seed=42):
        self.space, self.seed = space, seed
        self.model = GaussianProcess(config_dimensions=len(space.encode(space.defaults)), seed=seed)
        self.training_ids, self.record_fingerprints = [], []
        self.audit = None

    def update(self, trials):
        if len(trials) < POLICY["bootstrap"]:
            raise ValueError("At least six valid observations required")
        audit = repeat_noise(trials)
        ids, hashes = [t.trial_id for t in trials], [record_fingerprint(t) for t in trials]
        x = [self.space.encode(t.config)+t.context for t in trials]
        y = [math.log(t.value) for t in trials]
        n = len(self.training_ids)
        if ids[:n] != self.training_ids or hashes[:n] != self.record_fingerprints:
            raise ValueError("Training history must be immutable and append-only")
        if n and (self.model.x.tolist() != x[:n] or self.model.y.tolist() != y[:n]):
            raise ValueError("Snapshot training arrays differ from immutable records")
        if ids == self.training_ids:
            return False
        # 不复用“每10个模型版本才优化”的旧调度；观测新增后总是有界重优化。
        candidate = GaussianProcess(config_dimensions=self.model.config_dimensions, seed=self.seed)
        candidate.theta = self.model.theta.copy()
        candidate.version = self.model.version
        candidate.fit(x, y, optimize=True, fixed_noise_log_std=audit["fixed_noise_log_std"])
        self.model = candidate
        self.training_ids, self.record_fingerprints, self.audit = ids, hashes, audit
        return True

    def snapshot(self):
        if self.audit is None:
            raise ValueError("Fit before snapshot")
        return {"policy": deepcopy(POLICY), "space_key": self.space.key, "seed": self.seed,
                "training_ids": self.training_ids.copy(),
                "record_fingerprints": self.record_fingerprints.copy(), "audit": deepcopy(self.audit),
                "gp": self.model.snapshot(self.training_ids.copy())}

    @classmethod
    def from_snapshot(cls, space, payload):
        if payload["policy"] != POLICY or payload["space_key"] != space.key:
            raise ValueError("Space or repeat-noise policy mismatch")
        ids, hashes = payload["training_ids"], payload["record_fingerprints"]
        if (len(ids) < POLICY["bootstrap"] or len(set(ids)) != len(ids)
                or len(ids) != len(hashes) or any(
                    not isinstance(h, str) or not re.fullmatch("[0-9a-f]{64}", h) for h in hashes)
                or payload["gp"]["training_ids"] != ids or payload["audit"]["training_ids"] != ids):
            raise ValueError("Snapshot training provenance mismatch")
        result = cls(space, seed=payload["seed"])
        result.model = GaussianProcess.from_snapshot(payload["gp"])
        if result.model.config_dimensions != len(space.encode(space.defaults)):
            raise ValueError("Encoded configuration dimension mismatch")
        audit = payload["audit"]
        df, sse = audit["degrees_of_freedom"], audit["sse"]
        groups = audit["repeat_groups"]
        if (isinstance(df, bool) or not isinstance(df, int) or not 0 <= df < len(ids)
                or isinstance(sse, bool) or not isinstance(sse, (int, float))
                or not math.isfinite(sse) or sse < 0
                or isinstance(groups, bool) or not isinstance(groups, int) or not 0 <= groups <= df
                or (df == 0 and (sse != 0 or groups != 0)) or (df > 0 and groups == 0)):
            raise ValueError("Invalid repeat variance audit")
        eligible = df >= POLICY["min_repeat_degrees_of_freedom"]
        estimate = math.sqrt(sse/df) if df else None
        floor = estimate if eligible else 0.0
        if (audit["floor_eligible"] != eligible or audit["estimated_log_std"] != estimate
                or audit["fixed_noise_log_std"] != floor
                or result.model.fixed_noise_log_std != floor):
            raise ValueError("Repeat variance and fitted covariance mismatch")
        result.training_ids, result.record_fingerprints, result.audit = ids.copy(), hashes.copy(), deepcopy(audit)
        return result
