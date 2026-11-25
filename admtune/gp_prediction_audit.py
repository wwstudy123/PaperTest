"""零LLM固定配置验收：先封存预测，再允许后续真实测量。默认不启用。"""
from dataclasses import asdict
from copy import deepcopy
from datetime import datetime, timezone
import math
from pathlib import Path

from .contracts import Trial
from .continuous import atomic_json
from .repeat_aware_gp import POLICY, RepeatAwareGP
from .surrogate import GaussianProcess

AUDIT_POLICY = {
    "version": "prequential_gp_noise_audit_v1",
    "legacy": "optimize_on_versions_0_10_20",
    "candidate": POLICY,
    "label_precision_decimals": 6,
    "test_sequences": [7, 8, 9],
    "past_test_observation_may_train_next_prediction": True,
    "no_configuration_search_or_llm": True,
}


class GPPredictionAudit:
    def __init__(self, space, environment_key, output):
        self.space, self.environment_key, self.output = space, environment_key, Path(output)
        self.legacy = GaussianProcess(config_dimensions=len(space.encode(space.defaults)), seed=42)
        self.candidate = RepeatAwareGP(space)
        self.legacy_ids = []

    def prepare(self, records, config, elapsed_seconds):
        if len(records) < POLICY["bootstrap"]:
            return None
        if [r["seq"] for r in records] != list(range(1, len(records)+1)):
            raise ValueError("Audit requires contiguous chronological completed records")
        seq = len(records)+1
        if seq not in AUDIT_POLICY["test_sequences"]:
            raise ValueError("Audit only admits predeclared test sequences 7, 8, 9")
        self.output.mkdir(parents=True, exist_ok=True, mode=0o700)
        path = self.output/f"before-{seq:06d}.json"
        if path.exists():
            raise ValueError("Never overwrite an already sealed prediction")
        trials = []
        for r in records:
            m = r["measurement"]
            if not r["valid"]:
                raise ValueError("Invalid past measurement cannot enter GP audit")
            tps, p95 = m["value"], m["metadata"]["p95_latency_ms"]
            if (any(isinstance(v, bool) or not isinstance(v, (int, float))
                    or not math.isfinite(v) or v <= 0 for v in (tps, p95, r["joint_value"]))
                    or r["joint_value"] != tps/p95):
                raise ValueError("Audit score must match same-measurement TPS/P95")
            trials.append(Trial(f"repeat-{r['seq']}", "repeat-audit", self.environment_key,
                POLICY["objective_key"], r["config"], [0.], m["status"],
                round(r["joint_value"], 6), m["complete"], m["effective"], m["constraints_ok"]))
        ids = [t.trial_id for t in trials]
        x = [self.space.encode(t.config)+t.context for t in trials]
        y = [math.log(t.value) for t in trials]
        if ids != self.legacy_ids:
            if ids[:len(self.legacy_ids)] != self.legacy_ids:
                raise ValueError("Legacy audit history must be append-only")
            self.legacy.fit(x, y, optimize=self.legacy.version % 10 == 0)
            self.legacy_ids = ids
        self.candidate.update(trials)
        encoded = self.space.encode(config)+[0.]
        payload = {
            "next_seq": seq, "config": deepcopy(config), "submitted_at_utc": datetime.now(timezone.utc).isoformat(),
            "submitted_elapsed_seconds": elapsed_seconds,
            "policy": deepcopy(AUDIT_POLICY), "training_sequences": [r["seq"] for r in records],
            "models": {
                "legacy": {"prediction": asdict(self.legacy.predict([encoded])[0]),
                           "snapshot": self.legacy.snapshot(ids.copy())},
                "repeat_aware": {"prediction": asdict(self.candidate.model.predict([encoded])[0]),
                                 "snapshot": self.candidate.snapshot()}},
        }
        atomic_json(path, payload)
        return payload
