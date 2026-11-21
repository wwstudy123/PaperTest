from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


@dataclass(frozen=True)
class Environment:
    dbms: str
    version: str
    hardware_id: str
    data_id: str
    schema_id: str
    protocol_id: str
    encoding_version: str = "v1"

    @property
    def key(self) -> str:
        return fingerprint(asdict(self))


@dataclass(frozen=True)
class Objective:
    name: str = "tps"
    direction: str = "maximize"

    def __post_init__(self):
        if (self.name, self.direction) not in {
            ("tps", "maximize"), ("complete_latency_s", "minimize"),
            ("p95_latency_ms", "minimize"),
        }:
            raise ValueError("Unsupported objective; objectives cannot be mixed")

    @property
    def key(self) -> str:
        return f"{self.name}:{self.direction}"

    def target(self, value: float) -> float:
        if not math.isfinite(value) or value <= 0:
            raise ValueError("Only finite positive complete measurements are GP labels")
        return math.log(value) * (1 if self.direction == "maximize" else -1)

    def utility(self, value: float, baseline: float) -> float:
        return math.exp(self.target(value) - self.target(baseline))


@dataclass(frozen=True)
class Candidate:
    config: dict[str, Any]
    phase_id: str
    parent_id: str
    evidence_ids: tuple[str, ...] = ()

    @property
    def key(self) -> str:
        return fingerprint(self.config)


@dataclass
class Trial:
    trial_id: str
    phase_id: str
    environment_key: str
    objective_key: str
    config: dict[str, Any]
    context: list[float]
    status: str
    value: float | None = None
    complete: bool = False
    effective: bool = False
    constraints_ok: bool = False
    prediction: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def trainable(self) -> bool:
        return (
            self.status == "measured" and self.complete and self.effective
            and self.value is not None and math.isfinite(self.value) and self.value > 0
        )

    def to_dict(self) -> dict:
        return asdict(self)
