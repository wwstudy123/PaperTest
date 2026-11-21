from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Protocol

import numpy as np

from .space import Space


@dataclass
class Measurement:
    value: float | None
    status: str = "measured"
    complete: bool = True
    effective: bool = True
    constraints_ok: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


class Executor(Protocol):
    estimated_trial_seconds: float
    simulated: bool

    def readback(self) -> dict: ...
    def measure(self, config: dict, context: list[float]) -> Measurement: ...
    def restore(self, config: dict) -> None: ...


class RecoveryFailed(RuntimeError):
    pass


class SimulatedExecutor:
    """Synthetic landscape for plumbing tests. Never label these as DB results."""

    estimated_trial_seconds = 0.0
    simulated = True

    def __init__(self, space: Space, seed: int = 0, *, objective: str = "tps"):
        self.space, self.current = space, space.defaults
        self.rng = np.random.default_rng(seed)
        self.objective = objective

    def readback(self) -> dict:
        return dict(self.current)

    def measure(self, config: dict, context: list[float]) -> Measurement:
        self.current = self.space.merge(config, {})
        x = np.asarray(self.space.encode(config))
        # Unknown to the proposer; context-dependent interactions allow ranking reversals.
        shift = 0.15 * math.tanh(float(np.mean(context)))
        optimum = np.linspace(0.25 + shift, 0.75 - shift, len(x))
        log_score = 4.5 - float(((x - optimum)**2).mean()) + float(self.rng.normal(0, 0.005))
        value = math.exp(log_score if self.objective == "tps" else -log_score + 8)
        return Measurement(value, metadata={"simulated": True, "noise_log_std": 0.005})

    def restore(self, config: dict):
        self.current = self.space.merge(config, {})
