from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from typing import Callable


class BudgetExhausted(RuntimeError):
    pass


@dataclass(frozen=True)
class Limits:
    trials: int = 24
    wall_seconds: float = 6000
    llm_calls: int = 20
    llm_tokens: int = 80000
    recovery_reserve_seconds: float = 60

    def __post_init__(self):
        if self.trials < 1 or self.wall_seconds <= 0 or self.llm_calls < 0 or self.llm_tokens < 0:
            raise ValueError("Invalid experiment budgets")
        if not 0 <= self.recovery_reserve_seconds < self.wall_seconds:
            raise ValueError("Recovery reserve must fit in the wall budget")


class Ledger:
    def __init__(self, limits: Limits, *, clock: Callable[[], float] = time.monotonic):
        self.limits, self.clock, self.started = limits, clock, clock()
        self.trials = self.llm_calls = self.llm_tokens = 0
        self.events: list[dict] = []

    def remaining_seconds(self) -> float:
        return max(0.0, self.limits.wall_seconds - (self.clock() - self.started))

    def ensure_time(self, estimated_seconds: float = 0):
        if not math.isfinite(estimated_seconds) or estimated_seconds < 0:
            raise ValueError("Finite nonnegative duration required")
        if self.remaining_seconds() < estimated_seconds + self.limits.recovery_reserve_seconds:
            raise BudgetExhausted("Insufficient wall time including recovery reserve")

    def charge_trial(self, kind: str, estimated_seconds: float = 0):
        self.ensure_time(estimated_seconds)
        if self.trials >= self.limits.trials:
            raise BudgetExhausted("True trial budget exhausted")
        self.trials += 1  # Failed attempts are never refunded.
        self.events.append({"kind": kind, "trial_slot": self.trials})

    def reserve_llm(self, max_tokens: int):
        self.ensure_time()
        if max_tokens < 1 or self.llm_calls >= self.limits.llm_calls:
            raise BudgetExhausted("LLM call budget exhausted")
        if self.llm_tokens + max_tokens > self.limits.llm_tokens:
            raise BudgetExhausted("LLM token reservation exceeds budget")
        self.llm_calls += 1
        self.llm_tokens += max_tokens  # Keep reservation when API usage is unavailable.

    def settle_llm(self, reserved: int, actual: int | None):
        if actual is not None:
            self.llm_tokens += max(0, actual) - reserved
        if self.llm_tokens > self.limits.llm_tokens:
            raise BudgetExhausted("Reported API usage exceeded reserved tokens; stop further calls")

    def snapshot(self) -> dict:
        return {
            "limits": asdict(self.limits), "trial_slots_used": self.trials,
            "llm_calls": self.llm_calls, "llm_tokens_or_reserved": self.llm_tokens,
            "elapsed_seconds": self.clock() - self.started,
        }
