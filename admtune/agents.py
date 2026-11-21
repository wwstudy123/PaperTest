from __future__ import annotations

import json
import math
import random
from typing import Protocol

from .budget import Ledger
from .contracts import Candidate, fingerprint
from .space import Space


PROMPTS = {
    "selector": (
        "Select useful DBMS parameters from the supplied fixed catalog. Runtime metrics "
        "are diagnostic, not causal evidence. Do not change durability or benchmark "
        "semantics. Return JSON: selected (list of names), rationale (text)."
    ),
    "pruner": (
        "Suggest preferred ranges for selected DBMS knobs, within their immutable hard "
        "bounds. Keep automatic/sentinel values separate and preserve granularity. "
        "Return JSON: ranges mapping each selected name to lower, upper, choices, special."
    ),
    "recommender": (
        "Propose several small DBMS configuration deltas around the measured parent. "
        "Use only selected knobs and preferred ranges. Unchanged knobs inherit parent "
        "values. Histories are observations, not guaranteed causes. Return JSON: "
        "candidates, a list of delta dictionaries; do not invent measured outcomes."
    ),
    "reflection": (
        "Summarize only new evidence, using bounded old summaries for context. Return "
        "exactly facts, hypotheses, conditions, actions, evidence_ids. Facts must be "
        "objects copied from provided fact records (trial_id/status/value). Hypotheses "
        "are unverified text. Conditions contain environment_key and objective_key. "
        "Actions are resample/adjust_range/reselect/verify/stop. Never change hard "
        "constraints or mark unexecuted candidates successful."
    ),
}


class Backend(Protocol):
    def complete(self, role: str, payload: dict) -> dict: ...


class HeuristicBackend:
    """Free deterministic fixture, explicitly NOT an LLM or a research baseline."""

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)
        self.calls: list[str] = []

    def complete(self, role: str, payload: dict) -> dict:
        self.calls.append(role)
        if role == "selector":
            return {"selected": [k["name"] for k in payload["catalog"]][:payload["max_selected"]],
                    "rationale": "Offline fixture selects fixed catalog order."}
        if role == "pruner":
            return {"ranges": {
                k["name"]: {key: k[key] for key in ["lower", "upper", "choices", "special"]}
                for k in payload["catalog"] if k["name"] in payload["selected"]
            }}
        if role == "reflection":
            return {
                "facts": payload["facts"], "hypotheses": ["Combination effects need controlled confirmation."],
                "conditions": payload["conditions"], "actions": ["resample"],
                "evidence_ids": [f["trial_id"] for f in payload["facts"]],
            }
        if role != "recommender":
            raise ValueError("Unknown role")
        result = []
        for _ in range(payload["batch_size"]):
            delta = {}
            names = self.rng.sample(payload["selected"], self.rng.randint(1, len(payload["selected"])))
            for name in names:
                spec = next(k for k in payload["catalog"] if k["name"] == name)
                preferred = payload["ranges"][name]
                if spec["kind"] in {"enum", "boolean"}:
                    delta[name] = self.rng.choice(preferred["choices"])
                elif preferred["special"] and self.rng.random() < 0.1:
                    delta[name] = self.rng.choice(preferred["special"])
                else:
                    # A mixed local/global proposal avoids a fixture that knows the optimum.
                    low, high = preferred["lower"], preferred["upper"]
                    if self.rng.random() < 0.6:
                        parent = payload["parent"][name]
                        center = parent if low <= parent <= high else (low + high) / 2
                        value = self.rng.gauss(center, (high - low) / 5)
                    else:
                        value = self.rng.uniform(low, high)
                    value = low + round((min(high, max(low, value)) - low) / spec["step"]) * spec["step"]
                    value = min(high, max(low, value))
                    delta[name] = int(value) if spec["kind"] == "integer" else float(value)
            result.append(delta)
        return {"candidates": result}


class AgenticDBBackend:
    """Reuse the upstream API client with no silent heuristic fallback or retries."""

    def __init__(self, client, ledger: Ledger, *, max_output_tokens: int = 1500):
        if client.dry_run or not client.api_key or not client.model:
            raise ValueError("Explicit real LLM client settings required")
        self.client, self.ledger, self.max_output_tokens = client, ledger, max_output_tokens
        self.client.response_retries = 0
        self.cache: dict[str, dict] = {}

    def complete(self, role: str, payload: dict) -> dict:
        text = json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False)
        cache_key = fingerprint([role, payload])
        if cache_key in self.cache:
            return self.cache[cache_key]
        # Byte-length is a conservative reservation, not a reported tokenizer count.
        reservation = len((PROMPTS[role] + text).encode()) + self.max_output_tokens + 64
        self.ledger.reserve_llm(reservation)
        try:
            result = self.client._call_openai_compatible(
                text, operation=f"admtune_{role}", system_prompt=PROMPTS[role],
                max_output_tokens=self.max_output_tokens,
                timeout_s=max(1, min(self.client.timeout_s, int(
                    self.ledger.remaining_seconds() - self.ledger.limits.recovery_reserve_seconds
                ))),
            )
        finally:
            usage = self.client.consume_last_usage()
            self.ledger.settle_llm(reservation, usage.get("total_tokens") if usage else None)
        if not isinstance(result, dict):
            raise ValueError("Agent output must be a JSON object")
        self.cache[cache_key] = result
        return result


class Team:
    def __init__(self, space: Space, backend: Backend, *, batch_size: int = 4, max_selected: int = 6):
        self.space, self.backend = space, backend
        self.batch_size, self.max_selected = batch_size, max_selected
        self.selected: list[str] = []
        self.ranges: dict = {}
        self.generation = 0
        self.feedback: dict = {}
        self.diagnostics: dict = {}

    def initialize(self, context: list[float], experiences: list[dict], *, diagnostics: dict | None = None):
        if diagnostics is not None:
            self.diagnostics = diagnostics
        payload = {"catalog": self.space.describe(), "context": context,
                   "experiences": experiences[:5], "max_selected": self.max_selected,
                   "feedback": self.feedback, "diagnostics": self.diagnostics,
                   "fixed_resource_context": self.space.fixed_resources,
                   "host_memory_budget_bytes": self.space.memory_bytes}
        selected = self.backend.complete("selector", payload).get("selected")
        if (
            not isinstance(selected, list) or not selected or len(set(selected)) != len(selected)
            or len(selected) > self.max_selected or set(selected) - set(self.space.names)
        ):
            raise ValueError("Invalid selected parameter set")
        self.selected = selected
        output = self.backend.complete("pruner", {**payload, "selected": selected})
        self.set_ranges(output.get("ranges"))

    def set_ranges(self, ranges: dict):
        if not isinstance(ranges, dict) or set(ranges) != set(self.selected):
            raise ValueError("Preferred ranges must cover selected knobs exactly")
        for name, preferred in ranges.items():
            knob = self.space.knobs[name]
            if set(preferred) != {"lower", "upper", "choices", "special"}:
                raise ValueError("Malformed preferred range")
            if any(value not in knob.special for value in preferred["special"]):
                raise ValueError("Invented special value")
            if knob.kind in {"enum", "boolean"}:
                if not preferred["choices"] or any(c not in knob.choices for c in preferred["choices"]):
                    raise ValueError("Invalid preferred enum options")
            else:
                low, high = preferred["lower"], preferred["upper"]
                if not isinstance(low, (float, int)) or not isinstance(high, (float, int)):
                    raise ValueError("Numeric preferred bounds required")
                if not math.isfinite(low + high) or not knob.lower <= low <= high <= knob.upper:
                    raise ValueError("Preferred range exceeded hard bounds")
                knob.validate(low)
                knob.validate(high)
        self.ranges = ranges

    def relax(self):
        self.ranges = {
            n: {key: getattr(self.space.knobs[n], key) for key in ["lower", "upper", "choices", "special"]}
            for n in self.selected
        }

    def recommend(self, parent: dict, phase_id: str, context: list[float], experiences: list[dict]) -> list[Candidate]:
        self.generation += 1
        output = self.backend.complete("recommender", {
            "catalog": self.space.describe(), "selected": self.selected, "ranges": self.ranges,
            "parent": parent, "context": context, "experiences": experiences[:5],
            "batch_size": self.batch_size,
            "generation": self.generation, "feedback": self.feedback,
            "diagnostics": self.diagnostics,
            "fixed_resource_context": self.space.fixed_resources,
            "host_memory_budget_bytes": self.space.memory_bytes,
        })
        deltas = output.get("candidates")
        if not isinstance(deltas, list) or not 1 <= len(deltas) <= self.batch_size:
            raise ValueError("Invalid candidate batch")
        candidates, seen = [], set()
        for delta in deltas:
            try:
                if not isinstance(delta, dict) or set(delta) - set(self.selected):
                    continue
                full = self.space.merge(parent, delta)
                for name, value in delta.items():
                    bounds = self.ranges[name]
                    knob = self.space.knobs[name]
                    if knob.kind in {"enum", "boolean"}:
                        if value not in bounds["choices"]:
                            raise ValueError("Outside preferred choices")
                    elif value not in bounds["special"] and not bounds["lower"] <= value <= bounds["upper"]:
                        raise ValueError("Outside preferred range")
                item = Candidate(full, phase_id, fingerprint(parent))
                if item.key != fingerprint(parent) and item.key not in seen:
                    candidates.append(item)
                    seen.add(item.key)
            except (ValueError, TypeError):
                continue
        return candidates
