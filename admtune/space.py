from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

from dbms_policy import enforce_dbms_only
from .contracts import fingerprint


@dataclass(frozen=True)
class Knob:
    name: str
    kind: str
    default: Any
    lower: float | None = None
    upper: float | None = None
    choices: tuple[Any, ...] = ()
    special: tuple[Any, ...] = ()
    step: float = 1
    log_scale: bool = False
    restart_required: bool = False
    unit: str | None = None
    description: str = ""

    def __post_init__(self):
        enforce_dbms_only("db_config", {self.name: self.default})
        if self.kind not in {"integer", "float", "enum", "boolean"}:
            raise ValueError(f"Unsupported type: {self.kind}")
        if self.kind in {"integer", "float"}:
            if self.lower is None or self.upper is None or self.lower >= self.upper:
                raise ValueError("Numeric knobs require fixed lower < upper")
            if not all(math.isfinite(v) for v in [self.lower, self.upper, self.step]) or self.step <= 0:
                raise ValueError("Finite bounds and positive step required")
            if self.log_scale and self.lower <= 0:
                raise ValueError("Log bounds must be positive; encode sentinels separately")
        elif not self.choices:
            raise ValueError("Categorical knobs need explicit choices")
        self.validate(self.default)

    def validate(self, value: Any) -> Any:
        if value in self.special and type(value) in {int, float, str, bool}:
            return value
        if self.kind in {"enum", "boolean"}:
            if not any(type(value) is type(v) and value == v for v in self.choices):
                raise ValueError(f"{self.name}: invalid categorical value")
            return value
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{self.name}: finite numeric value required")
        if self.kind == "integer" and value != int(value):
            raise ValueError(f"{self.name}: integer required")
        if not self.lower <= value <= self.upper:
            raise ValueError(f"{self.name}: outside hard bounds")
        lattice = (value - self.lower) / self.step
        if abs(lattice - round(lattice)) > 1e-7:
            raise ValueError(f"{self.name}: not aligned to step")
        return int(value) if self.kind == "integer" else float(value)

    def encode(self, value: Any) -> list[float]:
        value = self.validate(value)
        if self.kind in {"enum", "boolean"}:
            return [float(type(value) is type(c) and value == c) for c in self.choices]
        flags = [float(value == v) for v in self.special]
        if any(flags):
            return [0.0, *flags]
        transform = math.log if self.log_scale else float
        scaled = (transform(value) - transform(self.lower)) / (transform(self.upper) - transform(self.lower))
        return [scaled, *flags]


class Space:
    def __init__(self, knobs: list[Knob], *, memory_bytes: int = 0):
        self.knobs = {k.name: k for k in knobs}
        if not knobs or len(self.knobs) != len(knobs):
            raise ValueError("A nonempty unique fixed knob universe is required")
        self.names = tuple(sorted(self.knobs))
        self.memory_bytes = memory_bytes

    @property
    def key(self) -> str:
        return fingerprint([asdict(self.knobs[n]) for n in self.names])

    @property
    def defaults(self) -> dict:
        return {n: self.knobs[n].default for n in self.names}

    def merge(self, parent: dict, delta: dict) -> dict:
        if set(parent) != set(self.names) or set(delta) - set(self.names):
            raise ValueError("Unknown fields or incomplete parent configuration")
        enforce_dbms_only("db_config", delta)
        full = {**parent, **delta}
        full = {n: self.knobs[n].validate(full[n]) for n in self.names}
        if (
            "innodb_io_capacity" in full and "innodb_io_capacity_max" in full
            and full["innodb_io_capacity"] > full["innodb_io_capacity_max"]
        ):
            raise ValueError("innodb_io_capacity exceeds innodb_io_capacity_max")
        if self.memory_bytes:
            global_bytes = sum(float(full.get(n, 0)) for n in ["innodb_buffer_pool_size", "key_buffer_size"])
            per_connection = sum(float(full.get(n, 0)) for n in [
                "sort_buffer_size", "join_buffer_size", "read_buffer_size", "read_rnd_buffer_size",
            ])
            if global_bytes + float(full.get("max_connections", 1)) * per_connection > self.memory_bytes * 0.8:
                raise ValueError("Conservative joint memory budget exceeded")
        return full

    def encode(self, config: dict) -> list[float]:
        full = self.merge(config, {})
        return [v for name in self.names for v in self.knobs[name].encode(full[name])]

    def describe(self) -> list[dict]:
        return [asdict(self.knobs[n]) for n in self.names]
