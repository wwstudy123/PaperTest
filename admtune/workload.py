from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from typing import Any

import numpy as np


class TemplateCatalog:
    """Frozen SQL AST templates and DBSCAN clusters; unseen templates stay unknown."""

    def __init__(self, sql: list[str], *, eps: float = 0.2, min_samples: int = 2):
        from sqlglot import exp, parse_one
        self._exp, self._parse = exp, parse_one
        templates = sorted({self.normalize(q) for q in sql})
        if not templates or eps < 0 or min_samples < 1:
            raise ValueError("Initialization SQL and valid clustering parameters required")
        kinds = [exp.Select, exp.Insert, exp.Update, exp.Delete, exp.Join,
                 exp.AggFunc, exp.Group, exp.Order, exp.Where]
        features = []
        for text in templates:
            tree = self._parse(text, read="mysql")
            features.append([sum(1 for _ in tree.find_all(k)) for k in kinds])
        x = np.asarray(features, dtype=float)
        x /= np.maximum(np.max(x, axis=0), 1)
        distance = np.linalg.norm(x[:, None] - x[None, :], axis=2)
        labels = [-2] * len(x)
        cluster = 0
        for i in range(len(x)):
            if labels[i] != -2:
                continue
            neighbors = list(np.flatnonzero(distance[i] <= eps))
            if len(neighbors) < min_samples:
                labels[i] = -1
                continue
            labels[i] = cluster
            queue, seen = deque(neighbors), set(neighbors)
            while queue:
                j = queue.popleft()
                if labels[j] == -1:
                    labels[j] = cluster
                if labels[j] != -2:
                    continue
                labels[j] = cluster
                other = list(np.flatnonzero(distance[j] <= eps))
                if len(other) >= min_samples:
                    for k in other:
                        if k not in seen:
                            queue.append(k)
                            seen.add(k)
            cluster += 1
        for i in range(len(labels)):
            if labels[i] == -1:  # Do not collapse unrelated noise templates together.
                labels[i], cluster = cluster, cluster + 1
        self.mapping = dict(zip(templates, labels))
        self.cluster_count = cluster
        self.unknown_channel = cluster

    def normalize(self, sql: str) -> str:
        tree = self._parse(sql, read="mysql")
        tree = tree.transform(
            lambda node: self._exp.Placeholder() if isinstance(node, self._exp.Literal) else node,
        )
        return tree.sql(dialect="mysql", normalize=True, comments=False)

    def channel(self, sql: str) -> int:
        return self.mapping.get(self.normalize(sql), self.unknown_channel)


@dataclass(frozen=True)
class Window:
    start: float
    end: float
    counts: list[list[float]]  # time slice x frozen template cluster, including unknown
    source: str = "external_scheduled"

    def features(self) -> tuple[np.ndarray, np.ndarray]:
        x = np.asarray(self.counts, dtype=float)
        if (
            self.source != "external_scheduled" or x.ndim != 2 or min(x.shape) < 1
            or not np.isfinite(x).all() or (x < 0).any()
            or not math.isfinite(self.start + self.end) or self.end <= self.start
        ):
            raise ValueError("Require finite external scheduled counts and a valid complete window")
        rates = x.sum(axis=1) / ((self.end - self.start) / len(x))
        shares = x / np.maximum(x.sum(axis=1, keepdims=True), 1)
        return np.log1p(rates), shares.ravel()


class WorkloadEncoder:
    """Fit on independent calibration windows only; never refit on test phases."""

    def __init__(self, calibration: list[Window], *, single_slice: bool = False):
        if not calibration:
            raise ValueError("Independent calibration windows required")
        self.single_slice = single_slice
        self.shape = np.asarray(calibration[0].counts).shape
        raw = np.asarray([self.raw(w) for w in calibration])
        self.mean = raw.mean(axis=0)
        self.scale = np.maximum(raw.std(axis=0), 0.1)

    def raw(self, window: Window) -> np.ndarray:
        window.features()
        if np.asarray(window.counts).shape != self.shape:
            raise ValueError("Workload encoding dimensions changed")
        if self.single_slice:
            counts = np.sum(window.counts, axis=0, keepdims=True).tolist()
            window = Window(window.start, window.end, counts, window.source)
        intensity, composition = window.features()
        return np.r_[intensity, composition]

    def encode(self, window: Window) -> list[float]:
        return ((self.raw(window) - self.mean) / self.scale).tolist()

    @property
    def intensity_dimensions(self) -> int:
        return 1 if self.single_slice else self.shape[0]

    def metadata(self) -> dict[str, Any]:
        return {
            "shape": list(self.shape), "single_slice": self.single_slice,
            "mean": self.mean.tolist(), "scale": self.scale.tolist(),
            "source": "external_scheduled",
        }


@dataclass(frozen=True)
class Detection:
    state: str
    event: str | None
    phase_id: str | None
    context: list[float]


class PhaseDetector:
    def __init__(
        self, encoder: WorkloadEncoder, *, change_threshold: float,
        stable_threshold: float, confirm_windows: int = 2, stable_windows: int = 2,
    ):
        if confirm_windows < 2 or stable_windows < 2 or min(change_threshold, stable_threshold) <= 0:
            raise ValueError("Positive calibrated thresholds and at least two windows required")
        self.encoder = encoder
        self.change_threshold, self.stable_threshold = change_threshold, stable_threshold
        self.confirm_windows, self.stable_windows = confirm_windows, stable_windows
        self.state = "initializing"
        self.phase = 0
        self.reference = None
        self.changed_count = 0
        self.recent: deque = deque(maxlen=stable_windows)
        self.last_end: float | None = None

    def distance(self, a, b) -> float:
        difference = np.asarray(a) - np.asarray(b)
        m = self.encoder.intensity_dimensions
        # Separate intensity and composition so numerous template dimensions do not drown intensity.
        return max(float(np.sqrt(np.mean(block**2))) for block in (difference[:m], difference[m:]))

    def observe(self, window: Window) -> Detection:
        if self.last_end is not None and window.start < self.last_end:
            raise ValueError("Use chronological non-overlapping windows")
        z = self.encoder.encode(window)
        self.last_end = window.end
        event = None
        if self.state == "stable":
            self.changed_count = self.changed_count + 1 if self.distance(z, self.reference) > self.change_threshold else 0
            if self.changed_count:
                event = "change_suspected"
            if self.changed_count >= self.confirm_windows:
                self.state, event = "waiting_stable", "change_confirmed"
                self.recent.clear()  # Stability uses only complete windows AFTER confirmation.
        else:
            self.recent.append(z)
            if len(self.recent) == self.stable_windows:
                center = np.mean(self.recent, axis=0)
                if all(self.distance(item, center) <= self.stable_threshold for item in self.recent):
                    self.phase += 1
                    self.reference = center
                    self.state, event, self.changed_count = "stable", "phase_ready", 0
                    self.recent.clear()
        return Detection(self.state, event, f"phase-{self.phase}" if self.phase else None, z)
