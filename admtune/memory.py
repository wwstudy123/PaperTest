from __future__ import annotations

import json
import math
import sqlite3
from pathlib import Path

import numpy as np

from .contracts import Trial, canonical


class Store:
    """Append-only trials and atomic reflection/evidence consumption."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS trials (
                seq INTEGER PRIMARY KEY, trial_id TEXT UNIQUE NOT NULL, payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS events (
                seq INTEGER PRIMARY KEY, kind TEXT NOT NULL, payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS reflections (
                seq INTEGER PRIMARY KEY, payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS consumed (
                trial_id TEXT PRIMARY KEY, reflection_seq INTEGER NOT NULL);
        """)
        self.db.commit()

    def close(self):
        self.db.close()

    def event(self, kind: str, payload: dict):
        with self.db:
            self.db.execute("INSERT INTO events(kind,payload) VALUES (?,?)", (kind, canonical(payload)))

    def append(self, trial: Trial):
        with self.db:
            self.db.execute(
                "INSERT INTO trials(trial_id,payload) VALUES (?,?)",
                (trial.trial_id, canonical(trial.to_dict())),
            )

    def trials(self) -> list[Trial]:
        return [Trial(**json.loads(row[0])) for row in self.db.execute("SELECT payload FROM trials ORDER BY seq")]

    def compatible(
        self, environment_key: str, objective_key: str, context: list[float],
        *, max_distance: float = 4, phase_id: str | None = None, limit: int = 64,
    ) -> list[Trial]:
        eligible = []
        for trial in self.trials():
            if (
                not trial.trainable or trial.environment_key != environment_key
                or trial.objective_key != objective_key or len(trial.context) != len(context)
                or (phase_id is not None and trial.phase_id != phase_id)
            ):
                continue
            distance = float(np.sqrt(np.mean((np.asarray(trial.context) - context) ** 2)))
            if distance <= max_distance:
                eligible.append((distance, trial))
        # Keep a recent/diverse subset, not only high performers.
        eligible.sort(key=lambda pair: pair[0])
        return [trial for _, trial in eligible[:limit]]

    def pending(self, *, phase_id: str | None = None) -> list[Trial]:
        consumed = {row[0] for row in self.db.execute("SELECT trial_id FROM consumed")}
        return [t for t in self.trials() if t.trial_id not in consumed and (phase_id is None or t.phase_id == phase_id)]

    def reflections(self) -> list[dict]:
        return [json.loads(row[0]) for row in self.db.execute("SELECT payload FROM reflections ORDER BY seq")]

    def commit_reflection(self, payload: dict, new_ids: list[str]):
        required = {"facts", "hypotheses", "conditions", "actions", "evidence_ids"}
        if set(payload) != required or not isinstance(payload["conditions"], dict):
            raise ValueError("Reflection fields or conditions invalid")
        if not all(isinstance(payload[k], list) for k in ["facts", "hypotheses", "actions", "evidence_ids"]):
            raise ValueError("Reflection list fields invalid")
        ids = payload["evidence_ids"]
        if not ids or len(ids) != len(set(ids)) or not new_ids or not set(new_ids) <= set(ids):
            raise ValueError("New evidence must be explicitly referenced")
        records = {t.trial_id: t for t in self.trials()}
        if set(ids) - set(records):
            raise ValueError("Unknown evidence ID")
        for key, value in payload["conditions"].items():
            if key not in {"environment_key", "objective_key", "phase_id"} or any(getattr(records[i], key) != value for i in ids):
                raise ValueError("Reflection conditions do not match evidence")
        if not {"environment_key", "objective_key"} <= set(payload["conditions"]):
            raise ValueError("Environment and objective conditions required")
        allowed_facts = {
            canonical({"trial_id": i, "status": records[i].status, "value": records[i].value})
            for i in ids
        }
        if not payload["facts"] or any(canonical(f) not in allowed_facts for f in payload["facts"]):
            raise ValueError("Observed facts must match stored evidence exactly")
        if any(not isinstance(h, str) for h in payload["hypotheses"]):
            raise ValueError("Hypotheses must be text, not executable rules")
        if any(a not in {"resample", "adjust_range", "reselect", "verify", "stop"} for a in payload["actions"]):
            raise ValueError("Unsupported reflection action")
        # Both changes commit or neither does. An already-consumed ID fails atomically.
        with self.db:
            cursor = self.db.execute("INSERT INTO reflections(payload) VALUES (?)", (canonical(payload),))
            self.db.executemany(
                "INSERT INTO consumed(trial_id,reflection_seq) VALUES (?,?)",
                [(trial_id, cursor.lastrowid) for trial_id in new_ids],
            )
