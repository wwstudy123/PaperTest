from __future__ import annotations

from .agents import Backend
from .memory import Store


class Reflector:
    def __init__(
        self, store: Store, backend: Backend, *, mode: str = "incremental",
        min_batch: int = 3, max_batch: int = 12,
        old_summary_limit: int = 3,
    ):
        if mode not in {"none", "incremental", "full"} or not 1 <= min_batch <= max_batch:
            raise ValueError("Invalid reflection settings")
        self.store, self.backend, self.mode = store, backend, mode
        self.min_batch, self.max_batch = min_batch, max_batch
        if old_summary_limit < 0:
            raise ValueError("Summary limit must be nonnegative")
        self.old_summary_limit = old_summary_limit

    def trigger(self, events: list[str], *, force: bool = False, phase_id: str | None = None) -> bool:
        pending = self.store.pending(phase_id=phase_id)
        if self.mode == "none" or not pending or (not force and (not events or len(pending) < self.min_batch)):
            return False
        pending = pending[:self.max_batch]
        environment, objective = pending[0].environment_key, pending[0].objective_key
        pending = [t for t in pending if (t.environment_key, t.objective_key) == (environment, objective)]
        records = pending
        if self.mode == "full":
            # Bounded full-history comparison; older facts are context, not consumed twice.
            compatible = [
                t for t in self.store.trials()
                if (t.environment_key, t.objective_key) == (environment, objective)
                and (phase_id is None or t.phase_id == phase_id)
                and t.trial_id not in {p.trial_id for p in pending}
            ]
            records = compatible[-(64 - len(pending)):] + pending
        facts = [{"trial_id": t.trial_id, "status": t.status, "value": t.value} for t in records]
        phase_ids = {t.trial_id for t in self.store.trials() if phase_id is None or t.phase_id == phase_id}
        old_summaries = [s for s in self.store.reflections() if set(s["evidence_ids"]) <= phase_ids]
        payload = {
            "facts": facts,
            "conditions": {"environment_key": environment, "objective_key": objective},
            "events": sorted(set(events)),
            "old_summaries": old_summaries[-self.old_summary_limit:]
            if self.mode == "incremental" and self.old_summary_limit else [],
            "record_context": [
                {"trial_id": t.trial_id, "config": t.config, "phase_id": t.phase_id,
                 "kind": t.metadata.get("kind"), "constraints_ok": t.constraints_ok}
                for t in records
            ],
        }
        # Failure leaves all evidence pending. No automatic free/unbounded API retries.
        output = self.backend.complete("reflection", payload)
        self.store.commit_reflection(output, [t.trial_id for t in pending])
        self.store.event("reflection_committed", {
            "mode": self.mode, "new_records": len(pending), "input_records": len(records),
        })
        return True
