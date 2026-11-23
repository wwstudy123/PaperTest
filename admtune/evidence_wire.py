"""反思证据编号的可逆传输编码，不修改事实、配置或验收规则。"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import re

from .contracts import fingerprint


class EvidenceWireBackend:
    """Only reflection IDs are aliased. The store still validates full records."""

    def __init__(self, backend, *, audit_dir: Path | None = None):
        self.backend, self.audit_dir = backend, audit_dir
        self.sequence = 0

    def complete(self, role: str, payload: dict) -> dict:
        if role != "reflection":
            return self.backend.complete(role, payload)
        # Old summaries may reference records outside this batch. Fail closed
        # rather than silently omit their references; this pilot sends none.
        if payload.get("old_summaries"):
            raise ValueError("Evidence wire encoding requires no old summaries")
        ids = [f["trial_id"] for f in payload["facts"]]
        if not ids or len(set(ids)) != len(ids):
            raise ValueError("Unique reflection evidence IDs required")
        encode = {identifier: f"t{i + 1}" for i, identifier in enumerate(ids)}
        decode = {alias: identifier for identifier, alias in encode.items()}
        wire = deepcopy(payload)
        for field in ("facts", "record_context"):
            for record in wire.get(field, []):
                if record["trial_id"] not in encode:
                    raise ValueError("Unknown reflection input evidence")
                record["trial_id"] = encode[record["trial_id"]]
        self.sequence += 1
        audit = {
            "encoding": "reflection_evidence_ids_v1", "alias_to_trial_id": decode,
            "original_payload_sha256": fingerprint(payload),
            "wire_payload_sha256": fingerprint(wire),
        }
        path = None
        if self.audit_dir is not None:
            self.audit_dir.mkdir(parents=True, exist_ok=True)
            path = self.audit_dir / f"reflection-{self.sequence:03d}.json"
            # Save before the call, including for an API/validation failure.
            with path.open("x", encoding="utf-8") as handle:
                path.chmod(0o600)
                json.dump(audit, handle, ensure_ascii=False, indent=2)
        try:
            result = deepcopy(self.backend.complete(role, wire))
            if not isinstance(result, dict):
                raise ValueError("Reflection must return an object")
            for fact in result.get("facts", []):
                if fact.get("trial_id") not in decode:
                    raise ValueError("Unknown reflection output evidence alias")
                fact["trial_id"] = decode[fact["trial_id"]]
            evidence = result.get("evidence_ids", [])
            if not isinstance(evidence, list) or any(x not in decode for x in evidence):
                raise ValueError("Unknown reflection output evidence alias")
            result["evidence_ids"] = [decode[x] for x in evidence]
            # Preserve the meaning of inline references when these hypotheses
            # are later retrieved without their per-call wire context.
            alias_pattern = r"(?<![A-Za-z0-9_])(" + "|".join(
                sorted(map(re.escape, decode), key=len, reverse=True)
            ) + r")(?![A-Za-z0-9_])"
            if isinstance(result.get("hypotheses"), list):
                result["hypotheses"] = [
                    re.sub(alias_pattern, lambda m: decode[m.group(1)], text)
                    if isinstance(text, str) else text for text in result["hypotheses"]
                ]
            # No fact/value/condition is filled in or repaired here. The store
            # rejects missing/changed facts using the original full-ID records.
            audit["decoded_response"] = result
            return result
        finally:
            if path is not None:
                path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
                path.chmod(0o600)
