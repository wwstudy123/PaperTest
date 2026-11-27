"""同一真实观测序列的增量/全量反思输入审计；模拟输出仅供事务推进，不评价LLM质量。"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import tempfile

from admtune.agents import HeuristicBackend
from admtune.continuous import atomic_json
from admtune.contracts import Trial
from admtune.evidence_wire import EvidenceWireBackend
from admtune.memory import Store
from admtune.reflection import Reflector
from admtune.replay_report import read_jsonl, write_csv


def replay(records, mode):
    captured = []
    fixture = HeuristicBackend()

    class Capture:
        def complete(self, role, payload):
            captured.append(deepcopy(payload))
            return fixture.complete(role, payload)

    with tempfile.TemporaryDirectory(prefix="reflection-input-audit-") as tmp:
        store = Store(Path(tmp) / "evidence.sqlite3")
        reflector = Reflector(store, EvidenceWireBackend(Capture()), mode=mode, old_summary_limit=0)
        try:
            for i, record in enumerate(records):
                store.append(Trial(**record))
                last_phase_record = i + 1 == len(records) or records[i + 1]["phase_id"] != record["phase_id"]
                if last_phase_record:
                    reflector.trigger(["phase_end"], force=True)
        finally:
            store.close()
    return [{"mode": mode, "trigger": i + 1, "input_records": len(p["facts"]),
             "serialized_input_bytes": len(json.dumps(p, ensure_ascii=False, separators=(",", ":")).encode())}
            for i, p in enumerate(captured)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trials", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    records = read_jsonl(a.trials)
    rows = replay(records, "incremental") + replay(records, "full")
    a.output.mkdir(parents=True, exist_ok=False)
    write_csv(a.output / "reflection_input_volume.csv", rows, list(rows[0]))
    result = {"scope": "input-selection audit using real old observations; output text is mocked and unused",
              "actual_llm_calls": 0, "database_calls": 0, "real_observation_records": len(records),
              "note": "Bytes/record appearances are not actual tokens, reflection quality or DB performance.",
              "totals": {mode: {
                  "record_appearances": sum(r["input_records"] for r in rows if r["mode"] == mode),
                  "input_bytes": sum(r["serialized_input_bytes"] for r in rows if r["mode"] == mode)}
                  for mode in ("incremental", "full")}}
    atomic_json(a.output / "summary.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
