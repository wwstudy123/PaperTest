"""独立零LLM拒绝审计：按原GP的乐观改善幅度选一个全拒批次候选，B-C-B实测。"""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import sqlite3

from dotenv import load_dotenv
from config import load_app_config
from .calibration import execute_trial, prepare
from .continuous import atomic_json
from .contracts import fingerprint, Objective
from .dynamic import instance_lock
from .dynamic_runtime import GenerationalRequests
from .live import load_space
from .provenance import source_provenance
from .real_execution import AgenticDBExecutor
from .replay import ReplayPlan, replay
from .replay_report import read_jsonl
from .security import redact
from .space import Space, Knob
from .surrogate import GaussianProcess


def select_rejection(run):
    manifest = json.loads((run/"manifest.json").read_text())
    summary = json.loads((run/"summary.json").read_text())
    space = Space([Knob(**k) for k in manifest["space"]])
    phases = [p["phase_id"] for p in summary["phases"]]
    candidate_map, context_map = {}, {}
    for index, p in enumerate(sorted((run/"llm_calls").glob("*-recommender.json"))):
        a = json.loads(p.read_text()); phase = phases[index]
        context_map[phase] = a["payload"]["context"]
        candidate_map[phase] = {}
        for delta in a["response"]["candidates"]:
            cfg = space.merge(a["payload"]["parent"], delta)
            candidate_map[phase][fingerprint(cfg)] = cfg
    trials = {r["trial_id"]: r for r in read_jsonl(run/"trials.jsonl")}
    db = sqlite3.connect(f"file:{run/'records.sqlite3'}?mode=ro", uri=True)
    baseline, incumbent, choices = {}, {}, []
    try:
        for seq, kind, raw in db.execute("SELECT seq,kind,payload FROM events ORDER BY seq"):
            a = json.loads(raw); phase = a.get("phase_id")
            if kind == "config_accepted":
                incumbent[phase] = a
                if trials[a["trial_id"]]["metadata"]["kind"] == "baseline":
                    baseline[phase] = a["value"]
            if kind != "prescreen" or a["eligible"]:
                continue
            for key, prediction in zip(a["config_ids"], a["predictions"]):
                cfg = candidate_map[phase][key]; context = context_map[phase]
                path = run/"gp_models"/f"gp-{prediction['model_version']:04d}.json"
                gp = GaussianProcess.from_snapshot(json.loads(path.read_text()))
                rebuilt = gp.predict([space.encode(cfg)+context])[0]
                if any(abs(getattr(rebuilt, name)-prediction[name]) > 1e-9
                       for name in ("mean", "observation_std", "latent_std")):
                    raise ValueError("Original prediction cannot be reconstructed; do not guess context")
                best = incumbent[phase]
                margin = prediction["mean"]+2*prediction["observation_std"]+math.log(best["value"])
                choices.append({"source_event_seq": seq, "phase_id": phase,
                    "candidate_id": key, "candidate_config": cfg, "reference_config": best["config"],
                    "reference_value_at_rejection": best["value"], "baseline_value_at_rejection": baseline[phase],
                    "prediction": prediction, "context": context, "optimistic_log_gain": margin,
                    "steady_scan_percent": 10 if phases.index(phase) == 1 else 90,
                    "original_admission_ceiling_p95_ms": baseline[phase]/.8,
                    "original_prediction_reconstructed": True,
                    "model_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    finally:
        db.close()
    if not choices:
        raise ValueError("No all-rejected batch available")
    selected = max(choices, key=lambda r: r["optimistic_log_gain"])
    return {**selected, "selection_rule": "highest original optimistic log improvement over incumbent, "
            "among candidates in completely rejected batches; selection before audit observations",
            "source_manifest_sha256": hashlib.sha256((run/"manifest.json").read_bytes()).hexdigest(),
            "rejected_candidates_available": len(choices), "audit_candidates": 1, "llm_calls": 0,
            "scope": "Independent post-run diagnostic; never backfilled into source model, budget or 30-minute curve"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("target", "benchmark", "knobs", "selection", "output"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--dedicated-instance", action="store_true")
    a = p.parse_args()
    if not a.dedicated_instance: p.error("Dedicated database required")
    root = Path(__file__).resolve().parents[1]; load_dotenv(root/".env", override=True)
    app = load_app_config(project_root=root, target_path=a.target, benchmark_path=a.benchmark,
                          knobs_path=a.knobs, dry_run_override=False)
    selection = json.loads(a.selection.read_text())
    lock = instance_lock(app.target.mysql.config_path)
    a.output.mkdir(mode=0o700, parents=True, exist_ok=False)
    space = load_space(a.knobs, memory_bytes=app.target.hardware_memory_bytes)
    ex = AgenticDBExecutor(app, space, Objective(), a.output, allow_mutations=True, local=True)
    records, result = [], {"completed": False, "llm_calls": 0}
    try:
        pre = ex.preflight()
        ref = space.merge(selection["reference_config"], {})
        candidate = space.merge(selection["candidate_config"], {})
        atomic_json(a.output/"manifest.json", {"selection": selection, "preflight": pre,
            "plan": ["reference", "rejected_candidate", "reference"], "new_measurements": 3,
            "warmup_stream_seconds": 180, "audit_seed": 813, "simulated": False, **source_provenance()})
        prepare(ex, ref)
        wp = ReplayPlan(scenario="steady", phase_seconds=60, tables=app.benchmark.workload.tables,
            table_size=app.benchmark.workload.table_size, read_percent=100, query_profile="range-mixed",
            range_rows=32768, top_k=8192, steady_scan_percent=selection["steady_scan_percent"], seed=813)
        client = GenerationalRequests(asdict(ex.settings), ref)
        folder = a.output/"warmup"; folder.mkdir()
        try:
            warm = replay(wp, client, folder, simulated=False)
            atomic_json(folder/"summary.json", warm)
            if not warm["complete"]: raise RuntimeError("Audit warmup incomplete")
        finally:
            client.close()
        for i, cfg in enumerate((ref, candidate, ref)):
            spec = {"name": f"audit-{i}", "kind": "rejected_candidate" if i == 1 else "reference",
                    "rate": 10, "read_percent": 100, "query_profile": "range-mixed",
                    "range_rows": 32768, "top_k": 8192, "seed": 813,
                    "steady_scan_percent": selection["steady_scan_percent"]}
            r = execute_trial(ex, asdict(ex.settings), spec, cfg, a.output,
                              app.benchmark.workload.tables, app.benchmark.workload.table_size)
            records.append(r)
            with (a.output/"trials.jsonl").open("a") as f:
                f.write(json.dumps(r, allow_nan=False)+"\n")
            atomic_json(a.output/"progress.json", {"completed_trials": len(records),
                                                  "last_p95_ms": r["value_p95_ms"]})
            if not r["valid"]: raise RuntimeError("Invalid audit observation")
        if len({r["query_result_signature"] for r in records}) != 1:
            raise ValueError("Query result signatures differ")
        left, observed, right = [r["value_p95_ms"] for r in records]
        reference = (left+right)/2; drift = abs(right/left-1)
        gain = reference/observed-1
        result.update(completed=True, reference_left_ms=left, candidate_ms=observed,
            reference_right_ms=right, reference_drift=drift, utility_gain_vs_bracket=gain,
            candidate_under_original_admission_ceiling=observed <= selection["original_admission_ceiling_p95_ms"],
            potential_missed_gain_signal=drift <= .1 and gain > max(.01, 2*drift),
            original_prediction_p95_ms=math.exp(-selection["prediction"]["mean"]),
            conclusion_scope="One candidate and one bracket; no overall false-rejection-rate claim")
    except Exception as exc:
        result["error"] = redact(str(exc))
    finally:
        try:
            ex.restore_initial(); result["initial_configuration_restored"] = ex.initial is not None
        except Exception as exc: result.update(completed=False, recovery_error=redact(str(exc)))
        ex.close(); lock.close(); atomic_json(a.output/"summary.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["completed"] else 1


if __name__ == "__main__":
    main()
