"""Synthetic ablation matrix and descriptive audit reports; no implicit live runs."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics

from .cli import VARIANTS, run_simulation
from .contracts import Objective
from .memory import Store


def audit_run(path: Path, *, target_utility: float = 1.05) -> dict:
    if target_utility <= 1:
        raise ValueError("Predeclare an improvement target above one")
    manifest = json.loads((path / "manifest.json").read_text())
    summary = json.loads((path / "summary.json").read_text())
    store = Store(path / "records.sqlite3")
    try:
        records = store.trials()
        events = [(kind, json.loads(payload)) for kind, payload in store.db.execute(
            "SELECT kind,payload FROM events ORDER BY seq"
        )]
    finally:
        store.close()
    if any(t.metadata.get("simulated") != summary["simulated"] for t in records):
        raise ValueError("Mixed simulated and real records")
    objective = Objective(*summary["objective"].split(":"))
    phase_baselines, phase_counts, first_hit = {}, {}, {}
    errors, widths, covered = [], [], []
    for record in records:
        if record.metadata.get("kind") == "verification":
            continue  # Independent verification does not retroactively improve search curves.
        phase_counts[record.phase_id] = phase_counts.get(record.phase_id, 0) + 1
        if record.metadata.get("kind") == "baseline" and record.trainable:
            phase_baselines[record.phase_id] = record.value
        baseline = phase_baselines.get(record.phase_id)
        if record.trainable and record.constraints_ok and baseline is not None:
            if objective.utility(record.value, baseline) >= target_utility:
                first_hit.setdefault(record.phase_id, phase_counts[record.phase_id])
        if record.trainable and record.prediction:
            p = record.prediction
            error = objective.target(record.value) - p["mean"]
            beta = manifest.get("policy", {}).get("beta", 2)
            errors.append(error)
            widths.append(2 * beta * p["observation_std"])
            covered.append(abs(error) <= beta * p["observation_std"])
    counts = {}
    for record in records:
        counts[record.status] = counts.get(record.status, 0) + 1
    reflection_events = [p for kind, p in events if kind == "reflection_committed"]
    return {
        "run": str(path), "simulated": summary["simulated"],
        "variant": manifest.get("variant", "unlabeled_live"), "seed": manifest.get("seed"),
        "scenario": manifest.get("scenario", "static"),
        "trial_slots_used": summary["budget"]["trial_slots_used"],
        "target_utility": target_utility,
        "phase_target_trials": {p: first_hit.get(p) for p in phase_baselines},
        "target_misses_are_null_not_dropped": True,
        "measured_log_mae": statistics.mean(map(abs, errors)) if errors else None,
        "measured_log_rmse": math.sqrt(statistics.mean(e**2 for e in errors)) if errors else None,
        "observation_interval_coverage": statistics.mean(covered) if covered else None,
        "mean_log_interval_width": statistics.mean(widths) if widths else None,
        "prediction_observations": len(errors),
        "prediction_evaluation": "selected online observations only; selection-biased, not a held-out audit",
        "statuses": counts, "reflection_calls_committed": len(reflection_events),
        "reflection_new_records": sum(p["new_records"] for p in reflection_events),
        "reflection_input_records": sum(p["input_records"] for p in reflection_events),
        "llm_api_calls": summary["budget"]["llm_calls"],
        "llm_tokens_or_reserved": summary["budget"]["llm_tokens_or_reserved"],
        "independent_verification_values": summary["independent_verification_values"],
        "is_thesis_evidence": False,
        "data_provenance": "synthetic" if summary["simulated"] else "live_unreviewed",
        "warning": summary.get("warning"),
    }


def run_suite(output: Path, *, seeds: list[int], variants: list[str], scenario: str, trials: int) -> dict:
    if not seeds or len(seeds) != len(set(seeds)) or not variants or set(variants) - set(VARIANTS):
        raise ValueError("Unique seeds and supported variants required")
    output.mkdir(parents=True, exist_ok=False)
    report = {"simulated": True, "is_thesis_evidence": False, "runs": [], "failures": []}
    for variant in variants:
        for seed in seeds:
            path = output / f"{scenario}-{variant}-seed{seed}"
            try:
                run_simulation(output=path, seed=seed, variant=variant, scenario=scenario, trials=trials)
                audit = audit_run(path)
                report["runs"].append(audit)
                (path / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
            except Exception as exc:
                report["failures"].append({"variant": variant, "seed": seed, "error": str(exc)})
    (output / "suite.json").write_text(json.dumps(report, indent=2) + "\n")
    columns = [
        "variant", "seed", "scenario", "simulated", "trial_slots_used", "measured_log_mae",
        "observation_interval_coverage", "reflection_calls_committed", "reflection_input_records",
        "llm_api_calls",
    ]
    with (output / "suite.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(report["runs"])
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulate", action="store_true", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scenario", choices=["static", "aba"], default="aba")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=VARIANTS)
    parser.add_argument("--trials", type=int, default=24)
    args = parser.parse_args()
    report = run_suite(args.output, seeds=args.seeds, variants=args.variants, scenario=args.scenario, trials=args.trials)
    print(json.dumps({
        "output": str(args.output), "simulated": True, "runs": len(report["runs"]),
        "failures": report["failures"], "paid_api_calls": sum(r["llm_api_calls"] for r in report["runs"]),
        "warning": "Synthetic plumbing validation only; no database performance evidence.",
    }, indent=2))
    return 1 if report["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
