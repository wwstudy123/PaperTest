from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess

import numpy as np

from .agents import HeuristicBackend, Team
from .budget import BudgetExhausted, Ledger, Limits
from .contracts import Environment, Objective, canonical
from .controller import Controller, Policy
from .execution import SimulatedExecutor
from .memory import Store
from .reflection import Reflector
from .space import Knob, Space
from .workload import PhaseDetector, Window, WorkloadEncoder


VARIANTS = ["full", "single-slice", "no-memory", "no-surrogate", "no-reflection", "full-reflection", "additive"]


def demo_space() -> Space:
    return Space([
        Knob("innodb_buffer_pool_size", "integer", 4 * 1024**3, 1024**3, 12 * 1024**3,
             step=1024**3, log_scale=True, unit="bytes"),
        Knob("innodb_io_capacity", "integer", 200, 100, 1500, step=100),
        Knob("innodb_io_capacity_max", "integer", 2000, 1500, 4000, step=500),
        Knob("sort_buffer_size", "integer", 2 * 1024**2, 1024**2, 8 * 1024**2,
             step=1024**2, log_scale=True, unit="bytes"),
    ], memory_bytes=32 * 1024**3)


def synthetic_window(index: int, share: float, rng) -> Window:
    # Two offered-request template clusters and a fixed unknown channel.
    counts = [[int(rng.poisson(300 * share)), int(rng.poisson(300 * (1 - share))), 0] for _ in range(6)]
    return Window(index * 30, (index + 1) * 30, counts)


def run_simulation(*, output: Path, seed: int = 0, scenario: str = "aba", variant: str = "full", trials: int = 24) -> dict:
    if variant not in VARIANTS or scenario not in {"static", "aba"} or trials < 6:
        raise ValueError("Invalid simulation configuration")
    output.mkdir(parents=True, exist_ok=False)
    space = demo_space()
    calibration_rng = np.random.default_rng(90001)  # Separate from evaluation seeds.
    calibration = [synthetic_window(i, 0.8, calibration_rng) for i in range(30)]
    encoder = WorkloadEncoder(calibration, single_slice=variant == "single-slice")
    detector = PhaseDetector(encoder, change_threshold=3.0, stable_threshold=1.5)
    policy = Policy(
        surrogate_gate=variant != "no-surrogate", cross_phase_memory=variant != "no-memory",
        kernel="additive" if variant == "additive" else "joint",
    )
    environment = Environment("synthetic_mysql", "fixture", "synthetic", "synthetic", space.key,
                              "synthetic_v1", space.key + ":" + str(encoder.single_slice))
    ledger = Ledger(Limits(trials=trials, wall_seconds=600, recovery_reserve_seconds=1))
    backend = HeuristicBackend(seed)
    store = Store(output / "records.sqlite3")
    reflection_mode = {"no-reflection": "none", "full-reflection": "full"}.get(variant, "incremental")
    controller = Controller(
        space=space, environment=environment, objective=Objective(), detector=detector,
        executor=SimulatedExecutor(space, seed), team=Team(space, backend),
        reflector=Reflector(store, backend, mode=reflection_mode), store=store,
        ledger=ledger, policy=policy, seed=seed,
    )
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = "unavailable"
    manifest = {
        "simulated": True, "seed": seed, "scenario": scenario, "variant": variant,
        "environment": asdict(environment), "space": space.describe(), "space_key": space.key,
        "encoder": encoder.metadata(), "policy": asdict(policy), "budget": asdict(ledger.limits),
        "python": platform.python_version(), "git_revision": revision,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "thresholds": {"change": 3.0, "stable": 1.5, "calibration": "synthetic fixture only"},
        "backend": "heuristic fixture; no API calls",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    rng = np.random.default_rng(seed + 1000)
    # Enough settling windows to exercise A -> B -> A without feeding labels to detector.
    shares = [0.8] * 10 + [0.2] * 10 + [0.8] * 10 if scenario == "aba" else [0.8] * 40
    try:
        for i, share in enumerate(shares):
            controller.observe(synthetic_window(i, share, rng))
            controller.step()
        summary = controller.finalize()
        summary["heuristic_role_invocations"] = len(backend.calls)
        summary["detected_phases"] = detector.phase
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        with (output / "trials.jsonl").open("w") as handle:
            for trial in store.trials():
                handle.write(canonical(trial.to_dict()) + "\n")
        return summary
    finally:
        store.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="ADMTune: offline first; real execution requires explicit preflight")
    parser.add_argument("--simulate", action="store_true", help="Run synthetic plumbing only; no network or DB")
    parser.add_argument("--scenario", choices=["static", "aba"], default="aba")
    parser.add_argument("--variant", choices=VARIANTS, default="full")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trials", type=int, default=24)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if not args.simulate:
        parser.error("Use --simulate. No live or paid operation is implicit.")
    output = args.output or Path("runs") / ("synthetic-" + datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
    try:
        summary = run_simulation(output=output, seed=args.seed, scenario=args.scenario, variant=args.variant, trials=args.trials)
    except (ValueError, BudgetExhausted) as exc:
        parser.exit(1, f"Run stopped: {exc}\n")
    print(json.dumps({"output": str(output), **summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
