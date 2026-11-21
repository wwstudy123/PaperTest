"""Explicit preflight, manual pilot and static tuning on a dedicated MySQL instance."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

from dotenv import load_dotenv
import yaml

from config import load_app_config
from llm_client import LLMClient
from .agents import AgenticDBBackend, HeuristicBackend, Team
from .budget import BudgetExhausted, Ledger, Limits
from .contracts import Environment, Objective, fingerprint
from .controller import Controller, Policy
from .memory import Store
from .real_execution import AgenticDBExecutor
from .reflection import Reflector
from .space import Knob, Space
from .workload import PhaseDetector, Window, WorkloadEncoder
from .security import redact
from .provenance import source_provenance


def load_space(path: Path, *, memory_bytes: int) -> Space:
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError("Knob catalog must be a mapping")
    knobs = []
    for name, spec in data.items():
        knobs.append(Knob(
            name, {"string": "enum"}.get(spec["type"], spec["type"]), spec["default"],
            lower=spec.get("min"), upper=spec.get("max"),
            choices=tuple(spec.get("allowed_values") or []),
            special=tuple(spec.get("special_values") or []), step=spec.get("step", 1),
            log_scale=spec.get("log_scale", False),
            restart_required=spec.get("restart_required", False),
            unit=spec.get("unit"), description=spec.get("description", ""),
        ))
    return Space(knobs, memory_bytes=memory_bytes)


def static_tune(executor, space, environment, output, *, seed, trials, wall_seconds, backend_kind, llm_calls, llm_tokens):
    objective = Objective()
    ledger = Ledger(Limits(
        trials=trials, wall_seconds=wall_seconds, llm_calls=llm_calls,
        llm_tokens=llm_tokens, recovery_reserve_seconds=180,
    ))
    if backend_kind == "llm":
        required = ["OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL"]
        if not all(os.getenv(name) for name in required):
            raise ValueError("Explicit API key, base URL and model required for paid LLM")
        client = LLMClient.from_env(dry_run=False)
        client.allow_fallback = False
        backend = AgenticDBBackend(client, ledger)
    else:
        backend = HeuristicBackend(seed)
    store = Store(output / "records.sqlite3")
    # No fabricated offered-load telemetry: this command declares a fixed protocol.
    encoder = WorkloadEncoder([Window(0, 1, [[1, 0]])])
    detector = PhaseDetector(encoder, change_threshold=3, stable_threshold=1)
    detector.state, detector.phase = "stable", 1
    controller = Controller(
        space=space, environment=environment, objective=objective, detector=detector,
        executor=executor, team=Team(space, backend),
        reflector=Reflector(store, backend), store=store, ledger=ledger, policy=Policy(), seed=seed,
    )
    try:
        controller.begin_phase("static-1", [0.0])
        while controller.active:
            controller.step()
        return controller.finalize()
    except BudgetExhausted as exc:
        store.event("budget_stop", {"reason": str(exc)})
        return controller.finalize()
    finally:
        with (output / "trials.jsonl").open("w") as handle:
            for record in store.trials():
                handle.write(json.dumps(record.to_dict(), allow_nan=False) + "\n")
        store.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["preflight", "pilot", "tune-static"])
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--knobs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--dedicated-instance", action="store_true")
    parser.add_argument("--allow-restart", action="store_true")
    parser.add_argument("--allow-paid-llm", action="store_true")
    parser.add_argument("--delta", type=Path, help="Pilot-only JSON configuration delta; omitted means baseline")
    parser.add_argument("--backend", choices=["heuristic", "llm"], default="heuristic")
    parser.add_argument("--trials", type=int, default=12)
    parser.add_argument("--wall-seconds", type=float, default=6000)
    parser.add_argument("--llm-calls", type=int, default=16)
    parser.add_argument("--llm-tokens", type=int, default=60000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.command != "preflight" and not args.dedicated_instance:
        parser.error("Live measurements require --dedicated-instance; never use production.")
    if args.delta and args.command != "pilot":
        parser.error("--delta is supported only by the manual pilot command.")
    if args.command == "tune-static" and args.backend == "llm" and not args.allow_paid_llm:
        parser.error("LLM requests require --allow-paid-llm.")
    if args.command == "tune-static" and args.trials < 8:
        parser.error("Reserve at least eight slots for baseline, startup, search and verification.")
    load_dotenv(args.env_file)
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parent.parent
    app = load_app_config(
        project_root=root, target_path=args.target.resolve(),
        benchmark_path=args.benchmark.resolve(), knobs_path=args.knobs.resolve(),
        dry_run_override=False,
    )
    app.auditor.dbms_only = True
    space = load_space(args.knobs, memory_bytes=app.target.hardware_memory_bytes)
    extra = (yaml.safe_load(args.target.read_text()) or {}).get("admtune", {})
    if args.command != "preflight" and any(
        not extra.get(k) or str(extra[k]).startswith("REPLACE")
        for k in ["hardware_id", "data_id", "schema_id"]
    ):
        parser.error("Set real hardware_id, initial data snapshot ID and schema_id first.")
    executor = AgenticDBExecutor(
        app, space, Objective(), args.output,
        allow_mutations=args.command != "preflight", allow_restart=args.allow_restart,
        max_p95_ms=float(extra.get("max_p95_ms", 1000)),
    )
    result = {}
    started = time.monotonic()
    restore_failed = False
    exit_code = 0
    try:
        preflight = executor.preflight()
        (args.output / "preflight.json").write_text(json.dumps(preflight, indent=2) + "\n")
        if args.command == "preflight":
            result = preflight
        else:
            environment = Environment(
                "mysql", preflight["db_version"], extra["hardware_id"], extra["data_id"],
                fingerprint([extra["schema_id"], preflight["readiness"].get("live_schema_hash")]), fingerprint({
                    "benchmark": asdict(app.benchmark), "constraints_p95_ms": executor.max_p95_ms,
                }), space.key + ":static",
            )
            manifest = {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "simulated": False, "command": args.command, "backend": args.backend,
                "seed": args.seed, "environment": asdict(environment),
                "space": space.describe(), "fixed_resource_context": space.fixed_resources,
                "benchmark": asdict(app.benchmark),
                "initial_preflight": preflight, "data_snapshot_restored_by_harness": False,
                "note": "Static only. Dynamic real offered-load replay is a later deployment gate.",
                **source_provenance(),
            }
            (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
            if args.command == "pilot":
                delta = json.loads(args.delta.read_text()) if args.delta else {}
                candidate = space.merge(preflight["current_config"], delta)
                ledger = Ledger(Limits(trials=1, wall_seconds=args.wall_seconds, recovery_reserve_seconds=180))
                ledger.charge_trial("manual_pilot", executor.estimated_trial_seconds)
                measurement = executor.measure(candidate, [0.0])
                result = {"measurement": asdict(measurement), "budget": ledger.snapshot()}
                exit_code = 0 if measurement.status == "measured" and measurement.constraints_ok else 1
            else:
                result = static_tune(
                    executor, space, environment, args.output, seed=args.seed, trials=args.trials,
                    wall_seconds=args.wall_seconds, backend_kind=args.backend,
                    llm_calls=args.llm_calls, llm_tokens=args.llm_tokens,
                )
    except Exception as exc:
        result = {"error": redact(str(exc)), "simulated": False}
        exit_code = 1
    finally:
        if args.command != "preflight":
            try:
                executor.restore_initial()
                result["initial_configuration_restored"] = executor.initial is not None
            except Exception as exc:
                result["recovery_error"] = redact(str(exc))
                restore_failed = True
        executor.close()
        result["total_wall_seconds_including_preflight_and_restore"] = time.monotonic() - started
        result["backend"] = args.backend if args.command == "tune-static" else "manual_or_preflight"
        (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 2 if restore_failed else exit_code


if __name__ == "__main__":
    raise SystemExit(main())
