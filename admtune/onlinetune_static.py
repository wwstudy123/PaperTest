"""同环境事务OnlineTune静态适配实测，零LLM，沿用四方法共同执行器与预算。"""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket

from dotenv import load_dotenv

from config import load_app_config
from .calibration import hardware_context, process_sample, status_sample
from .composite_static import CompositeExecutor, OBJECTIVE
from .continuous import Journal, atomic_json
from .contracts import Environment, Objective, fingerprint
from .dynamic import instance_lock
from .formal_baselines import POLICY, BaselineRun, FinalLedger
from .budget import Limits
from .live import load_space
from .onlinetune_adapter import OnlineTuneStrategy, ADAPTATION, REVISION
from .paper_pilot import PaperExecutor
from .provenance import source_provenance
from .security import redact
from .static_diagnostic import CONCURRENCY_VARIABLES, DiagnosticConnector, cgroup_sample


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("target", "benchmark", "knobs", "output", "reset-audit", "reference-batch", "upstream"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--dedicated-instance", action="store_true")
    a = p.parse_args()
    if not a.dedicated_instance:
        p.error("Dedicated recoverable experimental instance required")
    root = Path(__file__).resolve().parents[1]
    load_dotenv(root / ".env", override=True)
    reference = json.loads((a.reference_batch / "Default/manifest.json").read_text())
    batch = json.loads((a.reference_batch / "batch.json").read_text())
    if batch["status"] != "completed":
        p.error("Completed reference four-method batch required")
    app = load_app_config(project_root=root, target_path=a.target, benchmark_path=a.benchmark,
                          knobs_path=a.knobs, dry_run_override=False)
    space = load_space(a.knobs, memory_bytes=app.target.hardware_memory_bytes)
    if (asdict(app.benchmark) != reference["benchmark"] or space.describe() != reference["space"]
            or reference["comparison_protocol"] != POLICY):
        p.error("Reference workload, catalog or resource policy differs")
    if reference["environment"]["hardware_id"] != socket.gethostname():
        p.error("Reference four methods were not measured on this instance")
    lock = instance_lock(app.target.mysql.config_path)
    a.output.mkdir(mode=0o700, parents=True, exist_ok=False)
    connector = DiagnosticConnector(app.target.ssh, config_path=app.target.mysql.config_path,
                                    dbms="mysql", dry_run=False)
    raw = PaperExecutor(app, space, Objective(), a.output / "runtime",
                        allow_mutations=True, local=True, connector=connector)
    executor = CompositeExecutor(raw)
    journal = ledger = None
    result = {"status": "failed", "method": "OnlineTune"}
    limits = Limits(trials=8, wall_seconds=1500, llm_calls=20, llm_tokens=64000,
                    recovery_reserve_seconds=60)
    try:
        if connector.run("pgrep -x sysbench", check=False).ok:
            raise ValueError("Another benchmark active")
        pre = executor.preflight()
        reset = json.loads(a.reset_audit.read_text())
        if (not reset["completed"] or reset["config_sha256"] != pre["config_sha256"]
                or reset["logical_checksums"] != reference["reset"]["logical_checksums"]
                or pre["current_config"] != reference["preflight"]["current_config"]):
            raise ValueError("Initial configuration or data differs from four-method reference")
        pid = int(Path(app.target.mysql.pid_file).read_text())
        if Path(f"/proc/{pid}/cgroup").read_text() != Path("/proc/self/cgroup").read_text():
            raise ValueError("Unexpected MySQL resource group")
        connector.capture = lambda: {"at": __import__("time").monotonic(), "cgroup": cgroup_sample(),
            "process": process_sample(pid), "mysql": status_sample(raw.db),
            "concurrency_variables": raw.db.show_variables(list(CONCURRENCY_VARIABLES))}
        env = Environment("mysql", pre["db_version"], socket.gethostname(), "formal-workset-v2",
                          pre["readiness"]["live_schema_hash"], fingerprint({
                              "common_policy": POLICY, "method": "OnlineTune", "repeat": batch["repeat"],
                              "space": space.describe(), "benchmark": asdict(app.benchmark),
                              "initial": pre["current_config"], "logical_checksums": reset["logical_checksums"]}))
        journal = Journal(a.output / "records.sqlite3")
        ledger = FinalLedger(journal, limits, executor.estimated_trial_seconds)
        strategy = OnlineTuneStrategy(space, a.upstream, a.output, journal.store.event)
        atomic_json(a.output / "manifest.json", {
            "method": "OnlineTune", "display_method": "OnlineTune (static adapter)",
            "objective": OBJECTIVE, "comparison_protocol": POLICY, "limits": asdict(limits),
            "repeat": batch["repeat"], "simulated": False, "provider_seed_controlled": False,
            "benchmark": asdict(app.benchmark), "space": space.describe(), "preflight": pre,
            "reset": reset, "environment": asdict(env), "hardware": hardware_context(),
            "sources": {"OnlineTune": {"repository": "DBoptimizer/OnlineTune", "commit": REVISION}},
            "adapter_policy": ADAPTATION, "upstream_patch_manifest": strategy.source_report,
            "model": None, "imported_historical_observations": 0,
            "reference_batch": str(a.reference_batch), **source_provenance()})
        result = BaselineRun("OnlineTune", journal, executor, space, env, a.output, ledger, strategy).run()
        configs = [t.config for t in journal.store.trials()]
        result.update(adapter_policy=ADAPTATION, llm_calls=0,
                      unique_measured_configurations=len({fingerprint(c) for c in configs}),
                      upstream_model_state=strategy.snapshot())
    except Exception as exc:
        result.update(status="failed", error=redact(str(exc)))
    finally:
        try:
            executor.restore_initial()
            result["initial_configuration_restored"] = raw.initial is not None
        except Exception as exc:
            result.update(status="recovery_failed", recovery_error=redact(str(exc)))
        if ledger:
            ledger.persist()
            result["budget"] = ledger.snapshot()
            if result["budget"]["elapsed_seconds"] > limits.wall_seconds:
                result["status"] = "wall_budget_exceeded"
        if journal:
            journal.db.execute("PRAGMA wal_checkpoint(FULL)")
            journal.store.close()
        executor.close()
        lock.close()
        result["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        atomic_json(a.output / "summary.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "completed" and result.get("initial_configuration_restored") else 1


if __name__ == "__main__":
    raise SystemExit(main())
