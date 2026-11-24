"""按墙钟预算运行真实静态 ADMTune，复测和恢复预留在预算内。"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import socket

from dotenv import load_dotenv
import yaml

from config import load_app_config
from llm_client import LLMClient
from .agents import AgenticDBBackend
from .budget import BudgetExhausted, Limits
from .calibration import hardware_context
from .continuous import ContinuousRun, DurableLedger, Journal, POLICY, atomic_json, code_hash
from .contracts import Environment, Objective, fingerprint
from .dynamic import instance_lock
from .evidence_wire import EvidenceWireBackend
from .live import load_space
from .paper_pilot import PaperExecutor
from .provenance import source_provenance
from .security import redact


class TimedLedger(DurableLedger):
    def __init__(self, journal, seconds, final_reserve):
        super().__init__(journal, limits=Limits(trials=100, wall_seconds=21600, llm_calls=40,
                                               llm_tokens=160000, recovery_reserve_seconds=60))
        self.session_seconds = seconds
        self.session_started = self.clock()
        self.final_reserve = final_reserve
        self.finalizing = False

    def real_remaining(self):
        return min(super().remaining_seconds(), self.session_seconds - (self.clock() - self.session_started))

    def remaining_seconds(self):
        return max(0, self.real_remaining() - (0 if self.finalizing else self.final_reserve))

    def snapshot(self):
        return {**super().snapshot(), "session_budget_seconds": self.session_seconds,
                "session_elapsed_seconds": self.clock() - self.session_started,
                "reserved_final_seconds": 0 if self.finalizing else self.final_reserve}


class TimedRun(ContinuousRun):
    finalizing = False

    def replenish(self, seq):
        # 最多一次格式纠错，仍使用真实模型并累计收费，不削弱配置检查。
        try:
            return super().replenish(seq)
        except ValueError as exc:
            self.store.event("schema_retry", {"seq": seq, "reason": redact(str(exc)), "max_retry": 1})
            self.team.feedback = {"reason": "Previous structured output failed validation: " + str(exc)}
            self.team.ranges = {}
            self.journal.put("state", self.save_state())
            return super().replenish(seq)

    def choose(self, seq):
        if self.finalizing:
            return "verification", self.state["incumbent"], {"rule": "reserved_end_of_time_budget_verification"}
        return super().choose(seq)

    def run_budget(self, until=100):
        if not self.journal.count < until <= 100:
            raise ValueError("Cumulative endpoint must exceed the recorded count and be <=100")
        stop = "trial_cap"
        while self.journal.count < until - 1:
            try:
                self.ledger.ensure_time(self.executor.estimated_trial_seconds)
                self.step()
            except BudgetExhausted as exc:
                self.store.event("time_admission_stop", {"reason": str(exc)})
                stop = "time_or_api_budget"
                break
        self.finalizing = self.ledger.finalizing = True
        final_value = None
        try:
            self.ledger.ensure_time(self.executor.estimated_trial_seconds)
            if self.journal.count < until and self.state["baseline"] is not None:
                r = self.step()
                if r.trainable and r.constraints_ok:
                    final_value = r.value
        except BudgetExhausted as exc:
            self.store.event("final_verification_not_admitted", {"reason": str(exc)})
        self.fit()
        self.journal.put("state", self.save_state())
        self.ledger.persist()
        self.export()
        return {"status": "time_budget_checkpoint", "stop_reason": stop,
                "completed_until": self.journal.count, "next_trial": self.journal.count + 1,
                "can_resume": self.journal.count < 100,
                "wall_budget_seconds": self.ledger.session_seconds,
                "budget": self.ledger.snapshot(), "incumbent": self.state["incumbent"],
                "baseline_value": self.state["baseline"], "search_value": self.state["best_value"],
                "final_verification_value": final_value, "objective": "tps:maximize"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("target", "benchmark", "knobs", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--restore-audit", type=Path)
    p.add_argument("--minutes", type=float, default=100)
    p.add_argument("--until-trial", type=int, default=100)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--dedicated-instance", action="store_true")
    p.add_argument("--allow-paid-llm", action="store_true")
    a = p.parse_args()
    if not a.dedicated_instance or not a.allow_paid_llm:
        p.error("Explicit dedicated instance and paid authorization required")
    if not 5 <= a.minutes <= 180 or not 2 <= a.until_trial <= 100:
        p.error("Time budget must be 5..180 minutes, cumulative trial cap 2..100")
    load_dotenv(override=True)
    if not all(os.getenv(k) for k in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL")):
        p.error("Explicit key, endpoint and model required")
    root = Path(__file__).resolve().parent.parent
    app = load_app_config(project_root=root, target_path=a.target, benchmark_path=a.benchmark,
                          knobs_path=a.knobs, dry_run_override=False)
    w = app.benchmark.workload
    if (app.target.dbms != "mysql" or app.target.mysql.host not in {"127.0.0.1", "localhost"}
            or w.script_path != "/usr/share/sysbench/oltp_read_only.lua" or w.rand_type != "uniform"
            or w.duration != 60 or w.warmup_time != 15):
        p.error("Frozen local Sysbench uniform read-only 15+60 protocol required")
    lock = instance_lock(app.target.mysql.config_path)
    if not a.resume:
        a.output.mkdir(parents=True, mode=0o700, exist_ok=False)
    space = load_space(a.knobs, memory_bytes=app.target.hardware_memory_bytes)
    executor = PaperExecutor(app, space, Objective(), a.output / "runtime", allow_mutations=True, local=True)
    j = client = ledger = None
    result = {"status": "failed"}
    try:
        preflight = executor.preflight()
        extra = yaml.safe_load(a.target.read_text())["admtune"]
        environment = Environment("mysql", preflight["db_version"], socket.gethostname(), extra["data_id"],
                                  preflight["readiness"]["live_schema_hash"],
                                  fingerprint({"benchmark": asdict(app.benchmark), "space": space.describe(),
                                               "initial": preflight["current_config"], "policy": POLICY}))
        frozen = {"environment": asdict(environment), "code_sha256": code_hash(root),
                  "dependencies": source_provenance()["dependencies"], "hardware": hardware_context(),
                  "policy": POLICY, "benchmark": asdict(app.benchmark),
                  "fixed_resources": space.fixed_resources, "invariants": preflight["invariants"],
                  "model": os.getenv("OPENAI_MODEL"), "base_url": os.getenv("OPENAI_BASE_URL", "").rstrip("/"),
                  "enable_thinking": os.getenv("OPENAI_ENABLE_THINKING", "")}
        if a.resume:
            manifest = json.loads((a.output / "manifest.json").read_text())
            if frozen != manifest["frozen"]:
                raise ValueError("Environment, code or protocol changed; continuation refused")
        else:
            if a.restore_audit is None:
                raise ValueError("Initial run requires a cold restore audit")
            audit = json.loads(a.restore_audit.read_text())
            if not audit.get("completed") or audit["config_sha256"] != preflight["config_sha256"]:
                raise ValueError("Restore audit mismatch")
            manifest = {"method": "ADMTune", "simulated": False, "frozen": frozen,
                        "created_at_utc": datetime.now(timezone.utc).isoformat(),
                        "benchmark": asdict(app.benchmark), "space": space.describe(), "preflight": preflight,
                        "restore": audit, "planned_minutes": a.minutes, **source_provenance()}
            atomic_json(a.output / "manifest.json", manifest)
            backup = a.output / "initial.my.cnf"
            backup.write_text(executor.initial.file_content); backup.chmod(0o600)
        j = Journal(a.output / "records.sqlite3")
        j.audit()
        if j.db.execute("SELECT COUNT(*) FROM attempts WHERE result IS NULL").fetchone()[0]:
            raise ValueError("Unresolved interrupted attempt; explicit recovery audit required")
        ledger = TimedLedger(j, a.minutes * 60, executor.estimated_trial_seconds)
        client = LLMClient.from_env(dry_run=False); client.allow_fallback = False
        backend = EvidenceWireBackend(AgenticDBBackend(client, ledger, trace_dir=a.output / "llm_calls"),
                                      audit_dir=a.output / "evidence" / str(j.count))
        run = TimedRun(j, executor, space, environment, a.output, backend, ledger)
        result = run.run_budget(a.until_trial)
    except Exception as exc:
        result.update(status="failed", error=redact(str(exc)))
    finally:
        try:
            executor.restore_initial()
            result["initial_configuration_restored"] = executor.initial is not None
        except Exception as exc:
            result.update(status="recovery_failed", recovery_error=redact(str(exc)))
        if client is not None:
            with (a.output / "llm_usage_sessions.jsonl").open("a") as f:
                f.write(json.dumps(client.usage_summary(), ensure_ascii=False) + "\n")
        if ledger:
            ledger.persist()
            result["budget"] = ledger.snapshot()
        if j:
            j.db.execute("PRAGMA wal_checkpoint(FULL)"); j.store.close()
        executor.close(); lock.close()
        result["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        atomic_json(a.output / "summary.json", result)
        atomic_json(a.output / "checksums.json", {
            str(f.relative_to(a.output)): hashlib.sha256(f.read_bytes()).hexdigest()
            for f in a.output.rglob("*") if f.is_file() and f.suffix in {".json", ".jsonl", ".csv", ".log"}
            and f.name != "checksums.json"})
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("status") == "time_budget_checkpoint" and result.get("final_verification_value") else 1


if __name__ == "__main__":
    raise SystemExit(main())
