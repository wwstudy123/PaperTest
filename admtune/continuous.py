"""可持久化到第100次的真实静态调优；检查点不是另起一轮。"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import socket
import uuid

from dotenv import load_dotenv
import yaml

from config import load_app_config
from llm_client import LLMClient
from .agents import AgenticDBBackend, Team
from .budget import Ledger, Limits
from .calibration import hardware_context
from .contracts import Environment, Objective, Trial, fingerprint
from .discrimination import evaluate, plan as discrimination_plan
from .dynamic import instance_lock
from .evidence_wire import EvidenceWireBackend
from .execution import Measurement
from .real_execution import Snapshot
from .live import load_space
from .memory import Store
from .paper_pilot import PaperExecutor
from .provenance import source_provenance
from .reflection import Reflector
from .replay_report import write_csv
from .security import redact
from .surrogate import GaussianProcess


POLICY = {"max_trials": 100, "llm_calls": 40, "llm_tokens": 160000,
          "wall_seconds": 21600, "bootstrap": 6, "batch_size": 8, "beta": 2,
          "tolerated_drop": .1, "min_gain": .01, "reference_every": 10,
          "audit_every": 5, "verification_at": [50, 100], "protocol_version": 1}


def atomic_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w") as f:
        os.chmod(temp, 0o600)
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def code_hash(root):
    paths = list(root.glob("*.py"))
    for folder in ("admtune", "tools", "connectors", "profiles", "utils", "validators", "memory"):
        paths.extend((root / folder).rglob("*.py"))
    h = hashlib.sha256()
    for p in sorted(paths):
        h.update(str(p.relative_to(root)).encode() + b"\0" + p.read_bytes() + b"\0")
    return h.hexdigest()


def tuple_state(value):
    return tuple(tuple_state(v) for v in value) if isinstance(value, list) else value


class Journal:
    def __init__(self, path):
        self.store = Store(path)
        path.chmod(0o600)
        self.db = self.store.db
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS runtime (key TEXT PRIMARY KEY, payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS attempts (
              seq INTEGER PRIMARY KEY, trial_id TEXT UNIQUE NOT NULL,
              submitted TEXT NOT NULL, result TEXT);
        """)

    def get(self, key, default=None):
        row = self.db.execute("SELECT payload FROM runtime WHERE key=?", (key,)).fetchone()
        return default if row is None else json.loads(row[0])

    def put(self, key, value):
        with self.db:
            self._put(key, value)

    def _put(self, key, value):
        self.db.execute("INSERT OR REPLACE INTO runtime VALUES (?,?)",
                        (key, json.dumps(value, allow_nan=False)))

    @property
    def count(self):
        return self.db.execute("SELECT COUNT(*) FROM attempts").fetchone()[0]

    def reserve(self, payload, state):
        seq = self.count + 1
        if seq > POLICY["max_trials"]:
            raise ValueError("Global trial limit reached")
        if self.db.execute("SELECT COUNT(*) FROM attempts WHERE result IS NULL").fetchone()[0]:
            raise ValueError("Unresolved previous attempt")
        payload = {**payload, "seq": seq, "trial_id": uuid.uuid4().hex,
                   "submitted_at_utc": datetime.now(timezone.utc).isoformat()}
        with self.db:
            self.db.execute("INSERT INTO attempts VALUES (?,?,?,NULL)",
                            (seq, payload["trial_id"], json.dumps(payload, allow_nan=False)))
            self._put("state", state)
        return payload

    def finish(self, submitted, trial, state):
        with self.db:
            changed = self.db.execute("UPDATE attempts SET result=? WHERE seq=? AND result IS NULL",
                                      (json.dumps(trial.to_dict(), allow_nan=False), submitted["seq"])).rowcount
            if changed != 1:
                raise ValueError("Attempt already completed or missing")
            self.db.execute("INSERT INTO trials(trial_id,payload) VALUES (?,?)",
                            (trial.trial_id, json.dumps(trial.to_dict(), allow_nan=False)))
            self._put("state", state)

    def recover_pending(self, environment_key):
        pending = self.db.execute("SELECT submitted FROM attempts WHERE result IS NULL ORDER BY seq").fetchall()
        for (raw,) in pending:
            s = json.loads(raw)
            t = Trial(s["trial_id"], "static-uniform", environment_key, "tps:maximize",
                      s["config"], [0.0], "interrupted_unknown_result",
                      prediction=s.get("prediction"),
                      metadata={"seq": s["seq"], "kind": s["kind"], "simulated": False,
                                "reason": "Result missing after interruption; never rerun same trial number"})
            self.finish(s, t, self.get("state"))
        return len(pending)

    def audit(self):
        rows = self.db.execute("SELECT seq,trial_id,result FROM attempts ORDER BY seq").fetchall()
        if [r[0] for r in rows] != list(range(1, len(rows) + 1)):
            raise ValueError("Non-contiguous trial journal")
        records = self.store.trials()
        completed = [r for r in rows if r[2] is not None]
        if len(completed) != len(records) or any(
                r[1] != t.trial_id or json.loads(r[2]) != t.to_dict() for r, t in zip(completed, records)):
            raise ValueError("Trial ledger and evidence store disagree")


class DurableLedger(Ledger):
    def __init__(self, journal):
        super().__init__(Limits(trials=100, wall_seconds=21600, llm_calls=40,
                                llm_tokens=160000, recovery_reserve_seconds=180))
        self.journal = journal
        old = journal.get("ledger", {})
        self.base_elapsed = old.get("elapsed_seconds", 0)
        self.trials = journal.count
        self.llm_calls = old.get("llm_calls", 0)
        self.llm_tokens = old.get("llm_tokens_or_reserved", 0)

    def remaining_seconds(self):
        return max(0, super().remaining_seconds() - self.base_elapsed)

    def snapshot(self):
        return {**super().snapshot(), "elapsed_seconds": super().snapshot()["elapsed_seconds"] + self.base_elapsed}

    def persist(self):
        self.journal.put("ledger", self.snapshot())

    def reserve_llm(self, max_tokens):
        super().reserve_llm(max_tokens)
        self.persist()  # 在发送请求之前保存预留；未知用量不退还。

    def settle_llm(self, reserved, actual):
        try:
            super().settle_llm(reserved, actual)
        finally:
            self.persist()


class ContinuousRun:
    def __init__(self, journal, executor, space, environment, output, backend, ledger, seed=42):
        self.journal, self.executor, self.space = journal, executor, space
        self.environment, self.output, self.ledger = environment, output, ledger
        self.store = journal.store
        self.team = Team(space, backend, batch_size=8, max_selected=4)
        self.reflector = Reflector(self.store, backend, min_batch=3, max_batch=10, old_summary_limit=0)
        self.rng = random.Random(seed)
        self.state = journal.get("state") or {
            "initial": executor.readback(), "incumbent": executor.readback(),
            "baseline": None, "best_value": None, "pool": [], "next_proposal_at": 2,
            "selected": [], "ranges": {}, "generation": 0, "feedback": {}, "gp": None}
        if self.state.get("rng"):
            self.rng.setstate(tuple_state(self.state["rng"]))
        self.team.selected, self.team.ranges = self.state["selected"], self.state["ranges"]
        self.team.generation, self.team.feedback = self.state["generation"], self.state["feedback"]
        self.gp = (GaussianProcess.from_snapshot(self.state["gp"]) if self.state.get("gp")
                   else GaussianProcess(config_dimensions=len(space.encode(self.state["initial"])), seed=seed))

    def save_state(self):
        self.state.update(selected=self.team.selected, ranges=self.team.ranges,
                          generation=self.team.generation, feedback=self.team.feedback,
                          rng=self.rng.getstate())
        return self.state

    def fit(self):
        history = [t for t in self.store.trials() if t.trainable and t.constraints_ok]
        ids = [t.trial_id for t in history]
        if len(history) < 6 or (self.state.get("gp") and self.state["gp"]["training_ids"] == ids):
            return
        self.gp.fit([self.space.encode(t.config) + t.context for t in history],
                    [math.log(t.value) for t in history], optimize=self.gp.version % 10 == 0)
        payload = self.gp.snapshot(ids)
        folder = self.output / "gp_snapshots"
        folder.mkdir(exist_ok=True)
        path = folder / f"gp-{self.gp.version:04d}-{fingerprint(payload)[:12]}.json"
        if path.exists():
            if json.loads(path.read_text()) != payload:
                raise ValueError("Model snapshot hash collision")
        else:
            atomic_json(path, payload)
        self.state["gp"] = payload
        self.state["gp_file"] = str(path.relative_to(self.output))
        self.journal.put("state", self.save_state())

    def experiences(self):
        trials = [t for t in self.store.trials() if t.trainable and t.constraints_ok][-3:]
        numerical = [{"type": "measurement", "trial_id": t.trial_id, "config": t.config, "value": t.value}
                     for t in trials]
        refs = self.store.reflections()[-1:]
        return numerical + [{"type": "reflection", "hypothesis_previews": [
                             {"text": str(h)[:160], "truncated": len(str(h)) > 160} for h in r["hypotheses"][:2]],
                             "conditions": r["conditions"], "evidence_ids": r["evidence_ids"]} for r in refs]

    def replenish(self, seq):
        if self.state["pool"] or seq < self.state["next_proposal_at"]:
            return
        diagnostics = {**self.executor.diagnose(), "objective_key": "tps:maximize",
                       "protocol": "native Sysbench read-only, uniform random distribution, fixed offered concurrency"}
        if not self.team.selected or not self.team.ranges:
            self.team.initialize([0.0], self.experiences(), diagnostics=diagnostics)
        self.team.diagnostics = diagnostics
        self.journal.put("state", self.save_state())
        candidates = self.team.recommend(self.state["incumbent"], "static-uniform", [0.0], self.experiences())
        seen = {fingerprint(json.loads(r[0])["config"]) for r in
                self.journal.db.execute("SELECT submitted FROM attempts")}
        self.state["pool"] = [c.config for c in candidates if c.key not in seen]
        self.state["next_proposal_at"] = seq + 8
        self.journal.put("state", self.save_state())

    def choose(self, seq):
        if seq == 1:
            return "baseline", self.state["initial"], {"rule": "initial_baseline"}
        if self.state["baseline"] is None:
            raise ValueError("Initial baseline is invalid; repair protocol before continuing")
        if seq in POLICY["verification_at"]:
            return "verification", self.state["incumbent"], {"rule": "fixed_checkpoint_verification"}
        if seq % 10 == 0:
            return "reference", self.state["initial"], {"rule": "predeclared_initial_reference"}
        self.replenish(seq)
        pool = self.state["pool"]
        if seq % 5 == 0 and pool:
            i = self.rng.randrange(len(pool))
            return "audit", pool.pop(i), {"rule": "predeclared_prediction_audit_not_GP_admission"}
        if pool and not self.gp.fitted:
            return "startup", pool.pop(0), {"rule": "bootstrap"}
        if pool:
            predictions = self.gp.predict([self.space.encode(c) + [0.0] for c in pool])
            floor = math.log(self.state["baseline"]) + math.log1p(-.1)
            gain = math.log(self.state["best_value"]) + math.log1p(.01)
            eligible = [i for i, p in enumerate(predictions) if p.lower(2) >= floor and p.upper(2) > gain]
            decision = {"rule": "GP_LCB_then_UCB", "config_ids": [fingerprint(c) for c in pool],
                        "predictions": [asdict(p) for p in predictions], "eligible": eligible}
            if eligible:
                i = max(eligible, key=lambda i: predictions[i].upper(2))
                return "candidate", pool.pop(i), decision
            self.store.event("prescreen_refused", {"next_seq": seq, **decision})
        return "incumbent_repeat", self.state["incumbent"], {
            "rule": "no_admissible_new_candidate_repeat_for_noise_not_search_gain"}

    def step(self):
        seq = self.journal.count + 1
        self.ledger.ensure_time(self.executor.estimated_trial_seconds)
        self.fit()
        kind, config, decision = self.choose(seq)
        config = self.space.merge(config, {})
        prediction = asdict(self.gp.predict([self.space.encode(config) + [0.0]])[0]) if self.gp.fitted else None
        submitted = self.journal.reserve({"kind": kind, "config": config,
                                         "prediction": prediction, "decision": decision}, self.save_state())
        self.ledger.trials = self.journal.count
        self.ledger.persist()
        folder = self.output / "trials" / f"{seq:06d}"
        folder.mkdir(mode=0o700, parents=True, exist_ok=False)
        self.executor.run_dir = folder
        resize_offset = len(getattr(self.executor, "resize_checks", []))
        try:
            measured = self.executor.measure(config, [0.0])
        except Exception as exc:
            measured = Measurement(None, "execution_failed", False, False, False, {"reason": redact(str(exc))})
        # 每次真实试验后都恢复共同初始状态；推荐父配置仍保存在算法状态中。
        self.executor.restore_initial()
        measured.metadata["resize_checks"] = getattr(self.executor, "resize_checks", [])[resize_offset:]
        if measured.metadata.get("benchmark_log"):
            measured.metadata["benchmark_log"] = str(folder.relative_to(self.output) / measured.metadata["benchmark_log"])
        record = Trial(submitted["trial_id"], "static-uniform", self.environment.key, "tps:maximize",
                       config, [0.0], measured.status, measured.value, measured.complete,
                       measured.effective, measured.constraints_ok, prediction,
                       {**measured.metadata, "seq": seq, "kind": kind, "decision": decision,
                        "simulated": self.executor.simulated, "accepted": False,
                        "model_snapshot": self.state.get("gp_file"),
                        "active_elapsed_seconds": self.ledger.snapshot()["elapsed_seconds"]})
        if record.trainable and record.constraints_ok:
            if kind == "baseline":
                self.state.update(baseline=record.value, best_value=record.value, incumbent=config)
            elif kind in {"startup", "candidate", "audit"} and record.value > self.state["best_value"] * 1.01:
                self.state.update(best_value=record.value, incumbent=config)
                record.metadata["accepted"] = True
        self.team.feedback = {"trial_id": record.trial_id, "status": record.status,
                              "value": record.value, "accepted": record.metadata["accepted"]}
        self.journal.finish(submitted, record, self.save_state())
        self.ledger.persist()
        if kind == "baseline" and (not record.trainable or not record.constraints_ok):
            raise RuntimeError("Invalid initial baseline; cannot proceed")
        recent = self.store.trials()[-3:]
        if len(recent) == 3 and all(not (t.trainable and t.constraints_ok) for t in recent):
            raise RuntimeError("Three consecutive invalid attempts; stop for diagnosis, not budget filling")
        if seq % 10 == 0:
            try:
                self.reflector.trigger(["checkpoint"], force=True)
            except (RuntimeError, ValueError) as exc:
                self.store.event("reflection_deferred", {"at_seq": seq, "reason": redact(str(exc))})
        self.export()
        return record

    def export(self):
        self.journal.audit()
        records = self.store.trials()
        best, rows = None, []
        for t in records:
            kind = t.metadata["kind"]
            valid = t.trainable and t.constraints_ok
            if valid and kind in {"baseline", "startup", "candidate", "audit"} and (best is None or t.value > best.value):
                best = t
            rows.append({"trial": t.metadata["seq"], "kind": kind, "status": t.status,
                         "valid": valid, "tps": t.value, "p95_ms": t.metadata.get("p95_latency_ms"),
                         "best_found_tps": None if best is None else best.value,
                         "best_found_p95_ms": None if best is None else best.metadata.get("p95_latency_ms"),
                         "active_elapsed_seconds": t.metadata.get("active_elapsed_seconds"),
                         "accepted": t.metadata.get("accepted", False), "benchmark_log": t.metadata.get("benchmark_log"),
                         "config_json": json.dumps(t.config, sort_keys=True)})
        atomic_json(self.output / "progress.json", {
            "completed_attempts": len(records), "next_trial": self.journal.count + 1,
            "max_trials": 100, "trainable_observations": sum(t.trainable and t.constraints_ok for t in records),
            "incumbent": self.state["incumbent"], "incumbent_search_value": self.state["best_value"],
            "budget": self.ledger.snapshot(), "counts_by_kind": {
                k: sum(t.metadata["kind"] == k for t in records) for k in sorted({t.metadata["kind"] for t in records})}})
        write_csv(self.output / "curve.csv", rows, list(rows[0]) if rows else ["trial"])
        (self.output / "trials.jsonl").write_text("".join(json.dumps(t.to_dict(), allow_nan=False) + "\n" for t in records))
        (self.output / "events.jsonl").write_text("".join(
            json.dumps({"seq": n, "kind": kind, "payload": json.loads(payload)}) + "\n"
            for n, kind, payload in self.journal.db.execute("SELECT seq,kind,payload FROM events ORDER BY seq")))
        atomic_json(self.output / "reflections.json", self.store.reflections())

    def run_until(self, until):
        if not self.journal.count < until <= 100:
            raise ValueError("Cumulative endpoint must exceed current count and be <=100")
        while self.journal.count < until:
            self.step()
        self.fit()
        self.journal.put("state", self.save_state())
        self.ledger.persist()
        self.export()
        return {"status": "checkpoint" if until < 100 else "completed_100",
                "completed_until": until, "next_trial": until + 1 if until < 100 else None,
                "can_resume": until < 100, "budget": self.ledger.snapshot(),
                "incumbent": self.state["incumbent"], "search_value": self.state["best_value"]}


def verify_calibration(path):
    manifest = json.loads((path / "manifest.json").read_text())
    summary = json.loads((path / "summary.json").read_text())
    hashes = json.loads((path / "checksums.json").read_text())
    if not {"manifest.json", "summary.json", "trials.jsonl"} <= hashes.keys():
        raise ValueError("Calibration evidence incomplete")
    for name, digest in hashes.items():
        if Path(name).name != name or hashlib.sha256((path / name).read_bytes()).hexdigest() != digest:
            raise ValueError("Calibration checksum mismatch")
    records = [json.loads(line) for line in (path / "trials.jsonl").read_text().splitlines()]
    if (manifest.get("git_dirty") is not False or manifest.get("mode") != "zero_llm_discrimination"
            or len(records) != 8):
        raise ValueError("Complete clean-source real calibration required")
    initial = manifest["preflight"]["current_config"]
    for i, (record, (distribution, candidate)) in enumerate(zip(records, discrimination_plan()), 1):
        expected = {**initial, "innodb_buffer_pool_size": (2 if candidate else 1) * 1024**3}
        if (record["trial"] != i or record["distribution"] != distribution
                or record["candidate"] != candidate or record["config"] != expected
                or record["measurement"]["metadata"].get("simulated") is not False):
            raise ValueError("Calibration plan or measurement provenance mismatch")
    gate = evaluate(records)
    if not summary.get("completed") or not summary.get("initial_configuration_restored") or not gate["passed"]:
        raise ValueError("Calibration signal gate not passed; paid run prohibited")
    return {"manifest": manifest, "gate": gate, "source_checksums": json.loads((path / "checksums.json").read_text())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("target", "benchmark", "knobs", "output", "calibration"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--restore-audit", type=Path)
    parser.add_argument("--until-trial", type=int, default=50)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--dedicated-instance", action="store_true")
    parser.add_argument("--allow-paid-llm", action="store_true")
    args = parser.parse_args()
    if not args.dedicated_instance or (not args.check_only and not args.allow_paid_llm):
        parser.error("Dedicated instance and explicit paid authorization required")
    if not 1 <= args.until_trial <= 100 or (args.check_only and not args.resume):
        parser.error("Endpoint must be in 1..100; check-only requires resume")
    load_dotenv(override=True)
    if not args.check_only and not all(os.getenv(k) for k in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL")):
        parser.error("Explicit key, endpoint and model required; no fallback")
    root = Path(__file__).resolve().parent.parent
    app = load_app_config(project_root=root, target_path=args.target, benchmark_path=args.benchmark,
                          knobs_path=args.knobs, dry_run_override=False)
    if (app.target.dbms != "mysql" or app.target.mysql.host not in {"127.0.0.1", "localhost"}
            or app.benchmark.workload.rand_type != "uniform"
            or app.benchmark.workload.script_path != "/usr/share/sysbench/oltp_read_only.lua"):
        parser.error("Frozen local uniform read-only protocol required")
    proof = verify_calibration(args.calibration)
    space = load_space(args.knobs, memory_bytes=app.target.hardware_memory_bytes)
    lock = instance_lock(app.target.mysql.config_path)
    if args.resume:
        manifest = json.loads((args.output / "manifest.json").read_text())
    else:
        args.output.mkdir(parents=True, mode=0o700, exist_ok=False)
        manifest = None
    executor = PaperExecutor(app, space, Objective(), args.output / "runtime", allow_mutations=True, local=True)
    journal = None
    client = None
    session_id = uuid.uuid4().hex
    result = {"status": "failed"}
    try:
        if args.resume:
            journal = Journal(args.output / "records.sqlite3")
            pending = journal.db.execute("SELECT COUNT(*) FROM attempts WHERE result IS NULL").fetchone()[0]
            if pending and not args.check_only:
                if executor.connector.run("pgrep -x sysbench", check=False).ok:
                    raise ValueError("A Sysbench process remains; wait before interruption recovery")
                original = (args.output / "initial.my.cnf").read_text()
                if hashlib.sha256(original.encode()).hexdigest() != manifest["preflight"]["config_sha256"]:
                    raise ValueError("Initial configuration backup checksum mismatch")
                executor.initial = Snapshot(manifest["preflight"]["current_config"], original,
                                            manifest["preflight"]["invariants"])
                executor.restore_initial()
        preflight = executor.preflight()
        extra = yaml.safe_load(args.target.read_text())["admtune"]
        calibrated_benchmark = json.loads(json.dumps(proof["manifest"]["benchmark"]))
        expected = asdict(app.benchmark)
        calibrated_benchmark["workload"]["rand_type"] = "uniform"
        if (calibrated_benchmark != expected
                or preflight["config_sha256"] != proof["manifest"]["preflight"]["config_sha256"]
                or preflight["space_key"] != proof["manifest"]["preflight"]["space_key"]
                or preflight["readiness"]["live_schema_hash"] != proof["manifest"]["preflight"]["readiness"]["live_schema_hash"]):
            raise ValueError("Benchmark/configuration/schema incompatible with calibration")
        environment = Environment("mysql", preflight["db_version"], socket.gethostname(), extra["data_id"],
                                  preflight["readiness"]["live_schema_hash"],
                                  fingerprint({"benchmark": expected, "space": space.describe(),
                                               "initial": preflight["current_config"], "policy": POLICY}))
        frozen = {"environment": asdict(environment), "code_sha256": code_hash(root),
                  "dependencies": source_provenance()["dependencies"], "hardware": hardware_context(),
                  "fixed_resources": space.fixed_resources, "memory_budget_bytes": space.memory_bytes,
                  "invariants": preflight["invariants"],
                  "llm_identity": {"model": os.getenv("OPENAI_MODEL"), "base_url": os.getenv("OPENAI_BASE_URL", "").rstrip("/"),
                                   "enable_thinking": os.getenv("OPENAI_ENABLE_THINKING", ""),
                                   "temperature": .2, "max_output_tokens": 1500},
                  "policy": POLICY, "calibration_sha256": fingerprint(proof)}
        if args.resume:
            if manifest["frozen"] != frozen:
                raise ValueError("Continuation refused: changed code, environment, protocol, budget or calibration")
        else:
            if args.restore_audit is None:
                raise ValueError("Initial creation requires a fresh cold-restore audit")
            audit = json.loads(args.restore_audit.read_text())
            if not audit.get("completed") or audit["config_sha256"] != preflight["config_sha256"]:
                raise ValueError("Cold restore audit mismatch")
            manifest = {"method": "ADMTune-static-continuable", "simulated": False, "seed": 42,
                        "created_at_utc": datetime.now(timezone.utc).isoformat(),
                        "frozen": frozen, "benchmark": expected, "space": space.describe(),
                        "preflight": preflight, "restore": audit, **source_provenance()}
            atomic_json(args.output / "manifest.json", manifest)
            private = args.output / "initial.my.cnf"
            private.write_text(executor.initial.file_content)
            private.chmod(0o600)
        journal = journal or Journal(args.output / "records.sqlite3")
        journal.audit()
        if args.check_only:
            pending = journal.db.execute("SELECT COUNT(*) FROM attempts WHERE result IS NULL").fetchone()[0]
            if pending or not journal.count < args.until_trial <= 100:
                raise ValueError("No clean continuable checkpoint for requested endpoint")
            if journal.get("state", {}).get("gp"):
                GaussianProcess.from_snapshot(journal.get("state")["gp"])
            result = {"can_continue": True, "completed_until": journal.count,
                      "next_trial": journal.count + 1, "performed_trials": 0, "performed_llm_calls": 0}
        else:
            with (args.output / "sessions.jsonl").open("a") as f:
                f.write(json.dumps({"event": "started", "session_id": session_id, "at_utc": datetime.now(timezone.utc).isoformat(),
                                    "count_before": journal.count, "until_trial": args.until_trial,
                                    **source_provenance()}) + "\n")
            ledger = DurableLedger(journal)
            interrupted = journal.recover_pending(environment.key)
            if interrupted:
                ledger.base_elapsed += 600 * interrupted  # 未知中断耗时的保守预算，不当作准确测量时间。
                ledger.persist()
            client = LLMClient.from_env(dry_run=False)
            client.allow_fallback = False
            backend = EvidenceWireBackend(AgenticDBBackend(client, ledger, trace_dir=args.output / "llm_calls"),
                                          audit_dir=args.output / "evidence" / session_id)
            runner = ContinuousRun(journal, executor, space, environment, args.output, backend, ledger)
            result = runner.run_until(args.until_trial)
    except Exception as exc:
        result = {"status": "failed", "error": redact(str(exc))}
    finally:
        try:
            if not args.check_only:
                executor.restore_initial()
            elif executor.initial:
                executor._verify(executor.initial.config, executor.initial.invariants)
            result["initial_configuration_restored"] = executor.initial is not None
        except Exception as exc:
            result.update(status="recovery_failed", recovery_error=redact(str(exc)))
        if client is not None:
            with (args.output / "llm_usage_sessions.jsonl").open("a") as f:
                f.write(json.dumps(client.usage_summary(), ensure_ascii=False) + "\n")
        if journal:
            journal.db.execute("PRAGMA wal_checkpoint(FULL)")
            journal.store.close()
        executor.close()
        lock.close()
        if not args.check_only:
            with (args.output / "sessions.jsonl").open("a") as f:
                f.write(json.dumps({"event": "finished", "session_id": session_id, "at_utc": datetime.now(timezone.utc).isoformat(),
                                    "status": result.get("status")}) + "\n")
            atomic_json(args.output / "summary.json", result)
            atomic_json(args.output / "checksums.json", {
                str(p.relative_to(args.output)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in args.output.rglob("*") if p.is_file() and p.suffix in {".json", ".jsonl", ".csv", ".log"}
                and p.name != "checksums.json"})
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("status") in {"checkpoint", "completed_100"} or result.get("can_continue") else 1


if __name__ == "__main__":
    raise SystemExit(main())
