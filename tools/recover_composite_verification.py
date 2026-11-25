"""仅恢复中断的最终复测：保留坏槽位、原费用和预算，不恢复候选搜索。

部署在冻结工作树之外运行；必须先保存中断现场，并验证/启动同一专用实例。
"""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--execute", action="store_true")
    a = parser.parse_args()
    root = a.root.resolve(); p = root/"runs"/a.run_id
    sys.path.insert(0, str(root))
    from dotenv import load_dotenv
    from config import load_app_config
    from llm_client import LLMClient
    from admtune.agents import AgenticDBBackend
    from admtune.calibration import hardware_context, process_sample, status_sample
    from admtune.composite_static import CompositeExecutor, CompositeLedger, CompositeRun, OBJECTIVE
    from admtune.continuous import Journal, atomic_json, code_hash
    from admtune.contracts import Environment, Trial, Objective
    from admtune.dynamic import instance_lock
    from admtune.evidence_wire import EvidenceWireBackend
    from admtune.live import load_space
    from admtune.paper_pilot import PaperExecutor
    from admtune.security import redact
    from admtune.static_diagnostic import CONCURRENCY_VARIABLES, DiagnosticConnector, cgroup_sample
    from admtune.budget import Limits

    assert re.fullmatch(r"composite-pair-(legacy|repeat)-\d{3}", a.run_id)
    m = json.loads((p/"manifest.json").read_text())
    assert code_hash(root) == m["code_sha256"], "Frozen experiment code changed"
    assert not (p/"summary.json").exists(), "Already finalized; do not repeat"
    db = sqlite3.connect(f"file:{p}/records.sqlite3?mode=ro", uri=True)
    old = {k: json.loads(v) for k, v in db.execute("SELECT key,payload FROM runtime")}
    pending = [(n, json.loads(s)) for n, s in db.execute(
        "SELECT seq,submitted FROM attempts WHERE result IS NULL ORDER BY seq")]
    count = db.execute("SELECT COUNT(*) FROM attempts").fetchone()[0]; db.close()
    assert len(pending) == 1 and pending[0][0] == count and pending[0][1]["kind"] == "verification"
    assert old["ledger"]["limits"] == m["limits"] and count < m["limits"]["trials"]
    traces = [json.loads(f.read_text()) for f in sorted((p/"llm_calls").glob("*.json"))]
    assert len(traces) == old["ledger"]["llm_calls"]
    assert sum(t["usage"]["total_tokens"] for t in traces) == old["ledger"]["llm_tokens_or_reserved"]
    print(json.dumps({"pending_slot": count, "old_calls": len(traces),
                      "action": "mark_pending_unknown; one new verification; final reflection only"}), flush=True)
    if not a.execute:
        return 0
    assert (p/"recovery/pre-interruption.tar.gz").is_file(), "Archive scene before any journal mutation"
    snapshot = p/"recovery/pre-interruption.tar.gz"
    snapshot_sha = hashlib.sha256(snapshot.read_bytes()).hexdigest()
    assert snapshot_sha == json.loads(snapshot.with_suffix(".inventory.json").read_text())["archive_sha256"]
    load_dotenv(root/".env", override=True)
    assert os.getenv("OPENAI_MODEL") == m["model"]
    app = load_app_config(project_root=root, target_path=root/"configs/local/workset-v2-target.yaml",
                         benchmark_path=root/"configs/local/timed-readonly-100min.yaml",
                         knobs_path=root/"configs/local/concurrency-knobs.yaml", dry_run_override=False)
    assert asdict(app.benchmark) == m["benchmark"] and hardware_context() == m["hardware"]
    lock = instance_lock(app.target.mysql.config_path)
    connector = DiagnosticConnector(app.target.ssh, config_path=app.target.mysql.config_path,
                                    dbms="mysql", dry_run=False)
    raw = PaperExecutor(app, load_space(root/"configs/local/concurrency-knobs.yaml",
                        memory_bytes=app.target.hardware_memory_bytes), Objective(),
                        p/"runtime-recovery", allow_mutations=True, local=True, connector=connector)
    executor = CompositeExecutor(raw)
    j = client = ledger = None
    result = {"status": "recovery_failed", "objective": OBJECTIVE}
    try:
        pre = executor.preflight()
        for key in ("config_sha256", "current_config", "db_version", "invariants", "space_key"):
            assert pre[key] == m["preflight"][key], f"Recovery preflight mismatch: {key}"
        pid = int(Path(app.target.mysql.pid_file).read_text())
        assert Path(f"/proc/{pid}/cgroup").read_text() == Path("/proc/self/cgroup").read_text()
        connector.capture = lambda: {"at": time.monotonic(), "cgroup": cgroup_sample(),
            "process": process_sample(pid), "mysql": status_sample(raw.db),
            "concurrency_variables": raw.db.show_variables(list(CONCURRENCY_VARIABLES))}
        j = Journal(p/"records.sqlite3")
        ledger = CompositeLedger(j, Limits(**m["limits"]), executor.estimated_trial_seconds)
        penalty = executor.estimated_trial_seconds
        # 未知的中断执行时长按完整试验估计保守占用；不是伪称实测墙钟。
        ledger.base_elapsed += penalty
        seq, submitted = pending[0]
        unknown = Trial(submitted["trial_id"], "static-uniform", Environment(**m["environment"]).key,
            OBJECTIVE, submitted["config"], [0.], "interrupted_unknown_result",
            prediction=submitted.get("prediction"), metadata={
                "seq": seq, "kind": "verification", "simulated": False, "accepted": False,
                "raw_tps": None, "p95_latency_ms": None,
                "model_snapshot": old["state"].get("gp_file"),
                "active_elapsed_seconds": old["ledger"]["elapsed_seconds"]+penalty,
                "reason": "No committed result after interruption; never reconstruct a measured value"})
        j.finish(submitted, unknown, old["state"]); ledger.persist()
        recovery = {"resumed_at_utc": datetime.now(timezone.utc).isoformat(),
            "interrupted_slot": seq, "known_prior_active_seconds": old["ledger"]["elapsed_seconds"],
            "estimated_interruption_seconds_charged": penalty,
            "estimated_charge_is_not_measured_wall_time": True,
            "strict_uninterrupted_pair": False, "search_resumed": False,
            "recovered_final_used_for_training": False,
            "pre_interruption_archive_sha256": snapshot_sha,
            "helper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        atomic_json(p/"recovery/recovery.json", recovery)
        client = LLMClient.from_env(dry_run=False); client.allow_fallback = False
        client.usage_records = [t["usage"] for t in traces]
        backend = EvidenceWireBackend(AgenticDBBackend(client, ledger, trace_dir=p/"llm_calls"),
                                      audit_dir=p/"evidence-recovery")
        runner = CompositeRun(j, executor, raw.space, Environment(**m["environment"]), p, backend, ledger,
                             gp_policy=m["policy"]["gp_policy"],
                             initial_noise_repeats=m["policy"]["initial_noise_repeats"])
        runner.finalizing = ledger.finalizing = True
        final = runner.step()
        # 重启后的最终复测只作独立恢复观测，不更新中断前模型或搜索状态。
        ledger.final_reflection = True
        reflected = runner.reflector.trigger(["stage_end", "interruption_recovery"], force=True)
        j.put("state", runner.save_state()); ledger.persist(); runner.export()
        gp_rejections = j.db.execute("SELECT COUNT(*) FROM events WHERE kind='prescreen_refused'").fetchone()[0]
        result = {"status": "completed_with_recovery", "objective": OBJECTIVE,
            "label_precision_decimals": 6, "gp_policy": runner.gp_policy,
            "initial_noise_repeats": runner.initial_noise_repeats,
            "repeat_noise_audit": runner.repeat_gp.audit if runner.repeat_gp else None,
            "stop_reason": "Original search stopped for reflection reserve; interrupted verification recovered",
            "completed_trials": j.count, "baseline_value": runner.state["baseline"],
            "search_value": runner.state["best_value"], "final_verification_value": final.value,
            "incumbent": runner.state["incumbent"], "gp_training_count": len(runner.state["gp"]["training_ids"]),
            "prospective_predictions": sum(t.prediction is not None for t in j.store.trials()),
            "gp_selected_candidates": sum(t.metadata["kind"] == "candidate" for t in j.store.trials()),
            "gp_rejection_decisions": gp_rejections, "refresh_count": runner.state["refresh_count"],
            "final_reflection_committed": reflected, "reflection_error": None,
            "closed_loop_verified": bool(final.trainable and final.constraints_ok and reflected and gp_rejections),
            "recovery": recovery}
    except Exception as exc:
        result["error"] = redact(str(exc))
    finally:
        try:
            executor.restore_initial()
            result["initial_configuration_restored"] = raw.initial is not None
        except Exception as exc:
            result.update(status="recovery_failed", recovery_error=redact(str(exc)))
        if client:
            atomic_json(p/"llm_usage.json", client.usage_summary())
        if ledger:
            ledger.persist(); result["budget"] = ledger.snapshot()
        if j:
            j.db.execute("PRAGMA wal_checkpoint(FULL)"); j.store.close()
        executor.close(); lock.close()
        result["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        atomic_json(p/"summary.json", result)
        atomic_json(p/"checksums.json", {
            str(f.relative_to(p)): hashlib.sha256(f.read_bytes()).hexdigest()
            for f in p.rglob("*") if f.is_file() and f.suffix in {".json", ".jsonl", ".csv", ".log"}
            and f.name != "checksums.json"})
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("closed_loop_verified") and result.get("initial_configuration_restored") else 1


if __name__ == "__main__":
    raise SystemExit(main())
