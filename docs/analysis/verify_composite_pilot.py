"""离线核验综合目标真实小预算试验，导出原始指标、预测和决策，不连接API。"""
from dataclasses import asdict
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
import re
import statistics
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from admtune.contracts import fingerprint
from admtune.space import Knob, Space
from admtune.surrogate import GaussianProcess
from admtune.contracts import Trial
from admtune.composite_static import pilot_limits, runtime_policy
from admtune.repeat_aware_gp import repeat_noise, record_fingerprint
from dataclasses import asdict as dataclass_dict


def load(path):
    return json.loads(path.read_text())


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def write_csv(path, rows):
    with path.open("w") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]) if rows else ["trial"],
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="composite-pilot-001")
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not re.fullmatch(r"composite-(pilot-\d{3}|pair-(legacy|repeat)-\d{3})", args.run_id):
        parser.error("run-id必须为composite-pilot-NNN或composite-pair-legacy/repeat-NNN")
    args.archive = args.archive or ROOT / f"runs/archives/{args.run_id}.tar.gz"
    args.output = args.output or ROOT / f"runs/collected-{args.run_id}"
    inventory = load(args.archive.with_suffix(".inventory.json"))
    assert hashlib.sha256(args.archive.read_bytes()).hexdigest() == inventory["archive_sha256"]
    args.output.mkdir(mode=0o700, parents=True, exist_ok=False)
    with tarfile.open(args.archive) as archive:
        members = archive.getmembers()
        assert len(members) == len(inventory["files"])
        assert {m.name for m in members} == set(inventory["files"])
        assert sum(m.size for m in members) < 100_000_000
        for member in members:
            name = Path(member.name)
            assert (member.isfile() and not name.is_absolute() and ".." not in name.parts
                    and name.parts[0] == "runs" and member.size < 20_000_000)
            data = archive.extractfile(member).read()
            assert hashlib.sha256(data).hexdigest() == inventory["files"][member.name]
            out = args.output / name
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(data)
            out.chmod(0o600)
    p = args.output / "runs" / args.run_id
    audit = args.output / "runs" / f"{args.run_id}-audit"
    summary, manifest, post = (load(p / "summary.json"), load(p / "manifest.json"),
                               load(audit / "post_run.json"))
    runtime = load(audit / "runtime.json")
    assert not manifest["simulated"] and not manifest["git_dirty"]
    assert manifest["objective"] == "tps_per_p95:maximize"
    assert manifest["imported_historical_observations"] == 0
    assert manifest["policy"]["label_precision_decimals"] == 6
    calibration = manifest["policy"].get("initial_noise_repeats", 0)
    gp_policy = manifest["policy"].get("gp_policy", "legacy")
    limits = manifest["limits"]
    assert limits == dataclass_dict(pilot_limits(calibration))
    if calibration or gp_policy != "legacy":
        assert manifest["policy"] == runtime_policy(gp_policy, calibration)
    assert (manifest["benchmark"]["workload"]["duration"],
            manifest["benchmark"]["workload"]["warmup_time"],
            manifest["benchmark"]["workload"]["threads"],
            manifest["benchmark"]["workload"]["rand_type"]) == (60, 15, 32, "uniform")
    for name, digest in load(p / "checksums.json").items():
        assert hashlib.sha256((p / name).read_bytes()).hexdigest() == digest
    records, attempts = lines(p / "trials.jsonl"), lines(audit / "attempts.jsonl")
    assert 1 <= len(records) == len(attempts) <= limits["trials"]
    assert [r["metadata"]["seq"] for r in records] == list(range(1, len(records) + 1))
    assert len({r["trial_id"] for r in records}) == len(records)
    for attempt, record in zip(attempts, records):
        assert attempt["trial_id"] == record["trial_id"] and attempt["result"] == record
        assert attempt["submitted"]["config"] == record["config"]
        assert attempt["submitted"]["prediction"] == record["prediction"]
    byid = {r["trial_id"]: r for r in records}
    if calibration:
        for r in records[1:calibration+1]:
            assert r["metadata"]["kind"] == "noise_calibration"
            assert r["config"] == records[0]["config"] and not r["metadata"]["accepted"]
    space = Space([Knob(**spec) for spec in manifest["space"]])
    snapshots = {}
    for path in sorted((p / "gp_snapshots").glob("*.json")):
        saved = load(path)
        assert len(saved["training_ids"]) == len(saved["x"]) == len(saved["target"])
        for ident, x, y in zip(saved["training_ids"], saved["x"], saved["target"]):
            r = byid[ident]
            assert (r["status"] == "measured" and r["complete"] and r["effective"]
                    and r["constraints_ok"])
            expected = space.encode(r["config"]) + r["context"]
            assert len(x) == len(expected) and all(abs(a-b) < 1e-12 for a, b in zip(x, expected))
            assert math.isclose(y, math.log(r["value"]), abs_tol=1e-12)
        if gp_policy == "repeat-aware":
            audit_noise = repeat_noise([Trial(**byid[i]) for i in saved["training_ids"]])
            fixed = (saved.get("noise_policy") or {}).get("fixed_noise_log_std", 0)
            assert math.isclose(fixed, audit_noise["fixed_noise_log_std"], rel_tol=1e-9, abs_tol=1e-12)
            assert saved["optimizer"]["optimized"]
        snapshots[str(path.relative_to(p))] = (saved, GaussianProcess.from_snapshot(saved))
    rows, forecasts, accepted = [], [], []
    best = None
    previous_elapsed = 0
    for r in records:
        meta, seq = r["metadata"], r["metadata"]["seq"]
        assert r["objective_key"] == manifest["objective"] and r["environment_key"] == records[0]["environment_key"]
        assert meta["simulated"] is False and meta["kind"] != "incumbent_repeat"
        assert previous_elapsed < meta["active_elapsed_seconds"] <= limits["wall_seconds"]
        previous_elapsed = meta["active_elapsed_seconds"]
        valid = (r["status"] == "measured" and r["complete"] and r["effective"]
                 and r["constraints_ok"] and r["value"] is not None)
        if valid:
            text = (p / meta["benchmark_log"]).read_text()
            tps = float(re.search(r"transactions:\s*\d+\s*\(([\d.]+) per sec", text).group(1))
            p95 = float(re.search(r"95th percentile:\s*([\d.]+)", text).group(1))
            duration = float(re.search(r"total time:\s*([\d.]+)s", text).group(1))
            errors = int(re.search(r"ignored errors:\s*(\d+)", text).group(1))
            assert tps == meta["raw_tps"] and p95 == meta["p95_latency_ms"]
            assert duration >= 59.9 and errors == 0 and "Target transaction rate:" not in text
            assert r["value"] == round(tps/p95, 6) and meta["joint_value_unrounded"] == tps/p95
            assert [(x["duration"], x["rate_flag"]) for x in meta["command_evidence"]] == [(15, None), (60, None)]
            assert all(x["status_code"] == 0 and x["configured_bytes"] == x["expected_bytes"]
                       for x in meta["resize_checks"])
            assert meta["resize_checks"][-1]["expected_bytes"] == 1073741824
            assert 59.9 <= meta["telemetry"]["delta"]["boundary_elapsed_seconds"] <= 70
            if meta["kind"] == "baseline":
                best = r["value"]
            elif meta["kind"] in {"startup", "candidate", "audit"}:
                should_accept = r["value"] > best*1.01
                assert meta["accepted"] == should_accept
                if should_accept:
                    best = r["value"]
                    accepted.append(seq)
            else:
                assert not meta["accepted"]
        if r["prediction"]:
            saved, model = snapshots[meta["model_snapshot"]]
            assert all(byid[i]["metadata"]["seq"] < seq for i in saved["training_ids"])
            pred = asdict(model.predict([space.encode(r["config"]) + r["context"]])[0])
            for key in ("mean", "latent_std", "observation_std"):
                assert math.isclose(pred[key], r["prediction"][key], rel_tol=1e-6, abs_tol=1e-7)
            if valid:
                forecasts.append({
                    "trial": seq, "kind": meta["kind"], "actual_objective": r["value"],
                    "predicted_objective": math.exp(pred["mean"]),
                    "lower_2_observation_std": math.exp(pred["mean"]-2*pred["observation_std"]),
                    "upper_2_observation_std": math.exp(pred["mean"]+2*pred["observation_std"]),
                    "latent_log_std": pred["latent_std"],
                    "observation_log_std": pred["observation_std"],
                    "absolute_percent_error": abs(math.exp(pred["mean"])/r["value"]-1)*100,
                    "covered_2_observation_std": abs(math.log(r["value"])-pred["mean"]) <= 2*pred["observation_std"],
                    "training_n": len(saved["training_ids"])})
        rows.append({
            "trial": seq, "kind": meta["kind"], "valid": valid, "accepted": meta["accepted"],
            "objective_value": r["value"], "tps": meta.get("raw_tps"),
            "p95_ms": meta.get("p95_latency_ms"), "accepted_best_objective": best,
            "active_elapsed_minutes": meta["active_elapsed_seconds"]/60,
            "config_json": json.dumps(r["config"], sort_keys=True),
            "mysqld_cpu_cores": (meta.get("telemetry") or {}).get("delta", {}).get("mysqld_cpu_cores"),
            "cgroup_cpu_cores": (meta.get("telemetry") or {}).get("delta", {}).get("cgroup_cpu_cores"),
            "throttled_period_fraction": (meta.get("telemetry") or {}).get("delta", {}).get("throttled_period_fraction")})
    curve = list(csv.DictReader((p / "curve.csv").open()))
    assert len(curve) == len(records)
    for row, r in zip(curve, records):
        for field, expected in (("tps", r["metadata"].get("raw_tps")),
                                ("p95_ms", r["metadata"].get("p95_latency_ms")),
                                ("objective_value", r["value"])):
            assert (row[field] == "" if expected is None else float(row[field]) == expected)
    events, refs = lines(p / "events.jsonl"), load(p / "reflections.json")
    decisions, prescreen_predictions = [], []
    models_by_version = {s["version"]: s for s, _ in snapshots.values()}
    decision_inputs = [
        (r["metadata"]["seq"], "selected", r["metadata"]["decision"], r["config"])
        for r in records if r["metadata"]["kind"] == "candidate"]
    decision_inputs += [
        (e["payload"]["next_seq"], "refused", e["payload"], None)
        for e in events if e["kind"] == "prescreen_refused"]
    for seq, outcome, decision, config in decision_inputs:
        assert decision["rule"] == "GP_LCB_then_UCB"
        prior = [r for r in records if r["metadata"]["seq"] < seq]
        incumbent_score = prior[0]["value"]
        for r in prior:
            if r["metadata"]["accepted"]:
                incumbent_score = r["value"]
        floor, gain = math.log(prior[0]["value"]*.9), math.log(incumbent_score*1.01)
        preds = decision["predictions"]
        assert len(preds) == len(decision["config_ids"])
        eligible = [i for i, pred in enumerate(preds)
                    if pred["mean"]-2*pred["latent_std"] >= floor
                    and pred["mean"]+2*pred["latent_std"] > gain]
        assert eligible == decision["eligible"]
        for pred in preds:
            saved = models_by_version[pred["model_version"]]
            assert all(byid[ident]["metadata"]["seq"] < seq for ident in saved["training_ids"])
        for ident, pred in zip(decision["config_ids"], preds):
            low, high = pred["mean"]-2*pred["latent_std"], pred["mean"]+2*pred["latent_std"]
            prescreen_predictions.append({
                "next_trial": seq, "decision_outcome": outcome, "config_id": ident,
                "model_version": pred["model_version"], "predicted_objective": math.exp(pred["mean"]),
                "lower_latent": math.exp(low), "upper_latent": math.exp(high),
                "lcb_pass": low >= floor, "ucb_pass": high > gain,
                "prediction_is_not_actual_performance": True})
        if outcome == "refused":
            assert not eligible
        else:
            chosen = max(eligible, key=lambda i: preds[i]["mean"]+2*preds[i]["latent_std"])
            assert decision["config_ids"][chosen] == fingerprint(config)
        decisions.append({"next_trial": seq, "outcome": outcome, "pool_size": len(preds),
                          "eligible": len(eligible), "training_n": len(saved["training_ids"]),
                          "lcb_pass": sum(p["mean"]-2*p["latent_std"] >= floor for p in preds),
                          "ucb_pass": sum(p["mean"]+2*p["latent_std"] > gain for p in preds),
                          "floor_objective": math.exp(floor), "required_gain_objective": math.exp(gain)})
    covered = set()
    for ref in refs:
        assert not set(ref["evidence_ids"]) - set(byid)
        assert not covered.intersection(ref["evidence_ids"])
        for fact in ref["facts"]:
            r = byid[fact["trial_id"]]
            assert fact["status"] == r["status"] and fact["value"] == r["value"]
        assert ref["conditions"]["objective_key"] == manifest["objective"]
        covered.update(ref["evidence_ids"])
    traces = [load(f) for f in sorted((p / "llm_calls").glob("*.json"))]
    ledger, state = runtime["ledger"], runtime["state"]
    usage = load(p / "llm_usage.json")
    known_tokens = sum((t.get("usage") or {}).get("total_tokens", 0) for t in traces)
    assert len(traces) == ledger["llm_calls"] <= limits["llm_calls"]
    assert known_tokens == usage["total_tokens"] <= ledger["llm_tokens_or_reserved"] <= limits["llm_tokens"]
    assert ledger["trial_slots_used"] == len(records) and summary["budget"]["elapsed_seconds"] <= limits["wall_seconds"]
    if gp_policy == "repeat-aware" and state.get("repeat_gp"):
        wrapper = state["repeat_gp"]
        assert wrapper["gp"] == state["gp"]
        assert wrapper["space_key"] == manifest["preflight"]["space_key"]
        assert wrapper["record_fingerprints"] == [
            record_fingerprint(Trial(**byid[i])) for i in wrapper["training_ids"]]
    refresh = sum(e["kind"] == "candidate_refresh" for e in events)
    rejections = sum(e["kind"] == "prescreen_refused" for e in events)
    assert refresh == state["refresh_count"] <= 2
    if summary.get("closed_loop_verified"):
        assert summary["status"] in {"completed", "completed_with_recovery"} and summary["final_reflection_committed"]
        assert set(byid) == covered and snapshots and forecasts
        assert rejections or any(r["metadata"]["kind"] == "candidate" for r in records)
        assert {t["role"] for t in traces} == {"selector", "pruner", "recommender", "reflection"}
    if summary.get("final_verification_value") is not None:
        assert records[-1]["metadata"]["kind"] == "verification"
        assert records[-1]["config"] == state["incumbent"]
        assert records[-1]["value"] == summary["final_verification_value"]
    assert summary["initial_configuration_restored"] and post["database_state_verified"]
    assert post["counts"] == [[2000000, 1, 2000000]]*4
    assert post["config_sha256"] == manifest["preflight"]["config_sha256"]
    assert post["concurrency_context"] == manifest["preflight"]["concurrency_initial"]
    first, last = records[0], records[-1]
    result = {
        "verified": True, "run_id": args.run_id,
        "closed_loop_verified": summary.get("closed_loop_verified", False),
        "status": summary["status"], "error": summary.get("error"), "objective": manifest["objective"],
        "archive_sha256": inventory["archive_sha256"], "git_revision": manifest["git_revision"],
        "trials": len(records), "valid": sum(r["valid"] for r in rows),
        "unique_configs": len({fingerprint(r["config"]) for r in records}),
        "counts_by_kind": {k: sum(r["kind"] == k for r in rows) for k in sorted({r["kind"] for r in rows})},
        "accepted_at": accepted, "baseline_tps": first["metadata"]["raw_tps"],
        "baseline_p95_ms": first["metadata"]["p95_latency_ms"], "baseline_objective": first["value"],
        "last_tps": last["metadata"]["raw_tps"], "last_p95_ms": last["metadata"]["p95_latency_ms"],
        "last_objective": last["value"], "last_kind": last["metadata"]["kind"],
        "independent_verification_available": last["metadata"]["kind"] == "verification",
        "last_objective_vs_baseline_pct": 100*(last["value"]/first["value"]-1),
        "search_value": state["best_value"], "incumbent": state["incumbent"],
        "llm_calls": len(traces), "known_tokens": known_tokens,
        "charged_tokens_or_reserved": ledger["llm_tokens_or_reserved"],
        "roles": [t["role"] for t in traces], "model": manifest["model"],
        "gp_training_count": len((state.get("gp") or {}).get("training_ids", [])),
        "prospective_predictions": len(forecasts), "gp_rejections": rejections,
        "refresh_count": refresh, "reflection_count": len(refs), "reflected_records": len(covered),
        "reflection_error": summary.get("reflection_error"), "stop_reason": summary.get("stop_reason"),
        "active_seconds": summary["budget"]["elapsed_seconds"], "initial_restored": True,
        "prediction_mape_pct": statistics.mean(f["absolute_percent_error"] for f in forecasts) if forecasts else None,
        "prediction_coverage": statistics.mean(f["covered_2_observation_std"] for f in forecasts) if forecasts else None,
        "gp_hyperparameters": (state.get("gp") or {}).get("hyperparameters"),
        "gp_kernel": (state.get("gp") or {}).get("kernel"),
        "gp_final_optimizer": (state.get("gp") or {}).get("optimizer"),
        "gp_policy": gp_policy, "initial_noise_repeats": calibration,
        "repeat_noise_audit": summary.get("repeat_noise_audit"), "limits": limits,
        "recovery": summary.get("recovery"),
        "invalid_attempts": sum(not r["valid"] for r in rows),
    }
    public = ROOT / "docs/experiment_data" / args.run_id.replace("-", "_")
    public.mkdir(parents=True, exist_ok=True)
    write_csv(public / "observations.csv", rows)
    write_csv(public / "predictions.csv", forecasts)
    write_csv(public / "decisions.csv", decisions)
    write_csv(public / "candidate_prescreen_predictions.csv", prescreen_predictions)
    write_csv(public / "gp_fits.csv", [{
        "version": s["version"], "training_n": len(s["training_ids"]), "kernel": s["kernel"],
        **s["hyperparameters"], "target_log_scale": s["target_scale"],
        "noise_log_units": s["hyperparameters"]["noise_std"]*s["target_scale"],
        "fixed_noise_log_std": (s.get("noise_policy") or {}).get("fixed_noise_log_std", 0),
        "effective_noise_log_std": math.hypot(s["hyperparameters"]["noise_std"]*s["target_scale"],
            (s.get("noise_policy") or {}).get("fixed_noise_log_std", 0)),
        "optimized": s["optimizer"]["optimized"],
        "optimizer_success": s["optimizer"].get("success"),
        "negative_log_marginal_likelihood": s["optimizer"].get("negative_log_marginal_likelihood")}
        for s, _ in snapshots.values()])
    write_csv(public / "llm_usage.csv", [
        {"call": i+1, "role": t["role"], "total_tokens": (t.get("usage") or {}).get("total_tokens"),
         "elapsed_seconds": t.get("elapsed_seconds"), "status": "error" if t.get("error") else "response"}
        for i, t in enumerate(traces)])
    (public / "range_outputs.json").write_text(json.dumps([
        {"call": i+1, "role": t["role"], "ranges": t.get("response", {}).get("ranges")}
        for i, t in enumerate(traces) if t["role"] == "pruner"],
        ensure_ascii=False, indent=2)+"\n")
    (public / "reflections.json").write_text(json.dumps({
        "numeric_facts_and_evidence_verified": True,
        "hypotheses_are_unverified_not_causal_rules": True,
        "reflections": refs}, ensure_ascii=False, indent=2)+"\n")
    (public / "verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")
    (public / "checksums.json").write_text(json.dumps({
        f.name: hashlib.sha256(f.read_bytes()).hexdigest()
        for f in public.iterdir() if f.is_file() and f.name != "checksums.json"}, indent=2)+"\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
