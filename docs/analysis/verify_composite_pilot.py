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
    parser.add_argument("--archive", type=Path,
                        default=ROOT / "runs/archives/composite-pilot-001.tar.gz")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "runs/collected-composite-pilot-001")
    args = parser.parse_args()
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
    p = args.output / "runs/composite-pilot-001"
    audit = args.output / "runs/composite-pilot-001-audit"
    summary, manifest, post = (load(p / "summary.json"), load(p / "manifest.json"),
                               load(audit / "post_run.json"))
    runtime = load(audit / "runtime.json")
    assert not manifest["simulated"] and not manifest["git_dirty"]
    assert manifest["objective"] == "tps_per_p95:maximize"
    assert manifest["imported_historical_observations"] == 0
    assert manifest["policy"]["label_precision_decimals"] == 6
    assert manifest["limits"] == {
        "trials": 8, "wall_seconds": 1200, "llm_calls": 8,
        "llm_tokens": 32000, "recovery_reserve_seconds": 60}
    assert (manifest["benchmark"]["workload"]["duration"],
            manifest["benchmark"]["workload"]["warmup_time"],
            manifest["benchmark"]["workload"]["threads"],
            manifest["benchmark"]["workload"]["rand_type"]) == (60, 15, 32, "uniform")
    for name, digest in load(p / "checksums.json").items():
        assert hashlib.sha256((p / name).read_bytes()).hexdigest() == digest
    records, attempts = lines(p / "trials.jsonl"), lines(audit / "attempts.jsonl")
    assert 1 <= len(records) == len(attempts) <= 8
    assert [r["metadata"]["seq"] for r in records] == list(range(1, len(records) + 1))
    assert len({r["trial_id"] for r in records}) == len(records)
    for attempt, record in zip(attempts, records):
        assert attempt["trial_id"] == record["trial_id"] and attempt["result"] == record
        assert attempt["submitted"]["config"] == record["config"]
        assert attempt["submitted"]["prediction"] == record["prediction"]
    byid = {r["trial_id"]: r for r in records}
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
        snapshots[str(path.relative_to(p))] = (saved, GaussianProcess.from_snapshot(saved))
    rows, forecasts, accepted = [], [], []
    best = None
    previous_elapsed = 0
    for r in records:
        meta, seq = r["metadata"], r["metadata"]["seq"]
        assert r["objective_key"] == manifest["objective"] and r["environment_key"] == records[0]["environment_key"]
        assert meta["simulated"] is False and meta["kind"] != "incumbent_repeat"
        assert previous_elapsed < meta["active_elapsed_seconds"] <= 1200
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
            "throttled_period_fraction": (meta.get("telemetry") or {}).get("delta", {}).get("throttled_period_fraction")})
    curve = list(csv.DictReader((p / "curve.csv").open()))
    assert len(curve) == len(records)
    for row, r in zip(curve, records):
        assert float(row["tps"]) == r["metadata"]["raw_tps"]
        assert float(row["p95_ms"]) == r["metadata"]["p95_latency_ms"]
        assert float(row["objective_value"]) == r["value"]
    events, refs = lines(p / "events.jsonl"), load(p / "reflections.json")
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
    assert len(traces) == ledger["llm_calls"] <= 8
    assert known_tokens == usage["total_tokens"] <= ledger["llm_tokens_or_reserved"] <= 32000
    assert ledger["trial_slots_used"] == len(records) and summary["budget"]["elapsed_seconds"] <= 1200
    refresh = sum(e["kind"] == "candidate_refresh" for e in events)
    rejections = sum(e["kind"] == "prescreen_refused" for e in events)
    assert refresh == state["refresh_count"] <= 2
    if summary.get("closed_loop_verified"):
        assert summary["status"] == "completed" and summary["final_reflection_committed"]
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
        "verified": True, "closed_loop_verified": summary.get("closed_loop_verified", False),
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
    }
    public = ROOT / "docs/experiment_data/composite_pilot_001"
    public.mkdir(parents=True, exist_ok=True)
    write_csv(public / "observations.csv", rows)
    write_csv(public / "predictions.csv", forecasts)
    write_csv(public / "llm_usage.csv", [
        {"call": i+1, "role": t["role"], "total_tokens": (t.get("usage") or {}).get("total_tokens"),
         "elapsed_seconds": t.get("elapsed_seconds"), "status": "error" if t.get("error") else "response"}
        for i, t in enumerate(traces)])
    (public / "range_outputs.json").write_text(json.dumps([
        {"call": i+1, "role": t["role"], "ranges": t.get("response", {}).get("ranges")}
        for i, t in enumerate(traces) if t["role"] == "pruner"],
        ensure_ascii=False, indent=2)+"\n")
    (public / "verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")
    (public / "checksums.json").write_text(json.dumps({
        f.name: hashlib.sha256(f.read_bytes()).hexdigest()
        for f in public.iterdir() if f.is_file() and f.name != "checksums.json"}, indent=2)+"\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
