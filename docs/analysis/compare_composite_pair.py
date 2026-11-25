"""比较已核验的同预算两臂先导；只用各自实测，不拼接历史或文献曲线。"""
from pathlib import Path
import csv
import hashlib
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
datasets = []
for arm, label, color, marker in [
        ("legacy", "Legacy GP", "#1f77b4", "o"),
        ("repeat", "Repeat-aware GP", "#d62728", "s")]:
    folder = ROOT/f"docs/experiment_data/composite_pair_{arm}_001"
    v = json.loads((folder/"verification.json").read_text())
    assert v["verified"] and v["closed_loop_verified"] and v["initial_noise_repeats"] == 3
    assert v["independent_verification_available"] and v["limits"]["trials"] == 10
    rows = list(csv.DictReader((folder/"observations.csv").open()))
    datasets.append((arm, label, color, marker, v, rows))
assert datasets[0][4]["limits"] == datasets[1][4]["limits"]
assert datasets[0][4]["model"] == datasets[1][4]["model"]
manifests = []
for arm, *_ in datasets:
    run = f"composite-pair-{arm}-001"
    p = ROOT/f"runs/collected-{run}/runs/{run}/manifest.json"
    manifests.append(json.loads(p.read_text()))
for key in ("benchmark", "space", "git_revision", "model"):
    assert manifests[0][key] == manifests[1][key]
for key in ("current_config", "config_sha256", "db_version", "invariants"):
    assert manifests[0]["preflight"][key] == manifests[1]["preflight"][key]
assert manifests[0]["restore"]["snapshot_manifest_sha256"] == manifests[1]["restore"]["snapshot_manifest_sha256"]
out = ROOT/"docs/experiment_data/composite_gp_pair_001"
out.mkdir(parents=True, exist_ok=True)
summary = []
for arm, label, _, _, v, rows in datasets:
    summary.append({
        "arm": arm, "gp_policy": v["gp_policy"], "valid_measurements": v["valid"],
        "llm_calls": v["llm_calls"], "actual_known_tokens": v["known_tokens"],
        "charged_tokens_or_reserved": v["charged_tokens_or_reserved"],
        "budget_accounting_minutes": v["active_seconds"]/60, "baseline_tps": v["baseline_tps"],
        "baseline_p95_ms": v["baseline_p95_ms"], "baseline_objective": v["baseline_objective"],
        "search_objective": v["search_value"], "verification_tps": v["last_tps"],
        "verification_p95_ms": v["last_p95_ms"], "verification_objective": v["last_objective"],
        "verification_vs_own_baseline_pct": v["last_objective_vs_baseline_pct"],
        "accepted_new_configurations": len(v["accepted_at"]),
        "gp_selected_candidates": v["counts_by_kind"].get("candidate", 0),
        "gp_rejection_decisions": v["gp_rejections"], "prediction_n": v["prospective_predictions"],
        "prediction_mape_pct": v["prediction_mape_pct"],
        "prediction_coverage": v["prediction_coverage"], "stop_reason": v["stop_reason"],
        "interrupted_attempts": v.get("invalid_attempts", 0), "recovered": bool(v.get("recovery")),
        "archive_sha256": v["archive_sha256"]})
with (out/"comparison.csv").open("w") as f:
    w = csv.DictWriter(f, fieldnames=list(summary[0]), lineterminator="\n")
    w.writeheader(); w.writerows(summary)
(out/"protocol_check.json").write_text(json.dumps({
    "same_code_benchmark_space_model_initial_snapshot_and_budgets": True,
    "independent_measurements_and_llm_responses": True,
    "execution_order": ["legacy", "repeat"],
    "single_pilot_not_significance_or_causal_ablation": True,
    "strict_uninterrupted_pair": not any(v.get("recovery") for *_, v, rows in datasets),
    "results": summary}, ensure_ascii=False, indent=2)+"\n")
plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 10, "pdf.fonttype": 42})
fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
maxima = [100., 0., 0.]
for arm, label, color, marker, v, rows in datasets:
    valid = [r for r in rows if r["valid"] == "True"]
    x = [int(r["trial"]) for r in valid]
    accepted = [100*float(r["accepted_best_objective"])/v["baseline_objective"] for r in valid]
    maxima[0] = max(maxima[0], *accepted, 100*v["last_objective"]/v["baseline_objective"])
    maxima[1] = max(maxima[1], *(float(r["tps"]) for r in valid))
    maxima[2] = max(maxima[2], *(float(r["p95_ms"]) for r in valid))
    axes[0].step(x, accepted, where="post", color=color, marker=marker, lw=1.2, ms=3, label=label)
    axes[0].scatter(x[-1], 100*v["last_objective"]/v["baseline_objective"],
                    color=color, marker="*", s=85, zorder=6)
    for ax, field in zip(axes[1:], ("tps", "p95_ms")):
        ax.plot([int(r["trial"]) for r in rows],
                [float(r[field]) if r["valid"] == "True" else float("nan") for r in rows],
                "-"+marker, color=color, lw=1.2, ms=3, label=label)
    for row in rows:
        if row["valid"] != "True":
            for ax in axes:
                ax.axvline(int(row["trial"]), color=color, ls=":", lw=.7, alpha=.45)
axes[0].axhline(100, color="#777777", ls="--", lw=.9, label="Own initial baseline")
for ax, title, ylabel, ymax in zip(axes,
        ["(a) Accepted objective", "(b) Measured throughput", "(c) Measured tail latency"],
        ["TPS/P95 (% of own baseline)", "Throughput (TPS)", "Latency (ms)"], maxima):
    ax.set_title(title)
    ax.set_xlabel("Trial slot (including calibration)")
    ax.set_ylabel(ylabel)
    ax.set_xlim(left=0)
    ax.set_ylim(0, ymax*1.12)
    ax.grid(alpha=.25, lw=.6)
    ax.legend(frameon=False, fontsize=7, loc="best")
    ax.tick_params(direction="in", top=True, right=True)
fig.subplots_adjust(bottom=.27, top=.85, wspace=.38)
fig.text(.5, .025, "Budgeted pilot; stars: final measurements. Repeat-aware arm recovered after one interrupted slot. "
         "Not an uninterrupted or multi-seed ranking.", ha="center", fontsize=8)
image = ROOT/"docs/figures/composite_gp_pair_001"
for extension in (".png", ".pdf"):
    fig.savefig(image.with_suffix(extension), dpi=280, bbox_inches="tight")
plt.close(fig)
(out/"checksums.json").write_text(json.dumps({
    p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir()
    if p.is_file() and p.name != "checksums.json"}, indent=2)+"\n")
print(json.dumps(summary, ensure_ascii=False, indent=2))
