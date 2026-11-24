"""仅从通过核验的真实先导记录绘图；缺失基线明确列为待运行。"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics

from .contracts import fingerprint
from .replay_report import write_csv


PENDING = ["GPTuner", "AgentTune", "CDBTune", "MySQLTuner"]
METHOD_STYLE = {
    "ADMTune": ("#1f77b4", "v"),
    "AgentTune": ("#ff7f0e", "D"),
    "GPTuner": ("#9467bd", "P"),
    "CDBTune": ("#2ca02c", "*"),
    "MySQLTuner": ("#8c564b", "^"),
    "Default": ("#7f7f7f", None),
}


def load_run(root):
    manifest = json.loads((root / "manifest.json").read_text())
    summary = json.loads((root / "summary.json").read_text())
    if (manifest.get("simulated") is not False or manifest.get("git_dirty") is not False
            or not summary.get("complete") or not summary.get("initial_configuration_restored")
            or not manifest.get("data_snapshot_restored_by_harness")
            or manifest.get("method") not in {"Default", "ADMTune"}):
        raise ValueError("Only completed, restored, clean-source real pilot runs are accepted")
    checksums = json.loads((root / "checksums.json").read_text())
    for name, expected in checksums.items():
        p = Path(name)
        if p.is_absolute() or ".." in p.parts or root.joinpath(p).is_symlink():
            raise ValueError("Unsafe artifact path")
        if hashlib.sha256(root.joinpath(p).read_bytes()).hexdigest() != expected:
            raise ValueError("Artifact checksum mismatch: " + name)
    if not {"manifest.json", "summary.json", "curve.csv", "trials.jsonl"} <= checksums.keys():
        raise ValueError("Missing mandatory evidence")
    with (root / "curve.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    if not rows or any(row["method"] != manifest["method"] for row in rows):
        raise ValueError("Missing or inconsistent measurements")
    return manifest, summary, rows


def create(default_dir, admtune_dir, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    default, dsummary, drows = load_run(default_dir)
    tuned, tsummary, trows = load_run(admtune_dir)
    if default["environment"] != tuned["environment"] or default["benchmark"] != tuned["benchmark"]:
        raise ValueError("Different environment/protocol/space: do not combine curves")
    output.mkdir(parents=True, exist_ok=False)
    valid_d = [r for r in drows if r["valid"] == "True"]
    if not valid_d:
        raise ValueError("No valid Default measurement")
    # 同一未调优配置下三次顺序测量的中位数，不声称跨种子 CI。
    references = {
        "best_tps": statistics.median(float(r["tps"]) for r in valid_d),
        "best_config_p95_ms": statistics.median(float(r["p95_ms"]) for r in valid_d),
    }
    search = [r for r in trows if r["kind"] != "verification" and r["best_tps"]]
    verification = [r for r in trows if r["kind"] == "verification" and r["valid"] == "True"]
    if not search or not verification:
        raise ValueError("Search and independent verification evidence required")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7), constrained_layout=False)
    for row_index, (metric, title, raw) in enumerate([
        ("best_tps", "Best-found throughput (TPS; higher is better)", "tps"),
        ("best_config_p95_ms", "P95 of TPS-selected configuration (ms)", "p95_ms"),
    ]):
        for col, (xfield, xlabel, scale) in enumerate([
            ("trial", "True workload replays (verification uses a slot)", 1),
            ("ended_seconds", "Elapsed tuning time (min)", 60),
        ]):
            ax = axes[row_index, col]
            x = [float(r[xfield]) / scale for r in search]
            y = [float(r[metric]) for r in search]
            ax.step(x, y, where="post", color="#1565c0", linewidth=2, label="ADMTune static pilot")
            ax.scatter(x, y, color="#1565c0", s=19, zorder=3)
            xmax = max(float(r[xfield]) / scale for r in trows)
            ax.hlines(references[metric], 0, xmax, colors="#5d6672", linestyles="--",
                      linewidth=1.6, label=f"Default median ({len(valid_d)} measurements)")
            ax.scatter([float(r[xfield]) / scale for r in verification],
                       [float(r[raw]) for r in verification], s=65, marker="D",
                       color="#d77619", edgecolors="white", linewidths=.7,
                       label="Independent verification", zorder=4)
            ax.set(xlabel=xlabel, ylabel="TPS" if row_index == 0 else "P95 latency (ms)",
                   title=title, xlim=(0, xmax * 1.05))
            ax.grid(alpha=.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .925), ncol=3, frameon=False)
    fig.suptitle("MySQL / Sysbench read-only — real pilot, one seed", fontsize=15, y=.985)
    fig.text(.5, .025, "Incomplete baseline matrix: GPTuner, AgentTune, CDBTune and MySQLTuner not yet measured.\n"
             "TPS is the optimization objective. No cross-seed confidence interval; no formal superiority claim.",
             ha="center", fontsize=9, color="#636b75")
    fig.subplots_adjust(top=.82, bottom=.15, hspace=.6, wspace=.28)
    fig.savefig(output / "performance_diagnostics.png", dpi=220)
    fig.savefig(output / "performance_diagnostics.pdf")
    plt.close(fig)
    # 主图参照 AgentTune 图8：细折线、离散标记、原始 replay 次数、灰色默认线。
    fig, ax = plt.subplots(figsize=(6.2, 4.5))
    color, marker = METHOD_STYLE["ADMTune"]
    x = [int(r["trial"]) for r in search]
    ax.plot(x, [float(r["best_tps"]) for r in search], color=color, marker=marker,
            markersize=4.5, linewidth=1.1, label="ADMTune (static pilot)")
    xmax = max(int(r["trial"]) for r in trows)
    ax.hlines(references["best_tps"], 0, xmax, color=METHOD_STYLE["Default"][0],
              linestyle="--", linewidth=1, label=f"Default (median, n={len(valid_d)})")
    ax.scatter([int(r["trial"]) for r in verification], [float(r["tps"]) for r in verification],
               color="#d77619", marker="D", s=30, edgecolors="white", linewidths=.5,
               label="Final verification (not search)", zorder=4)
    ax.set(xlabel="# Workload Replays", ylabel="Throughput (TPS)",
           title="SYSBENCH - OLTP (read-only)", xlim=(0, xmax + .35))
    from matplotlib.ticker import MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(color="#d0d0d0", linewidth=.5, alpha=.7)
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.legend(loc="lower center", bbox_to_anchor=(.5, 1.16), ncol=2, frameon=False, fontsize=8.5)
    fig.text(.5, .035, "One-seed pilot. Other baselines not yet measured; no superiority claim.",
             ha="center", fontsize=8, color="#666666")
    fig.subplots_adjust(top=.74, bottom=.18, left=.15, right=.96)
    fig.savefig(output / "performance_pilot.png", dpi=260)
    fig.savefig(output / "performance_pilot.pdf")
    plt.close(fig)
    merged = drows + trows
    write_csv(output / "all_measurements.csv", merged, list(merged[0]))
    statuses = [{"method": m, "status": "measured_pilot" if m in {"ADMTune", "Default"} else "not_run",
                 "curve_data_created": m in {"ADMTune", "Default"}}
                for m in ("ADMTune", "GPTuner", "AgentTune", "CDBTune", "MySQLTuner", "Default")]
    write_csv(output / "method_status.csv", statuses, list(statuses[0]))
    report = {
        "pilot": True, "formal_comparison_complete": False, "pending": PENDING,
        "default_reference": references, "admtune_summary": tsummary,
        "default_summary": dsummary, "environment": default["environment"],
        "sources": {str(p.resolve()): fingerprint(json.loads((p / "checksums.json").read_text()))
                    for p in (default_dir, admtune_dir)},
    }
    (output / "plot_manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--default-run", type=Path, required=True)
    parser.add_argument("--admtune-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(create(args.default_run, args.admtune_run, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
