"""阶段1主指标：全过程P95、按真实完成时间累计成功请求、预定义规则的探索性适应耗时。"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from admtune.continuous import atomic_json
from admtune.replay_report import read_jsonl, write_csv
from tools.analyze_online_dynamic import analyze


def adaptation(rows, start_s, threshold_ms, consecutive=3):
    count = 0
    for r in rows:
        count = count+1 if r["p95_ms"] <= threshold_ms and r["requests"] == 200 else 0
        if count >= consecutive:
            return r["end_s"]-start_s
    return None


def create(batch, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    analyze(batch, output)
    summary = json.loads((output/"summary.json").read_text())
    windows = []
    with (output/"window_metrics.csv").open() as f:
        for r in csv.DictReader(f):
            windows.append({**r, **{k: float(r[k]) for k in (
                "start_s", "end_s", "p95_ms", "requests", "cumulative_response_time_s")}})
    targets, adaptations, completion = {}, [], []
    for index, phase in enumerate(("A1", "B", "A2")):
        ref = [r["p95_ms"] for r in windows if r["method"] == "Default" and r["phase"] == phase]
        targets[phase] = .9*float(np.median(ref))
        for method in ("Default", "ADMTune"):
            rows = [r for r in windows if r["method"] == method and r["phase"] == phase]
            adaptations.append({"method": method, "phase": phase, "target_p95_ms": targets[phase],
                "attainment_seconds": adaptation(rows, 600*index, targets[phase]),
                "definition": "first confirmation of three consecutive complete 20s cohorts at <=90% "
                              "of paired Default median 20s P95",
                "pilot_posthoc_scoring": True})
    for method in ("Default", "ADMTune"):
        rs = read_jsonl(batch/method/"requests.jsonl")
        successful = sorted(r["completed"] for r in rs if r["ok"])
        times = sorted(set(range(0, 1801, 20)) | {max(1800., successful[-1])})
        for t in times:
            completion.append({"method": method, "at_s": t,
                "cumulative_successful_requests": int(np.searchsorted(successful, t, side="right"))})
    write_csv(output/"adaptation_metrics.csv", adaptations, list(adaptations[0]))
    write_csv(output/"successful_request_curves.csv", completion, list(completion[0]))
    atomic_json(output/"metric_protocol.json", {"primary": [
        "end-to-end P95 including configuration transitions", "actual cumulative successful completions",
        "time to first three qualifying 20s cohorts"], "threshold_rule": "0.9 * paired Default median window P95",
        "pilot_scoring_defined_after_run": True, "pilot_not_counted_as_frozen_formal_repetition": True,
        "fixed_rate_counts_may_overlap": True, "all_requests_included": True})
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.9))
    colors = {"Default": "#777777", "ADMTune": "#d62728"}
    for method in ("Default", "ADMTune"):
        w = [r for r in windows if r["method"] == method]
        axes[0].plot([r["end_s"]/60 for r in w], [r["p95_ms"] for r in w],
                     color=colors[method], lw=1.15, label=method)
        c = [r for r in completion if r["method"] == method]
        axes[1].plot([r["at_s"]/60 for r in c], [r["cumulative_successful_requests"] for r in c],
                     color=colors[method], lw=1.3, ls="--" if method == "Default" else "-", label=method)
    axes[0].set(title="(a) Continuous end-to-end latency", ylabel="P95 latency (ms; 20 s cohorts)")
    axes[1].set(title="(b) Successful request completions", ylabel="Cumulative successful requests")
    for ax in axes[:2]:
        for boundary in (10, 20): ax.axvline(boundary, color="#444444", ls=":", lw=.7)
        ax.set(xlabel="Offered workload time (min)", xlim=(0, 30.1))
        ax.legend(frameon=False, fontsize=8)
    ax = axes[2]; x = np.arange(3)
    finite = [r["attainment_seconds"] for r in adaptations if r["attainment_seconds"] is not None]
    ymax = max([600, *finite])
    for index, method in enumerate(("Default", "ADMTune")):
        rows = [r for r in adaptations if r["method"] == method]
        for k, r in enumerate(rows):
            pos = x[k]+(index-.5)*.35
            value = r["attainment_seconds"]
            if value is None:
                ax.text(pos, ymax*.025, "NA", ha="center", fontsize=8, color=colors[method])
            else:
                ax.bar(pos, value, width=.35, color=colors[method])
    ax.set(xticks=x, xticklabels=["A1", "B", "A2"], xlim=(-.55, 2.55), ylim=(0, ymax*1.1),
           title="(c) First sustained target attainment", ylabel="Confirmation time from phase start (s)")
    for ax in axes:
        ax.grid(alpha=.2, lw=.5); ax.tick_params(direction="in", top=True, right=True)
    fig.subplots_adjust(top=.86, bottom=.28, wspace=.34)
    fig.text(.5, .06, "Single-run pilot. NA: target not attained. Success-count curves can overlap at fixed 10 QPS.\n"
             "Attainment scoring is exploratory; the same rule must be frozen before formal runs.",
             ha="center", fontsize=8)
    for suffix in ("png", "pdf"):
        fig.savefig(output/f"stage1_primary_metrics.{suffix}", dpi=260, bbox_inches="tight")
    plt.close(fig)
    return {"performance": summary["comparison"], "adaptation": adaptations}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batch", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(create(a.batch, a.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
