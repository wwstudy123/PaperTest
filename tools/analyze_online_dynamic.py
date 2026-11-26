"""从连续真实请求导出动态对比，不把试验最优点当作全过程服务性能。"""
import argparse
import json
from pathlib import Path

import numpy as np

from admtune.continuous import atomic_json
from admtune.contracts import fingerprint
from admtune.replay_report import read_jsonl, write_csv


def analyze(batch, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output.mkdir(parents=True, exist_ok=False)
    calibration = read_jsonl(batch/"calibration/trials.jsonl")
    crows = [{"block": r["block"], "kind": r["kind"], "name": r["name"],
              "p95_ms": r["value_p95_ms"], "valid": r["valid"],
              "config_json": json.dumps(r["config"], sort_keys=True),
              "cpu_core_equivalents": r["process_metrics_full_stream"]["cpu_core_equivalents"]}
             for r in calibration]
    write_csv(output/"calibration.csv", crows, list(crows[0]))
    atomic_json(output/"calibration_summary.json", json.loads((batch/"calibration/summary.json").read_text()))
    plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 10, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
    for ax, block in zip(axes, ("scan", "sort")):
        rs = [r for r in crows if r["block"] == block]
        ax.plot(range(len(rs)), [r["p95_ms"] for r in rs], "-o", lw=1.3, ms=5)
        ax.set(xticks=range(len(rs)), xticklabels=["Initial", "Candidate", "Initial"][:len(rs)],
               ylabel="P95 latency (ms)", title=block.capitalize())
        ax.grid(alpha=.2); ax.tick_params(direction="in", top=True, right=True)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output/f"calibration.{suffix}", dpi=240, bbox_inches="tight")
    plt.close(fig)
    if not all((batch/m/"summary.json").exists() for m in ("Default", "ADMTune")):
        return {"comparison_available": False, "reason": "Dynamic pair not completed; calibration only"}
    phase_rows, windows, config_rows, summaries = [], [], [], {}
    manifests = {m: json.loads((batch/m/"manifest.json").read_text()) for m in ("Default", "ADMTune")}
    summary = {m: json.loads((batch/m/"summary.json").read_text()) for m in manifests}
    if not all(s.get("complete") for s in summary.values()):
        raise ValueError("Incomplete continuous request run; do not imply full-pair performance")
    if summary["Default"]["schedule_hash"] != summary["ADMTune"]["schedule_hash"]:
        raise ValueError("Request plans differ")
    for method in ("Default", "ADMTune"):
        root = batch/method
        rs = sorted(read_jsonl(root/"requests.jsonl"), key=lambda r: r["seq"])
        t = np.array([r["scheduled"] for r in rs])
        lat = np.array([1000*(r["completed"]-r["emitted"]) for r in rs])
        if len(rs) != 18000 or any(not r["ok"] for r in rs):
            raise ValueError("Expected complete successful 18000-request read-only stream")
        total = 0.
        for left in range(0, 1800, 20):
            mask = (t >= left) & (t < left+20)
            ys = lat[mask]; total += float(ys.sum())/1000
            windows.append({"method": method, "start_s": left, "end_s": left+20,
                "phase": ("A1", "B", "A2")[left//600], "requests": int(mask.sum()),
                "p95_ms": float(np.percentile(ys, 95)), "mean_ms": float(ys.mean()),
                "cumulative_response_time_s": total})
        for index, phase in enumerate(("A1", "B", "A2")):
            ys = lat[(t >= 600*index) & (t < 600*(index+1))]
            phase_rows.append({"method": method, "phase": phase, "requests": len(ys),
                              "p95_ms": float(np.percentile(ys, 95)), "mean_ms": float(ys.mean()),
                              "total_response_time_s": float(ys.sum())/1000})
        summaries[method] = {"requests": len(rs), "overall_p95_ms": float(np.percentile(lat, 95)),
                            "total_response_time_s": float(lat.sum())/1000,
                            "counts": summary[method]["counts"], "phase_ready": summary[method]["phase_ready"]}
        if method == "ADMTune":
            trials = read_jsonl(root/"trials.jsonl")
            configs = {fingerprint(r["config"]): r["config"] for r in trials}
            initial = manifests[method]["preflight"]["current_config"]
            configs[fingerprint(initial)] = initial
            last = None
            for r in rs:
                h = r.get("session_config_hash")
                if h != last and h in configs:
                    c = configs[h]; last = h
                    config_rows.append({"at_s": r["scheduled"], "generation": r["config_generation"],
                        "buffer_gib": c["innodb_buffer_pool_size"]/1024**3,
                        "sort_mib": c["sort_buffer_size"]/1024**2,
                        "config_json": json.dumps(c, sort_keys=True)})
            if config_rows and config_rows[-1]["at_s"] < rs[-1]["scheduled"]:
                config_rows.append({**config_rows[-1], "at_s": rs[-1]["scheduled"]})
            usage = json.loads((root/"llm_usage.json").read_text())
            summaries[method].update({k: summary[method].get(k) for k in (
                "valid_observations", "reflection_commits", "gp_fit_events", "llm_calls",
                "control_flow_completed", "closed_loop_completed", "gp_selected_execution_exercised", "phases")})
            summaries[method]["llm_tokens"] = usage["total_tokens"]
            summaries[method]["model"] = usage["model"]
            (output/"configuration_trials.jsonl").write_text("".join(
                json.dumps({"trial_id": r["trial_id"], "phase_id": r["phase_id"], "kind": r["metadata"]["kind"],
                    "config": r["config"], "value_p95_ms": r["value"], "prediction": r["prediction"],
                    "accepted": r["metadata"].get("accepted"), "status": r["status"],
                    "measurement_start": r["metadata"].get("measurement_start"),
                    "measurement_end": r["metadata"].get("measurement_end")}, ensure_ascii=False)+"\n" for r in trials))
    write_csv(output/"window_metrics.csv", windows, list(windows[0]))
    write_csv(output/"phase_metrics.csv", phase_rows, list(phase_rows[0]))
    write_csv(output/"configuration_trace.csv", config_rows, list(config_rows[0]))
    summaries["comparison"] = {"cumulative_response_time_reduction_pct": 100*(
        1-summaries["ADMTune"]["total_response_time_s"]/summaries["Default"]["total_response_time_s"]),
        "schedule_hash": summary["Default"]["schedule_hash"], "same_request_plan": True,
        "requests_include_configuration_transitions": True,
        "window_seconds": 20, "independent_repetitions": 1}
    atomic_json(output/"summary.json", summaries)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2))
    colors = {"Default": "#777777", "ADMTune": "#d62728"}
    for method in ("Default", "ADMTune"):
        ws = [r for r in windows if r["method"] == method]
        for ax, key in zip(axes[0], ("p95_ms", "cumulative_response_time_s")):
            ax.plot([r["end_s"]/60 for r in ws], [r[key] for r in ws],
                    color=colors[method], lw=1.35, label=method)
    for ax, title, ylabel in zip(axes[0],
            ("(a) Continuous service latency", "(b) Cumulative response time"),
            ("P95 latency (ms; 20 s cohorts)", "Sum of request response times (s)")):
        ax.set(xlabel="Offered workload time (min)", ylabel=ylabel, title=title, xlim=(0, 30))
        for boundary in (10, 20): ax.axvline(boundary, color="#444444", ls=":", lw=.8)
        ax.legend(frameon=False, fontsize=9)
    ax = axes[1, 0]; right = ax.twinx()
    ax.step([r["at_s"]/60 for r in config_rows], [r["buffer_gib"] for r in config_rows],
            where="post", color="#1f77b4", lw=1.2, label="Buffer pool")
    right.step([r["at_s"]/60 for r in config_rows], [r["sort_mib"] for r in config_rows],
               where="post", color="#ff7f0e", lw=1.2, label="Sort buffer")
    ax.set(title="(c) Applied configurations (including trials)", xlabel="Offered workload time (min)",
           ylabel="Buffer pool (GiB)", xlim=(0, 30))
    right.set_ylabel("Sort buffer (MiB)")
    ax.legend(loc="upper left", frameon=False, fontsize=8)
    right.legend(loc="upper right", frameon=False, fontsize=8)
    ax = axes[1, 1]; x = np.arange(3)
    for index, method in enumerate(("Default", "ADMTune")):
        rows = [r for r in phase_rows if r["method"] == method]
        ax.bar(x+(index-.5)*.35, [r["p95_ms"] for r in rows], width=.35,
               color=colors[method], label=method)
    ax.set(xticks=x, xticklabels=["A1: 90% scan", "B: 90% sort", "A2: 90% scan"],
           title="(d) Whole-phase tail latency", ylabel="P95 latency (ms; all 6000 requests)")
    ax.legend(frameon=False, fontsize=8)
    for ax in axes.flat:
        ax.grid(alpha=.2, lw=.5); ax.tick_params(direction="in", top=True, right=True)
    fig.subplots_adjust(bottom=.13, top=.94, wspace=.35, hspace=.48)
    fig.text(.5, .025, "One real run per method; identical read-only request schedule. "
             "All requests included, not only accepted configurations.", ha="center", fontsize=8)
    for suffix in ("png", "pdf"):
        fig.savefig(output/f"dynamic_comparison.{suffix}", dpi=260, bbox_inches="tight")
    plt.close(fig)
    return summaries["comparison"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batch", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(analyze(a.batch, a.output), indent=2))


if __name__ == "__main__":
    main()
