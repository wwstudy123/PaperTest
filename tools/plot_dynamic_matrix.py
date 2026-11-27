"""离线整理已完成的四方法动态首轮：真实服务曲线、阶段指标和资源开销。"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from admtune.continuous import atomic_json
from admtune.replay_report import read_jsonl, write_csv
from tools.analyze_dynamic_stage1 import adaptation

METHODS = ("Default", "AgenticDB", "AgentTune", "ADMTune")
PHASES = ("A1", "B", "A2")
COLORS = ("#777777", "#1f77b4", "#ff7f0e", "#d62728")


def configuration_counts(trials, fixed_default=False):
    attempted = {json.dumps(r["config"], sort_keys=True) for r in trials}
    measured = {json.dumps(r["config"], sort_keys=True) for r in trials
                if r["status"] == "measured" and r["complete"] and r["effective"]}
    return {"unique_measured_configurations": 1 if fixed_default else len(measured),
            "unique_attempted_configurations": 1 if fixed_default else len(attempted)}


def service_metrics(records):
    """不以调优观测窗口代替服务全过程，不把窗口分位数平均成全程分位数。"""
    if len(records) != 18000 or len({r["seq"] for r in records}) != 18000:
        raise ValueError("Expected exactly 18000 unique requests")
    if not all(r["ok"] for r in records):
        raise ValueError("Failures present: inspect and report them before using this all-success plot")
    scheduled = np.array([r["scheduled"] for r in records])
    latency = np.array([1000 * (r["completed"] - r["emitted"]) for r in records])
    if (not np.isfinite(latency).all() or (latency < 0).any()
            or (scheduled < 0).any() or (scheduled >= 1800).any()):
        raise ValueError("Invalid timing evidence")
    windows, phases = [], []
    for left in range(0, 1800, 20):
        ys = latency[(scheduled >= left) & (scheduled < left + 20)]
        if len(ys) != 200:
            raise ValueError("Incomplete fixed-rate request cohort")
        windows.append({"start_s": left, "end_s": left + 20, "phase": PHASES[left // 600],
                        "requests": len(ys), "p95_ms": float(np.percentile(ys, 95)),
                        "mean_ms": float(ys.mean())})
    for i, phase in enumerate(PHASES):
        ys = latency[(scheduled >= 600 * i) & (scheduled < 600 * (i + 1))]
        phases.append({"phase": phase, "requests": len(ys),
                       "p95_ms": float(np.percentile(ys, 95)), "mean_ms": float(ys.mean())})
    successful = sorted(r["completed"] for r in records)
    times = sorted(set(range(0, 1801, 20)) | {max(1800., successful[-1])})
    completions = [{"at_s": t, "successful_requests": int(np.searchsorted(successful, t, side="right"))}
                   for t in times]
    return {"overall_p95_ms": float(np.percentile(latency, 95)),
            "mean_ms": float(latency.mean()), "successful_requests": len(records)}, windows, phases, completions


def collect(batch, seed, extra_runs=()):
    matrix = json.loads((batch / "matrix.json").read_text())
    if matrix["status"] != "completed":
        raise ValueError("Matrix has not completed; never manufacture missing methods")
    parent = batch / f"seed-{seed}"
    roots = {m: parent / m for m in METHODS}
    for root in extra_runs:
        manifest = json.loads((root / "manifest.json").read_text())
        method = manifest.get("method")
        if (method != "OnlineTune" or method in roots or manifest.get("smoke")
                or not manifest.get("formal_protocol") or manifest.get("backend") != "onlinetune"):
            raise ValueError("Only completed formal dynamic OnlineTune can extend the measured comparison")
        roots[method] = root
    rows, windows, phases, completions, sources = [], [], [], [], {}
    schedule_hash = None
    reference_post = None
    reference_space = None
    for method, root in roots.items():
        s = json.loads((root / "summary.json").read_text())
        post_path = (parent / f"{method}-postcheck.json") if method in METHODS else root.parent / "postcheck.json"
        post = json.loads(post_path.read_text())
        manifest = json.loads((root / "manifest.json").read_text())
        if reference_post is None:
            reference_post = post
        if reference_space is None and manifest.get("space") is not None:
            reference_space = manifest["space"]
        if method not in METHODS and (
                post.get("logical_checksums") != reference_post.get("logical_checksums")
                or post.get("config_sha256") != reference_post.get("config_sha256")
                or manifest.get("space") != reference_space):
            raise ValueError("OnlineTune data, initial configuration or hard catalog differs")
        if not s["complete"] or s.get("error") or not post["completed"]:
            raise ValueError(f"{method}: incomplete or failed postcheck")
        if not s.get("schedule_hash"):
            raise ValueError("Missing request-plan hash")
        schedule_hash = schedule_hash or s["schedule_hash"]
        if schedule_hash != s["schedule_hash"]:
            raise ValueError("Request schedules differ")
        raw = root / "requests.jsonl"
        metrics, ws, ps, cs = service_metrics(read_jsonl(raw))
        usage = json.loads((root / "llm_usage.json").read_text()) if method != "Default" else {}
        trials_path = root / "trials.jsonl"
        trials = read_jsonl(trials_path) if trials_path.exists() else []
        rows.append({"method": method, **metrics, "valid_observations": s.get("valid_observations", 0),
                     **configuration_counts(trials, fixed_default=method == "Default"),
                     "trial_slots_used": s.get("budget", {}).get("trial_slots_used", 0),
                     "failed_config_observations": sum(v for k, v in s.get("trial_statuses", {}).items()
                                                       if k != "measured"),
                     "llm_calls": s.get("llm_calls", 0), "tokens": usage.get("total_tokens", 0),
                     "failed": s["counts"]["failed"], "dropped": s["counts"]["dropped"],
                     "control_flow_completed": s.get("control_flow_completed"),
                     "closed_loop_completed": s.get("closed_loop_completed")})
        windows.extend({"method": method, **w} for w in ws)
        phases.extend({"method": method, **p} for p in ps)
        completions.extend({"method": method, **c} for c in cs)
        sources[method] = {"requests_sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
                           "schedule_hash": schedule_hash, "actual_source": str(root),
                           "source_revision": manifest.get("git_revision"),
                           "hardware_id": manifest.get("environment", {}).get("hardware_id"),
                           "adapter": manifest.get("onlinetune_adapter")}
    default = rows[0]["overall_p95_ms"]
    for row in rows:
        row["p95_reduction_vs_default_pct"] = 100 * (1 - row["overall_p95_ms"] / default)
    attainments = []
    for i, phase in enumerate(PHASES):
        reference = [w["p95_ms"] for w in windows if w["method"] == "Default" and w["phase"] == phase]
        threshold = .9 * float(np.median(reference))
        for method in roots:
            ws = [w for w in windows if w["method"] == method and w["phase"] == phase]
            attainments.append({"method": method, "phase": phase, "target_p95_ms": threshold,
                                "attainment_seconds": adaptation(ws, i * 600, threshold)})
    return rows, windows, phases, completions, attainments, sources


def plot(batch, output, seed=43, extra_runs=()):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    rows, windows, phases, completions, attainments, sources = collect(batch, seed, extra_runs)
    methods = tuple(r["method"] for r in rows)
    display = {m: m for m in methods}
    if "OnlineTune" in methods:
        display["OnlineTune"] = "OnlineTune (cold-start adapter)"
    colors = (*COLORS, "#2ca02c")[:len(methods)]
    width = .75 / len(methods)
    output.mkdir(parents=True, exist_ok=False)
    for name, data in (("overall_metrics", rows), ("window_metrics", windows), ("phase_metrics", phases),
                       ("successful_request_curves", completions), ("adaptation_metrics", attainments)):
        write_csv(output / f"{name}.csv", data, list(data[0]))
    atomic_json(output / "provenance.json", {"seed": seed, "independent_repetitions": 1, "sources": sources,
                "all_requests_included": True, "window_seconds": 20,
                "attainment": "three complete 20s cohorts at <=.9*paired Default median window P95",
                "latency_origin": "completed minus emitted; end-to-end including queue and transitions",
                "note": "No literature estimates, smoothing or invented error bands"})
    plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 10, "pdf.fonttype": 42})

    def save(fig, name, top=1):
        for ax in fig.axes:
            ax.grid(alpha=.18, linewidth=.5)
            ax.tick_params(direction="in", top=True, right=True)
            ax.set_axisbelow(True)
        fig.tight_layout(rect=(0, .10 if "OnlineTune" in methods else .065, 1, top))
        note = f"Paired request seed {seed}; one run per method."
        if "OnlineTune" in methods:
            online = next(r for r in rows if r["method"] == "OnlineTune")
            note += ("\nOnlineTune: cold-start dynamic adapter; only one configuration measured."
                     if online["unique_measured_configurations"] == 1
                     else "\nOnlineTune: dynamic adapter with observed SQL features.")
        fig.text(.5, .015, note, ha="center", fontsize=8)
        for ext in ("png", "pdf"):
            fig.savefig(output / f"{name}.{ext}", dpi=280, bbox_inches="tight")
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for index, (method, color) in enumerate(zip(methods, colors)):
        ws = [w for w in windows if w["method"] == method]
        axes[0].plot([w["end_s"] / 60 for w in ws], [w["p95_ms"] for w in ws],
                     color=color, lw=1.2, ls="--" if method == "Default" else "-", label=display[method])
        ps = [p for p in phases if p["method"] == method]
        axes[1].bar(np.arange(3) + (index - (len(methods) - 1) / 2) * width,
                    [p["p95_ms"] for p in ps], width=width * .95, color=color, label=method)
    for at in (10, 20):
        axes[0].axvline(at, color="#555555", ls=":", lw=.8)
    axes[0].set(xlabel="Workload time (min)", ylabel="P95 latency (ms; 20 s cohorts)",
                xlim=(0, 30), title="(a) Continuous service")
    axes[1].set(xticks=range(3), xticklabels=PHASES, ylabel="P95 latency (ms; whole phase)",
                title="(b) Whole-phase performance")
    if "OnlineTune" in methods:
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, frameon=False, fontsize=8, ncol=3,
                   loc="upper center", bbox_to_anchor=(.5, 1))
        save(fig, "dynamic_performance", top=.83)
    else:
        axes[0].legend(frameon=False, fontsize=8, ncol=2)
        save(fig, "dynamic_performance")

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
    for ax, key, title in zip(axes, ("trial_slots_used", "llm_calls", "tokens"),
                               ("Trial slots (including failures)", "LLM API calls", "LLM tokens")):
        ax.bar(range(len(methods)), [r[key] for r in rows], color=colors)
        ax.set(xticks=range(len(methods)), xticklabels=methods, title=title, ylim=(0, None))
        ax.tick_params(axis="x", labelrotation=25)
    save(fig, "tuning_overhead")

    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    for index, (method, color) in enumerate(zip(methods, colors)):
        rs = [r for r in attainments if r["method"] == method]
        for i, r in enumerate(rs):
            x = i + (index - (len(methods) - 1) / 2) * width
            if r["attainment_seconds"] is None:
                ax.text(x, 10, "NA", ha="center", color=color, fontsize=8)
            else:
                ax.bar(x, r["attainment_seconds"], width=width * .95, color=color,
                       label=method if i == 0 else None)
    ax.set(xticks=range(3), xticklabels=PHASES, xlim=(-.55, 2.55), ylim=(0, 660),
           ylabel="Target confirmation time (s)", title="First sustained target attainment")
    ax.legend(handles=[Patch(color=c, label=display[m]) for m, c in zip(methods, colors)],
              frameon=False, fontsize=8, ncol=2)
    save(fig, "adaptation")
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batch", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=43)
    p.add_argument("--extra-run", type=Path, nargs="*", default=[])
    a = p.parse_args()
    print(json.dumps(plot(a.batch, a.output, a.seed, a.extra_run), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
