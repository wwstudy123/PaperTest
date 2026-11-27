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
COLORS = ("#777777", "#1f77b4", "#2ca02c", "#d62728")


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


def collect(batch, seed):
    matrix = json.loads((batch / "matrix.json").read_text())
    if matrix["status"] != "completed":
        raise ValueError("Matrix has not completed; never manufacture missing methods")
    parent = batch / f"seed-{seed}"
    rows, windows, phases, completions, sources = [], [], [], [], {}
    schedule_hash = None
    for method in METHODS:
        root = parent / method
        s = json.loads((root / "summary.json").read_text())
        post = json.loads((parent / f"{method}-postcheck.json").read_text())
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
        rows.append({"method": method, **metrics, "valid_observations": s.get("valid_observations", 0),
                     "llm_calls": s.get("llm_calls", 0), "tokens": usage.get("total_tokens", 0),
                     "failed": s["counts"]["failed"], "dropped": s["counts"]["dropped"],
                     "control_flow_completed": s.get("control_flow_completed"),
                     "closed_loop_completed": s.get("closed_loop_completed")})
        windows.extend({"method": method, **w} for w in ws)
        phases.extend({"method": method, **p} for p in ps)
        completions.extend({"method": method, **c} for c in cs)
        sources[method] = {"requests_sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
                           "schedule_hash": schedule_hash}
    default = rows[0]["overall_p95_ms"]
    for row in rows:
        row["p95_reduction_vs_default_pct"] = 100 * (1 - row["overall_p95_ms"] / default)
    attainments = []
    for i, phase in enumerate(PHASES):
        reference = [w["p95_ms"] for w in windows if w["method"] == "Default" and w["phase"] == phase]
        threshold = .9 * float(np.median(reference))
        for method in METHODS:
            ws = [w for w in windows if w["method"] == method and w["phase"] == phase]
            attainments.append({"method": method, "phase": phase, "target_p95_ms": threshold,
                                "attainment_seconds": adaptation(ws, i * 600, threshold)})
    return rows, windows, phases, completions, attainments, sources


def plot(batch, output, seed=43):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    rows, windows, phases, completions, attainments, sources = collect(batch, seed)
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

    def save(fig, name):
        for ax in fig.axes:
            ax.grid(alpha=.18, linewidth=.5)
            ax.tick_params(direction="in", top=True, right=True)
            ax.set_axisbelow(True)
        fig.tight_layout(rect=(0, .065, 1, 1))
        fig.text(.5, .015, f"Paired request seed {seed}; one run per method.", ha="center", fontsize=8)
        for ext in ("png", "pdf"):
            fig.savefig(output / f"{name}.{ext}", dpi=280, bbox_inches="tight")
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for index, (method, color) in enumerate(zip(METHODS, COLORS)):
        ws = [w for w in windows if w["method"] == method]
        axes[0].plot([w["end_s"] / 60 for w in ws], [w["p95_ms"] for w in ws],
                     color=color, lw=1.2, ls="--" if method == "Default" else "-", label=method)
        ps = [p for p in phases if p["method"] == method]
        axes[1].bar(np.arange(3) + (index - 1.5) * .19, [p["p95_ms"] for p in ps],
                    width=.18, color=color, label=method)
    for at in (10, 20):
        axes[0].axvline(at, color="#555555", ls=":", lw=.8)
    axes[0].set(xlabel="Workload time (min)", ylabel="P95 latency (ms; 20 s cohorts)",
                xlim=(0, 30), title="(a) Continuous service")
    axes[1].set(xticks=range(3), xticklabels=PHASES, ylabel="P95 latency (ms; whole phase)",
                title="(b) Whole-phase performance")
    axes[0].legend(frameon=False, fontsize=8, ncol=2)
    save(fig, "dynamic_performance")

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
    for ax, key, title in zip(axes, ("valid_observations", "llm_calls", "tokens"),
                               ("Real configuration observations", "LLM API calls", "LLM tokens")):
        ax.bar(range(4), [r[key] for r in rows], color=COLORS)
        ax.set(xticks=range(4), xticklabels=METHODS, title=title, ylim=(0, None))
        ax.tick_params(axis="x", labelrotation=25)
    save(fig, "tuning_overhead")

    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    for index, (method, color) in enumerate(zip(METHODS, COLORS)):
        rs = [r for r in attainments if r["method"] == method]
        for i, r in enumerate(rs):
            x = i + (index - 1.5) * .19
            if r["attainment_seconds"] is None:
                ax.text(x, 10, "NA", ha="center", color=color, fontsize=8)
            else:
                ax.bar(x, r["attainment_seconds"], width=.18, color=color,
                       label=method if i == 0 else None)
    ax.set(xticks=range(3), xticklabels=PHASES, xlim=(-.55, 2.55), ylim=(0, 660),
           ylabel="Target confirmation time (s)", title="First sustained target attainment")
    ax.legend(handles=[Patch(color=c, label=m) for m, c in zip(METHODS, COLORS)],
              frameon=False, fontsize=8, ncol=2)
    save(fig, "adaptation")
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batch", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=43)
    a = p.parse_args()
    print(json.dumps(plot(a.batch, a.output, a.seed), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
