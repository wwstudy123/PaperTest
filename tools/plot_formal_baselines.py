"""一轮实测对比图；论文比例参考只用虚线，不进入实测性能表。"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np

from admtune.continuous import atomic_json
from admtune.replay_report import write_csv


STYLE = {"ADMTune": ("#d62728", "s"), "AgenticDB": ("#1f77b4", "o"),
         "AgentTune": ("#ff7f0e", "^"), "Default": ("#777777", None),
         "GPTuner": ("#9467bd", "D"), "CDBTune": ("#2ca02c", "v")}


def load(root):
    manifest = json.loads((root/"manifest.json").read_text())
    summary = json.loads((root/"summary.json").read_text())
    records = [json.loads(line) for line in (root/"trials.jsonl").read_text().splitlines()]
    if manifest["simulated"] or summary["status"] != "completed":
        raise ValueError(f"需要已完成的真实运行：{root}")
    valid = [r for r in records if r["status"] == "measured" and r["complete"]
             and r["effective"] and r["constraints_ok"] and r["value"] is not None]
    if not valid or valid[-1]["metadata"]["kind"] != "verification":
        raise ValueError("需要最终独立实测，不能以搜索极值替代")
    return manifest, summary, valid


def plot(run_dirs, output, literature=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    loaded = [load(p) for p in run_dirs]
    reference = loaded[0][0]
    for m, _, _ in loaded[1:]:
        for key in ("comparison_protocol", "benchmark", "space"):
            if m[key] != reference[key]:
                raise ValueError(f"不同协议不能直接组合：{key}")
    output.mkdir(parents=True, exist_ok=False)
    curves, table, evidence, observations = [], [], [], []
    default = next((r for m, _, r in loaded if m["method"] == "Default"), None)
    if default is None:
        raise ValueError("需要本次真实 Default 参考")
    default_values = {field: statistics.median(
        r["value"] if field == "objective" else r["metadata"][field] for r in default)
        for field in ("objective", "raw_tps", "p95_latency_ms")}
    horizon = max(r["metadata"]["active_elapsed_seconds"]/60 for _, _, rs in loaded for r in rs)
    for path, (m, s, rs) in zip(run_dirs, loaded):
        method, best = m["method"], rs[0]
        for r in rs:
            observations.append({"method": method, "repeat": m["repeat"], "trial_id": r["trial_id"],
                "trial": r["metadata"]["seq"], "kind": r["metadata"]["kind"], "config": r["config"],
                "objective": r["value"], "tps": r["metadata"]["raw_tps"],
                "p95_ms": r["metadata"]["p95_latency_ms"], "accepted": r["metadata"].get("accepted", False),
                "active_elapsed_seconds": r["metadata"]["active_elapsed_seconds"],
                "prediction": r.get("prediction")})
            if r["metadata"]["kind"] != "verification":
                if r["metadata"]["kind"] == "baseline" or r["metadata"].get("accepted"):
                    best = r
            value_source = r if r["metadata"]["kind"] == "verification" else best
            curves.append({"method": method, "source_type": "measured", "repeat": m["repeat"],
                "trial": r["metadata"]["seq"], "trial_id": r["trial_id"], "kind": r["metadata"]["kind"],
                "time_min": r["metadata"]["active_elapsed_seconds"]/60,
                "objective": value_source["value"], "tps": value_source["metadata"]["raw_tps"],
                "latency_ms": value_source["metadata"]["p95_latency_ms"],
                "raw_objective": r["value"], "raw_tps": r["metadata"]["raw_tps"],
                "raw_latency_ms": r["metadata"]["p95_latency_ms"],
                "value_trial_id": value_source["trial_id"]})
        final = rs[-1]
        values = default_values if method == "Default" else {
            "objective": final["value"], "raw_tps": final["metadata"]["raw_tps"],
            "p95_latency_ms": final["metadata"]["p95_latency_ms"]}
        table.append({"method": m.get("display_method", method), "source_type": "measured",
            "repeat": m["repeat"], "true_measurements": s["completed_trials"],
            "llm_calls": s["budget"]["llm_calls"], "llm_tokens": s["budget"]["llm_tokens_or_reserved"],
            "tps": values["raw_tps"], "p95_latency_ms": values["p95_latency_ms"],
            "objective_tps_over_p95": values["objective"],
            "objective_gain_vs_default_pct": 100*(values["objective"]/default_values["objective"]-1),
            "reported_value": "three-measurement median" if method == "Default" else "independent final measurement"})
        evidence.append({"method": method, "source_path": str(path),
                         "manifest_sha256": hashlib.sha256((path/"manifest.json").read_bytes()).hexdigest()})
    estimates = []
    if literature:
        with literature.open() as handle:
            rows = list(csv.DictReader(handle))
        # 保留论文原时间，不把100分钟压缩成短实测，也不凭偏好强制调整方法排名。
        for method in ("GPTuner", "CDBTune"):
            parts = {metric: [r for r in rows if r["method"] == method and r["metric"] == metric]
                     for metric in ("throughput", "latency")}
            grid = sorted({0., horizon} | {float(r["time_min"]) for r in parts["throughput"]
                                          if float(r["time_min"]) <= horizon})
            def projected(metric, baseline):
                rs = parts[metric]
                return baseline*np.interp(grid, [float(r["time_min"]) for r in rs],
                                         [float(r["value"]) for r in rs])/float(rs[0]["value"])
            tps = projected("throughput", default_values["raw_tps"])
            latency = projected("latency", default_values["p95_latency_ms"])
            for x, t, l in zip(grid, tps, latency):
                estimates.append({"method": method, "source_type": "literature_ratio_estimate",
                                  "time_min": x, "tps": t, "latency_ms": l, "objective": t/l,
                                  "formula": "local Default median * paper value(t) / paper value(0)"})
    write_csv(output/"measured_performance.csv", table, list(table[0]))
    write_csv(output/"measured_curves.csv", curves, list(curves[0]))
    (output/"raw_observations.jsonl").write_text("".join(
        json.dumps(r, ensure_ascii=False, allow_nan=False)+"\n" for r in observations))
    atomic_json(output/"protocol_and_sources.json", {
        "policy": reference["comparison_protocol"], "space": reference["space"],
        "sources": reference["sources"], "model": reference.get("model"),
        "performance_table_uses": "independent final measurement; Default: median of three actual measurements"})
    if estimates:
        write_csv(output/"literature_ratio_curves.csv", estimates, list(estimates[0]))
    atomic_json(output/"provenance.json", {"real_run_evidence": evidence,
        "literature_file": str(literature) if literature else None,
        "literature_sha256": hashlib.sha256(literature.read_bytes()).hexdigest() if literature else None,
        "literature_time_rescaled": False, "reference_curves_are_measurements": False})
    plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 10, "pdf.fonttype": 42,
                         "axes.linewidth": .8})
    def draw(include_reference, name, raw=False):
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
        for ax, field, title, ylabel in zip(axes,
                ("objective", "tps", "latency_ms"),
                ("(a) Joint performance", "(b) Throughput", "(c) Latency"),
                ("TPS / P95 (ms)", "Throughput (TPS)", "Latency (ms)")):
            for method in ("Default", "AgenticDB", "AgentTune", "ADMTune"):
                rs = [r for r in curves if r["method"] == method]
                if not rs:
                    continue
                color, marker = STYLE[method]
                label = "AgenticDB (DB-only)" if method == "AgenticDB" else method
                if method == "Default":
                    value = default_values[{"objective": "objective", "tps": "raw_tps",
                                            "latency_ms": "p95_latency_ms"}[field]]
                    ax.plot([0, horizon], [value, value], "--", color=color, lw=1.3, label=label)
                    continue
                search = [r for r in rs if r["kind"] != "verification"]
                yfield = "raw_"+field if raw else field
                ax.plot([r["time_min"] for r in search], [r[yfield] for r in search],
                        "-", color=color, marker=marker, ms=4, lw=1.4, label=label)
                final = rs[-1]
                ax.scatter(final["time_min"], final[field], marker=marker, s=45,
                           facecolors="none", edgecolors=color, linewidths=1.3, zorder=4)
            if include_reference:
                for method in ("GPTuner", "CDBTune"):
                    rs = [r for r in estimates if r["method"] == method]
                    if rs:
                        ax.plot([r["time_min"] for r in rs], [r[field] for r in rs],
                            "--", color=STYLE[method][0], lw=1.1,
                            label=method+" (paper-ratio estimate)")
            ax.set(xlabel="Active tuning time (min)", ylabel=ylabel, title=title, xlim=(0, horizon*1.04))
            ax.grid(alpha=.25, lw=.5); ax.tick_params(direction="in", top=True, right=True)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .98),
                   ncol=3, frameon=False, fontsize=8)
        footer = "One real run per method; hollow markers: independent final measurements."
        if raw:
            footer += "\nSearch markers show actual candidate measurements, including rejected configurations."
        if include_reference:
            footer += "\nDashed colored curves: published ratios anchored to local Default; not local experiments."
        fig.text(.5, .02, footer, ha="center", fontsize=8)
        fig.subplots_adjust(top=.75, bottom=.23, wspace=.35)
        for suffix in ("png", "pdf"):
            fig.savefig(output/f"{name}.{suffix}", dpi=260, bbox_inches="tight")
        plt.close(fig)
    draw(False, "measured_comparison")
    draw(False, "raw_measurement_comparison", raw=True)
    if estimates:
        draw(True, "comparison_with_paper_ratio_references")
    lines = ["# 单轮真实基线结果", "", "实线为实测配置的运行内已接受最优；空心点为最终独立复测。"
             "Default 为三次测量中位数。本轮不是跨重复均值。", "",
             "| 方法 | TPS | P95(ms) | TPS/P95 | 相对Default | 测量次数 | LLM调用 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for r in table:
        lines.append(f'| {r["method"]} | {r["tps"]:.2f} | {r["p95_latency_ms"]:.2f} | '
                     f'{r["objective_tps_over_p95"]:.2f} | {r["objective_gain_vs_default_pct"]:+.2f}% | '
                     f'{r["true_measurements"]} | {r["llm_calls"]} |')
    if estimates:
        lines += ["", "GPTuner、CDBTune 虚线仅按 AgenticDB 图5 MySQL Sysbench Read 的"
                  "原始时间与相对 Default 比例映射到本地尺度，不进入上表，不代表本机实测。"
                  "未压缩时间、未添加随机抖动或按预期强制排名。"]
    (output/"结果说明.md").write_text("\n".join(lines)+"\n")
    return table


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", type=Path, nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--literature", type=Path)
    a = p.parse_args()
    print(json.dumps(plot(a.runs, a.output, a.literature), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
