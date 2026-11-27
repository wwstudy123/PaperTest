"""将已完成的真实实验整理为核心图包；不调用数据库或LLM。"""
import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

from tools.plot_dynamic_matrix import plot


def build(repo, output, batch):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if output.exists():
        raise FileExistsError(f"请使用新输出目录，避免覆盖既有证据：{output}")
    output.mkdir(parents=True)
    plot(batch, output / "01_dynamic")
    sources = {
        "02_static": "static_quick_20261127/figures_and_data",
        "03_a1_diagnosis": "admtune_a1_diagnosis_20261128",
        "04_reflection_input": "reflection_input_audit_20261128",
        "05_onlinetune_diagnostic": "onlinetune_dynamic_r0_20261128",
    }
    data = repo / "docs/experiment_data"
    for destination, source in sources.items():
        shutil.copytree(data / source, output / destination)

    plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 10, "pdf.fonttype": 42})
    colors = ["#777777", "#1f77b4", "#ff7f0e", "#d62728"]
    static = list(csv.DictReader((output / "02_static/measured_performance.csv").open()))
    dynamic = list(csv.DictReader((output / "01_dynamic/overall_metrics.csv").open()))
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
    labels = ["Default", "AgenticDB\n(DB-only)", "AgentTune", "ADMTune"]
    for ax, rows, key, title, ylabel in (
        (axes[0], static, "tps", "(a) Static throughput", "TPS (higher is better)"),
        (axes[1], static, "p95_latency_ms", "(b) Static latency", "P95 latency (ms)"),
        (axes[2], dynamic, "overall_p95_ms", "(c) Dynamic A-B-A latency", "P95 latency (ms)"),
    ):
        values = [float(row[key]) for row in rows]
        ax.bar(range(4), values, color=colors)
        for i, value in enumerate(values):
            ax.text(i, value, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
        ax.set(xticks=range(4), xticklabels=labels, title=title, ylabel=ylabel,
               ylim=(0, max(values) * 1.18))
        ax.tick_params(axis="x", labelsize=8)
        ax.grid(axis="y", alpha=.18)
        ax.set_axisbelow(True)
    fig.tight_layout(rect=(0, .09, 1, 1))
    fig.text(.5, .015, "Separate static and dynamic protocols; one run per tuned method; no pooled repeats.",
             ha="center", fontsize=8)
    for ext in ("png", "pdf"):
        fig.savefig(output / f"overall_performance.{ext}", dpi=280, bbox_inches="tight")
    plt.close(fig)

    audit = json.loads((output / "04_reflection_input/summary.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.4))
    for ax, key, title in zip(axes, ("record_appearances", "input_bytes"),
                              ("Record appearances", "Serialized input bytes")):
        values = [audit["totals"][mode][key] for mode in ("full", "incremental")]
        ax.bar(["Full history", "Incremental"], values, color=["#777777", "#d62728"])
        for i, value in enumerate(values):
            ax.text(i, value, str(value), ha="center", va="bottom")
        ax.set(title=title, ylim=(0, max(values) * 1.2))
    fig.tight_layout(rect=(0, .12, 1, 1))
    fig.text(.5, .015, "Offline input-selection audit only; not measured tokens, cost or reflection quality.",
             ha="center", fontsize=8)
    for ext in ("png", "pdf"):
        fig.savefig(output / f"04_reflection_input/input_volume.{ext}", dpi=280, bbox_inches="tight")
    plt.close(fig)

    notes = {
        "静态实验": "STATIC_QUICK_RESULTS_20261128.md",
        "动态协议": "DYNAMIC_MATRIX_PROTOCOL_20261126.md",
        "动态结果": "DYNAMIC_MATRIX_PROGRESS_20261127.md",
        "首次适应诊断": "ADMTUNE_A1_DIAGNOSIS_20261128.md",
        "OnlineTune诊断": "ONLINETUNE_DYNAMIC_RESULTS_20261128.md",
    }
    (output / "说明").mkdir()
    for name, source in notes.items():
        shutil.copy2(repo / "docs" / source, output / "说明" / f"{name}.md")
    (output / "图表目录.md").write_text(
        "# ADMTune现有实测核心图包\n\n"
        "本图包离线整理既有结果，新增数据库试验0次、LLM调用0次。"
        "一次运行不代表多种子统计结论；本图包不是全部论文实验已完成的声明。\n\n"
        "| 图 | 文件 | 用途 |\n|---|---|---|\n"
        "| 整体性能 | overall_performance | 静态TPS、静态P95、动态P95；不同协议分面展示 |\n"
        "| 动态过程及分阶段 | 01_dynamic/dynamic_performance | 四方法真实20秒窗口及整阶段P95 |\n"
        "| 达标耗时 | 01_dynamic/adaptation | 连续三个窗口达标；NA不是0秒 |\n"
        "| 调优开销 | 01_dynamic/tuning_overhead | 槽位含失败、实际调用、token |\n"
        "| 静态已接受配置过程 | 02_static/measured_comparison | 组合指标、TPS和延迟 |\n"
        "| 静态全部候选 | 02_static/raw_measurement_comparison | 包括未接受配置的真实观测 |\n"
        "| 首次适应慢诊断 | 03_a1_diagnosis/a1_diagnosis | 原运行失败与控制时间线 |\n"
        "| 增量输入审计 | 04_reflection_input/input_volume | 输入字节及重复处理，非质量/费用消融 |\n"
        "| OnlineTune诊断 | 05_onlinetune_diagnostic/five_method_preview/ | "
        "三图；冷启动仅测默认配置，不作为充分复现的正式排名 |\n\n"
        "每幅图均提供PNG和PDF。对应CSV/JSON为绘图数值；manifest.json记录文件SHA256。\n\n"
        "## 当前结论\n\n"
        "- 动态首轮：ADMTune全程P95为59.50ms，Default为72.42ms，"
        "AgenticDB为53.83ms，AgentTune为56.15ms；ADMTune不是最优。\n"
        "- 静态末端：ADMTune的P95为17.01ms，TPS为2473.74；"
        "组合值较Default高9.13%，但存在运行波动，不能直接推出稳定优势。\n"
        "- 新版本父配置缓存及反思输入修复尚未完成真实性能复测；不以旧图冒充修复后的结果。\n\n"
        "## 尚缺实验（未在本图包补造）\n\n"
        "冻结修复版后的独立复测、多种子动态配对重复、四项真实消融、"
        "OnlineTune冷启动探索核查及TPC-H补充。GPTuner/CDBTune未实测，不绘制冒充实测的曲线。\n",
        encoding="utf-8")
    manifest = {
        "database_calls_added": 0, "llm_calls_added": 0,
        "scope": "existing measured results and explicitly separated offline diagnostics",
        "files": {
            str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(output.rglob("*")) if p.is_file()
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    archive = repo / "docs/exports" / f"{output.name}.zip"
    with ZipFile(archive, "w", ZIP_DEFLATED) as z:
        for p in sorted(output.rglob("*")):
            if p.is_file():
                z.write(p, str(Path(output.name) / p.relative_to(output)))
    return {"output": str(output), "archive": str(archive),
            "png_figures": len(list(output.rglob("*.png")))}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(Path(__file__).resolve().parents[1], args.output, args.batch),
                     ensure_ascii=False, indent=2))
