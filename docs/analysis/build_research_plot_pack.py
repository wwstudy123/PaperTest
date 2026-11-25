"""汇总现有真实数据可支持的科研图包；缺失实验明确留空，不补造基线或消融。"""
from pathlib import Path
import csv
import hashlib
import json
import zipfile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT/"docs/experiment_data"
OUT = DATA/"research_plot_pack_20261126"
OUT.mkdir(parents=True, exist_ok=True)
FIG = ROOT/"docs/figures"


def load(path):
    return json.loads(path.read_text())


def csv_rows(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def save_csv(name, rows):
    with (OUT/name).open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader(); w.writerows(rows)


plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 10, "pdf.fonttype": 42})
timed = DATA/"timed_static_100min_002"
v = load(timed/"verification.json")
assert v["verified"]
own = csv_rows(timed/"admtune_trials.csv")
literature = csv_rows(timed/"literature_curves.csv")
fig, ax = plt.subplots(figsize=(8.2, 4.8))
ax.plot([float(r["active_elapsed_seconds"])/60 for r in own],
        [float(r["best_found_tps"]) for r in own], "-v", color="#1f77b4", lw=1.4,
        ms=4, markevery=max(1, len(own)//9), label="ADMTune (TPS prototype)")
ax.scatter(float(own[-1]["active_elapsed_seconds"])/60, float(own[-1]["tps"]),
           marker="*", s=85, color="#1f77b4", zorder=6)
endpoints = [{
    "method": "ADMTune (TPS prototype)", "source": "local measured",
    "endpoint_tps": float(own[-1]["tps"]), "source_baseline_tps": float(own[0]["tps"]),
    "endpoint_kind": "independent final verification", "same_protocol_across_entire_table": False}]
for method, color, marker in [
        ("AgenticDB", "#d62728", "o"), ("AgentTune", "#ff7f0e", "D"),
        ("GPTuner", "#9467bd", "P"), ("CDBTune", "#2ca02c", "s")]:
    rows = [r for r in literature if r["method"] == method and r["metric"] == "throughput"]
    assert rows and all(r["unit"] == "TPS" for r in rows)
    ax.plot([float(r["time_min"]) for r in rows], [float(r["value"]) for r in rows],
            color=color, marker=marker, ms=3, markevery=max(1, len(rows)//8), lw=1,
            label=method+" (published)")
    endpoints.append({
        "method": method, "source": "AgenticDB Fig5 MS Sysbench Read; vector extraction",
        "endpoint_tps": float(rows[-1]["value"]), "source_baseline_tps": float(rows[0]["value"]),
        "endpoint_kind": "published curve endpoint", "same_protocol_across_entire_table": False})
ax.axhline(float(own[0]["tps"]), color="#777777", ls="--", lw=1, label="Default (local)")
endpoints.append({"method": "Default", "source": "local measured initial configuration",
                  "endpoint_tps": float(own[0]["tps"]), "source_baseline_tps": float(own[0]["tps"]),
                  "endpoint_kind": "initial reference", "same_protocol_across_entire_table": False})
ax.set_xlabel("Tuning time (min)"); ax.set_ylabel("Throughput (TPS)")
ax.set_xlim(0, 100); ax.set_ylim(bottom=0)
ax.grid(alpha=.25, lw=.6); ax.tick_params(direction="in", top=True, right=True)
ax.set_title("Cross-study reference — not a controlled ranking", pad=42)
ax.legend(loc="lower center", bbox_to_anchor=(.5, 1.015), ncol=3, fontsize=8, frameon=False)
fig.subplots_adjust(top=.74, bottom=.24, left=.11, right=.98)
fig.text(.5, .035, "Local ADMTune/Default are measured; other methods are published curves with different protocols.\n"
         "The local curve is a TPS-only prototype, not the latest joint-objective method.", ha="center", fontsize=8)
for ext in (".png", ".pdf"):
    fig.savefig((FIG/"six_method_reference_20261126").with_suffix(ext), dpi=280, bbox_inches="tight")
plt.close(fig)
save_csv("published_reference_endpoints.csv", endpoints)

overall = [{
    "run": "timed-static-100min-002", "version": "ADMTune TPS prototype",
    "optimization_objective": "TPS maximize", "baseline_tps": v["baseline_tps"],
    "baseline_p95_ms": v["baseline_p95_ms"], "final_tps": v["verification_tps"],
    "final_p95_ms": v["verification_p95_ms"], "valid_measurements": v["valid"],
    "llm_calls": v["llm_calls"], "actual_tokens": v["known_tokens"],
    "interrupted_slots": 0, "recovered": False, "accepted_configuration_changed": True}]
workflows = [("TPS prototype", v["llm_calls"], v["known_tokens"])]
for run, label in [
        ("composite_pilot_002", "Joint pilot"),
        ("composite_pair_legacy_001", "Legacy GP"),
        ("composite_pair_repeat_001", "Repeat GP\n(recovered)")]:
    d = load(DATA/run/"verification.json"); assert d["verified"]
    overall.append({
        "run": run.replace("_", "-"), "version": label.replace("\n", " "),
        "optimization_objective": "TPS/P95 maximize", "baseline_tps": d["baseline_tps"],
        "baseline_p95_ms": d["baseline_p95_ms"], "final_tps": d["last_tps"],
        "final_p95_ms": d["last_p95_ms"], "valid_measurements": d["valid"],
        "llm_calls": d["llm_calls"], "actual_tokens": d["known_tokens"],
        "interrupted_slots": d.get("invalid_attempts", 0), "recovered": bool(d.get("recovery")),
        "accepted_configuration_changed": bool(d["accepted_at"])})
    workflows.append((label, d["llm_calls"], d["known_tokens"]))
save_csv("overall_measured.csv", overall)
table = ["# 本地实测终点表（不同预算/目标，不作方法排名）", "",
         "| 版本 | 优化目标 | 初始TPS | 最终TPS | 初始P95(ms) | 最终P95(ms) | 有效测量 | 实际API/tokens |",
         "|---|---|---:|---:|---:|---:|---:|---:|"]
for row in overall:
    table.append(f"| {row['version']} | {row['optimization_objective']} | {row['baseline_tps']:.2f} | "
                 f"{row['final_tps']:.2f} | {row['baseline_p95_ms']:.2f} | {row['final_p95_ms']:.2f} | "
                 f"{row['valid_measurements']} | {row['llm_calls']}/{row['actual_tokens']} |")
table += ["", "恢复臂最后仍是初始配置；终点变动不能当作调优收益。文献数值单独存放在published_reference_endpoints.csv，不能拿本地Default给文献方法计算统一提升率。"]
(OUT/"overall_measured.md").write_text("\n".join(table)+"\n")
fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8))
for ax, idx, title, ylabel in zip(axes, [1, 2], ["(a) Actual API calls", "(b) Actual tokens"],
                                 ["API calls", "Tokens"]):
    vals = [row[idx] for row in workflows]
    bars = ax.bar(range(len(workflows)), vals, color=["#777777", "#1f77b4", "#ff7f0e", "#d62728"], width=.6)
    ax.bar_label(bars, labels=[f"{n:,}" for n in vals], fontsize=8, padding=3)
    ax.set_xticks(range(len(workflows)), [
        row[0].replace("TPS prototype", "TPS\nprototype").replace("Joint pilot", "Joint\npilot")
        .replace("Legacy GP", "Legacy\nGP") for row in workflows])
    ax.set_ylabel(ylabel); ax.set_title(title); ax.set_ylim(0, max(vals)*1.2)
    ax.grid(axis="y", alpha=.25); ax.set_axisbelow(True)
fig.subplots_adjust(bottom=.28, wspace=.3)
fig.text(.5, .025, "Different targets and budgets: descriptive provider usage, not an efficiency ranking. "
         "Recovered run retains pre-interruption usage.", ha="center", fontsize=8)
for ext in (".png", ".pdf"):
    fig.savefig((FIG/"llm_overhead_20261126").with_suffix(ext), dpi=280, bbox_inches="tight")
plt.close(fig)

figures = [
    ("01", "真实静态调优轨迹", "admtune_100min_measured", "timed_static_100min_002", "TPS旧原型，71次真实观测；非新综合目标"),
    ("02", "六方法文献参照", "six_method_reference_20261126", "timed_static_100min_002", "四条文献曲线；不同协议，非同条件排名"),
    ("03", "吞吐与尾延迟诊断", "static_diagnostic_001", "static_diagnostic_001", "9次固定配置测量，非算法搜索"),
    ("04", "固定请求强度复验", "rate_diagnostic_001", "rate_diagnostic_001", "1000TPS限速；不可拼接不限速TPS"),
    ("05", "综合目标闭环", "composite_pilot_002", "composite_pilot_002", "真实LLM，未发现优于初始配置的分数"),
    ("06", "重复与区间失配", "uncertainty_repeat_001", "uncertainty_repeat_001", "原固定6/3留出负结果保留"),
    ("07", "新GP顺序预测验收", "gp_noise_validation_001", "gp_noise_validation_001", "真实测量前封存，只有3个检验点"),
    ("08", "GP闭环先导记录", "composite_gp_pair_001", "composite_gp_pair_001", "新GP臂恢复后补最终复测，非不中断对照"),
    ("09", "LLM实测开销", "llm_overhead_20261126", "research_plot_pack_20261126", "真实调用和tokens；不猜测费用"),
    ("10", "负载区分度校准", "discrimination_001", "discrimination_001", "热点/均匀分布校准，非动态方法优势"),
]
save_csv("figure_index.csv", [
    {"id": n, "title": title, "figure": name, "dataset": dataset, "scope": scope}
    for n, title, name, dataset, scope in figures])
gaps = [
    {"requirement": "六方法同协议整体性能主图", "status": "尚未完成",
     "remaining": "AgenticDB、AgentTune、GPTuner、CDBTune需按真实实现同协议重跑，不能用文献曲线冒充"},
    {"requirement": "多种子独立重复与误差带", "status": "尚未完成",
     "remaining": "现有轨迹主要为单次先导；需独立重置并重复，不对现有曲线虚构误差带"},
    {"requirement": "第二基准负载（例如TPC-H）", "status": "尚未完成",
     "remaining": "完整数据、查询执行与指标协议尚需验收；TPC-DS/JOB并非必须全部添加"},
    {"requirement": "动态阶段与适应耗时", "status": "已有早期实机验收，正式矩阵未完成",
     "remaining": "须将新协议接到真实动态闭环并重复，不能用静态常量上下文证明动态建模"},
    {"requirement": "四模块真实消融", "status": "尚未完成",
     "remaining": "多时间片、历史复用、代理预筛和反思需同预算真实消融；模拟矩阵不冒充实测"},
    {"requirement": "当前可用先导图、总表和来源", "status": "已汇总",
     "remaining": "图包含10组图、CSV/JSON数据及报告；不能等同于完整论文实验结论"},
]
save_csv("requirements_status.csv", gaps)
index = ["# ADMTune科研绘图包（2026-11-26）", "",
         "已整理当前真实记录能支持的10组PNG/PDF及数据源。**这是可用先导图包，不是整套正式科研实验已完成。**", "",
         "## 图目录", "", "| 编号 | 图 | 可用范围 |", "|---|---|---|"]
for n, title, name, dataset, scope in figures:
    index.append(f"| {n} | [{title}](figures/{name}.pdf) | {scope} |")
index += ["", "## 表格与原始数值", "",
          "[本地实测总体表](experiment_data/research_plot_pack_20261126/overall_measured.md)",
          "；CSV同时提供，文献终点单独存放，不能计算统一基线增益。",
          "", "## 未完成的正式图", ""]
for gap in gaps[:-1]:
    index.append(f"- **{gap['requirement']}**：{gap['remaining']}。")
index += ["", "## 使用边界", "",
          "- 新GP臂第7槽位中断，未填性能值；第8槽位为恢复后的独立复测，未重新搜索或训练中断前模型。",
          "- 两种综合闭环均未接受新配置，恢复后默认配置分数变化不等于调优增益。",
          "- 100分钟图是旧TPS目标；不要改成已经按TPS/P95优化或完整动态ADMTune。",
          "- 原始私有归档含配置凭证，不放入图包；图包只含公开绘图数据、报告和图。",
          "- 矢量图按AgentTune类似的细线、衬线字体与向内刻度绘制；不补造100分钟终点或置信带。",
          "", "## 尽快补齐的顺序", "",
          "先冻结正式目标和预算、完成可运行的同协议真实基线，再做独立重复及动态/消融。当前GP后没有新候选审计，先验证预筛误拒与请求上下文开销，不要只重复噪声校准或继续堆100分钟先导。",
          "", "脚本：`docs/analysis/build_research_plot_pack.py`；归档核验需另有私有原始记录。"]
index_path = ROOT/"docs/PLOT_PACKAGE_20261126.md"
index_path.write_text("\n".join(index)+"\n")
(OUT/"checksums.json").write_text(json.dumps({
    p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir()
    if p.is_file() and p.name != "checksums.json"}, indent=2)+"\n")
exports = ROOT/"docs/exports"; exports.mkdir(exist_ok=True)
dest = exports/"admtune_plot_pack_20261126.zip"
assets = {index_path}
for _, _, name, _, _ in figures:
    for ext in (".png", ".pdf"):
        asset = (FIG/name).with_suffix(ext); assert asset.is_file(); assets.add(asset)
datasets = {f[3] for f in figures} | {"composite_pair_legacy_001", "composite_pair_repeat_001"}
for dataset in datasets:
    assets.update(p for p in (DATA/dataset).glob("*") if p.is_file() and p.suffix in {".csv", ".json", ".md"})
for name in [
        "TIMED_STATIC_100MIN_RESULTS_20261125.md", "STATIC_DIAGNOSTIC_RESULTS_20261125.md",
        "FIXED_RATE_RESULTS_20261125.md", "COMPOSITE_PILOT_002_RESULTS_20261125.md",
        "UNCERTAINTY_REPEAT_RESULTS_20261125.md", "GP_NOISE_VALIDATION_RESULTS_20261125.md",
        "COMPOSITE_GP_PAIR_RESULTS_20261126.md", "DYNAMIC_LLM_002_20261123.md"]:
    path = ROOT/"docs"/name
    if path.is_file():
        assets.add(path)
inventory = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(assets)}
with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as archive:
    archive.writestr("README.md", "科研图包入口：docs/PLOT_PACKAGE_20261126.md。\n含10组图及公开数据，不含私有配置。正式实验缺口见入口文档。\n")
    for path in sorted(assets):
        archive.write(path, str(path.relative_to(ROOT)))
    archive.writestr("manifest.json", json.dumps(inventory, ensure_ascii=False, indent=2))
digest = hashlib.sha256(dest.read_bytes()).hexdigest()
dest.with_suffix(".zip.sha256").write_text(f"{digest}  {dest.name}\n")
print(json.dumps({"zip": str(dest), "files": len(inventory)+2, "bytes": dest.stat().st_size,
                  "sha256": digest}, ensure_ascii=False, indent=2))
