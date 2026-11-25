"""绘制本轮真实综合目标闭环；不拼接旧目标、限速实验或论文基线。"""
from pathlib import Path
import csv
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
data = ROOT / "docs/experiment_data/composite_pilot_001"
verified = json.loads((data / "verification.json").read_text())
assert verified["verified"] and verified["objective"] == "tps_per_p95:maximize"
if not verified["independent_verification_available"]:
    raise SystemExit("本轮尚无独立复测，不生成可能被误认作完整调优结果的性能曲线。")
rows = list(csv.DictReader((data / "observations.csv").open()))
rows = [r for r in rows if r["valid"] == "True"]
plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 10, "pdf.fonttype": 42})
fig, axes = plt.subplots(1, 3, figsize=(11.6, 3.65))
x = [float(r["active_elapsed_minutes"]) for r in rows]
for ax, field, title, ylabel in zip(
        axes, ["objective_value", "tps", "p95_ms"],
        ["(a) Joint objective", "(b) Throughput", "(c) Tail latency"],
        ["TPS / P95 (ms)", "Throughput (TPS)", "Latency (ms)"]):
    y = [float(r[field]) for r in rows]
    ax.plot(x, y, "-o", color="#1f77b4", lw=1.2, ms=4, label="Measured")
    ax.axhline(y[0], color="#777777", ls="--", lw=1, label="Initial observation")
    if field == "objective_value":
        ax.step(x, [float(r["accepted_best_objective"]) for r in rows],
                where="post", color="#d62728", lw=1.3, label="Accepted search best")
    for r, a, b in zip(rows, x, y):
        if r["kind"] == "verification":
            ax.scatter(a, b, marker="*", s=100, color="#2ca02c", zorder=5,
                       label="Independent verification")
    ax.set_xlabel("Tuning time (min)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=.25, lw=.6)
    ax.tick_params(direction="in", top=True, right=True)
    ax.legend(fontsize=7, loc="best", frameon=False)
fig.subplots_adjust(bottom=.26, top=.85, wspace=.37)
fig.text(.5, .025,
         f"Real LLM pilot; {verified['trials']} measurements, "
         f"{verified['llm_calls']} API calls. Single run; raw observations, not multi-seed averages.",
         ha="center", fontsize=8)
dest = ROOT / "docs/figures/composite_pilot_001"
dest.parent.mkdir(exist_ok=True, parents=True)
for extension in (".png", ".pdf"):
    fig.savefig(dest.with_suffix(extension), dpi=280, bbox_inches="tight")
plt.close(fig)
print("Saved actual composite pilot figure")
