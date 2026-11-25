"""绘制固定配置重复及预先固定时间留出的观测区间；不是算法性能排名。"""
from pathlib import Path
import csv
import json
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
data = ROOT/"docs/experiment_data/uncertainty_repeat_001"
verification = json.loads((data/"verification.json").read_text())
assert verification["verified"] and verification["valid"] == 9 and verification["llm_calls"] == 0
assert verification["holdout_sequences"] == [7, 8, 9]
observations = list(csv.DictReader((data/"observations.csv").open()))
predictions = list(csv.DictReader((data/"holdout_predictions.csv").open()))
names = ["Default", "BufferOnly", "Concurrency8"]
labels = ["Default", "Buffer only", "Concurrency 8"]
plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 10, "pdf.fonttype": 42})
fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
ax = axes[0]
for name, label, color, marker in zip(names, labels, ["#777777", "#ff7f0e", "#1f77b4"], ["o", "s", "^"]):
    rows = sorted([r for r in observations if r["name"] == name], key=lambda r: int(r["block"]))
    ax.plot([int(r["block"]) for r in rows], [float(r["tps_per_p95"]) for r in rows],
            "-"+marker, color=color, lw=1.2, ms=4, label=label)
ax.axvspan(2.5, 3.3, color="#e8eef4", alpha=.65, zorder=-2)
ax.set_xlim(.7, 3.3)
ax.set_xticks([1, 2, 3], ["1 (cal.)", "2 (cal.)", "3 (test)"])
ax.set_ylim(bottom=0)
ax.set_title("(a) Repeated measurements")
ax.set_ylabel("TPS / P95 (ms)")
ax.set_xlabel("Measurement block")
ax.legend(fontsize=8, frameon=False)
for ax, model, title in zip(axes[1:], ["Frozen7", "Refit6"],
                            ["(b) Frozen parent GP", "(c) GP refit on six repeats"]):
    for i, name in enumerate(names):
        rows = {r["model"]: r for r in predictions if r["name"] == name}
        actual = float(rows[model]["actual_objective"])
        for j, (key, color, label) in enumerate([
                (model, "#1f77b4", "Original interval"),
                (model+"+floor", "#ff7f0e", "With noise floor")]):
            r = rows[key]
            mean, low, high = [float(r[k]) for k in ("predicted_objective", "lower", "upper")]
            assert all(math.isfinite(v) and v > 0 for v in (mean, low, high))
            ax.errorbar(i+(j-.5)*.24, mean, yerr=[[mean-low], [high-mean]],
                        fmt="o", color=color, capsize=3, lw=1.1, ms=4,
                        label=label if i == 0 else None, zorder=3)
        ax.scatter(i, actual, color="black", marker="x", s=30,
                   label="Held-out measurement" if i == 0 else None, zorder=5)
    ax.set_xticks(range(3), labels, rotation=12)
    ax.set_ylabel("TPS / P95 (ms), log scale")
    ax.set_yscale("log")
    ax.set_title(title)
    ax.legend(fontsize=7, frameon=False, loc="best")
for ax in axes:
    ax.grid(alpha=.25, lw=.6)
    ax.tick_params(direction="in", top=True, right=True)
fig.subplots_adjust(bottom=.27, top=.85, wspace=.36)
fig.text(.5, .025, "Zero LLM calls. Calibration: first six measurements; test: last three. "
         "Intervals use observation uncertainty; offline comparison.", ha="center", fontsize=8)
out = ROOT/"docs/figures/uncertainty_repeat_001"
for extension in (".png", ".pdf"):
    fig.savefig(out.with_suffix(extension), dpi=280, bbox_inches="tight")
plt.close(fig)
print("Saved repeat and held-out interval comparison")
