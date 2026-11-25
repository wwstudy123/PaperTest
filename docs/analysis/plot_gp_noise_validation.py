"""绘制独立顺序检验实测与测量前提交的两版GP预测，不冒充调优性能排名。"""
from pathlib import Path
import csv
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
data = ROOT/"docs/experiment_data/gp_noise_validation_001"
v = json.loads((data/"verification.json").read_text())
assert v["verified"] and v["prospective_submissions_verified"] and v["valid"] == 9 and v["llm_calls"] == 0
obs = list(csv.DictReader((data/"observations.csv").open()))
pred = list(csv.DictReader((data/"predictions.csv").open()))
params = list(csv.DictReader((data/"gp_parameters.csv").open()))
plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 10, "pdf.fonttype": 42})
fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
for name, label, color, marker in [
        ("Default", "Default", "#777777", "o"), ("BufferOnly", "Buffer only", "#ff7f0e", "s"),
        ("Concurrency8", "Concurrency 8", "#1f77b4", "^")]:
    rows = sorted([r for r in obs if r["name"] == name], key=lambda r: int(r["block"]))
    axes[0].plot([int(r["block"]) for r in rows], [float(r["tps_per_p95"]) for r in rows],
                 "-"+marker, color=color, lw=1.2, ms=4, label=label)
axes[0].axvspan(2.5, 3.3, color="#e8eef4", alpha=.65)
axes[0].set_xlim(.7, 3.3)
axes[0].set_xticks([1, 2, 3], ["1", "2", "3 (test)"])
axes[0].set_xlabel("Measurement block")
axes[0].set_ylabel("TPS / P95 (ms)")
axes[0].set_ylim(bottom=0)
axes[0].set_title("(a) Repeated measurements")
axes[0].legend(frameon=False, fontsize=8)
for i, seq in enumerate([7, 8, 9]):
    rows = {r["model"]: r for r in pred if int(r["trial"]) == seq}
    for j, (name, label, color) in enumerate([
            ("legacy", "Legacy GP", "#1f77b4"), ("repeat_aware", "Training-noise GP", "#ff7f0e")]):
        r = rows[name]; mean, low, high = [float(r[k]) for k in ("predicted_objective", "lower", "upper")]
        axes[1].errorbar(i+(j-.5)*.24, mean, yerr=[[mean-low], [high-mean]], fmt="o",
                         color=color, capsize=3, lw=1.1, ms=4, label=label if i == 0 else None)
    axes[1].scatter(i, float(rows["legacy"]["actual_objective"]), color="black", marker="x",
                    s=30, label="Measurement" if i == 0 else None, zorder=5)
axes[1].set_xticks(range(3), ["Concurrency 8\n#7", "Default\n#8", "Buffer only\n#9"])
axes[1].set_yscale("log")
axes[1].set_ylabel("TPS / P95 (ms), log scale")
axes[1].set_title("(b) Sealed predictive intervals")
axes[1].legend(frameon=False, fontsize=7)
for name, label, color, marker in [
        ("legacy", "Legacy GP", "#1f77b4", "o"), ("repeat_aware", "Training-noise GP", "#ff7f0e", "s")]:
    rows = sorted([r for r in params if r["model"] == name], key=lambda r: int(r["trial"]))
    axes[2].plot([int(r["trial"]) for r in rows], [float(r["effective_noise_log_std"]) for r in rows],
                 "-"+marker, color=color, lw=1.2, ms=4, label=label)
axes[2].set_xticks([7, 8, 9], ["7 (n=6)", "8 (n=7)", "9 (n=8)"])
axes[2].set_xlabel("Predicted trial (training count)")
axes[2].set_ylabel("Noise SD (log target)")
axes[2].set_ylim(0, 1.15*max(float(r["effective_noise_log_std"]) for r in params))
axes[2].ticklabel_format(axis="y", style="sci", scilimits=(-2, -2), useMathText=True)
axes[2].set_title("(c) Observation noise updates")
axes[2].legend(frameon=False, fontsize=8)
for ax in axes:
    ax.set_title(ax.get_title(), y=1.08)
    ax.grid(alpha=.25, lw=.6)
    ax.tick_params(direction="in", top=True, right=True)
fig.subplots_adjust(bottom=.29, top=.85, wspace=.40)
fig.text(.5, .025, "Zero LLM calls; fixed configurations. Predictions sealed before each test. "
         "Completed test observations may train the next prediction.", ha="center", fontsize=8)
out = ROOT/"docs/figures/gp_noise_validation_001"
for extension in (".png", ".pdf"):
    fig.savefig(out.with_suffix(extension), dpi=280, bbox_inches="tight")
plt.close(fig)
print("Saved prospective fixed-configuration GP audit figure")
