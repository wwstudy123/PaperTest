"""限时串行动态验证：四方法、优先消融及自动实测绘图。"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from zipfile import ZipFile, ZIP_DEFLATED

from admtune.continuous import atomic_json
from admtune.replay_report import read_jsonl, write_csv
from tools.analyze_dynamic_stage1 import adaptation
from tools.plot_dynamic_matrix import configuration_counts, plot, service_metrics


JOBS = (
    ("Default", "full"), ("AgenticDB", "full"), ("AgentTune", "full"), ("ADMTune", "full"),
    ("ADMTune", "no_history"), ("ADMTune", "no_gp"),
    ("ADMTune", "single_slice"), ("ADMTune", "no_reflection"),
)


def enough_time(deadline, now):
    # 留35分钟服务/核验时间及10分钟恢复、绘图余量，不缩短任何方法的协议。
    return deadline - now >= 45 * 60


def evidence(root):
    summary = json.loads((root / "summary.json").read_text())
    post = json.loads((root.parent / f"{root.name}-postcheck.json").read_text())
    if not summary.get("complete") or summary.get("execution_error") or not post.get("completed"):
        raise ValueError(f"Incomplete evidence: {root.name}")
    if root.name != "Default" and not summary.get("initial_configuration_restored"):
        raise ValueError(f"Configuration not restored: {root.name}")
    metrics, windows, phases, _ = service_metrics(read_jsonl(root / "requests.jsonl"))
    trials_file = root / "trials.jsonl"
    trials = read_jsonl(trials_file) if trials_file.exists() else []
    usage_file = root / "llm_usage.json"
    usage = json.loads(usage_file.read_text()) if usage_file.exists() else {}
    row = {"method": root.name, **metrics,
           "trial_slots_used": summary.get("budget", {}).get("trial_slots_used", 0),
           "valid_observations": summary.get("valid_observations", 0),
           "llm_calls": summary.get("llm_calls", 0), "tokens": usage.get("total_tokens", 0),
           **configuration_counts(trials, root.name == "Default"),
           "failed": summary["counts"]["failed"], "dropped": summary["counts"]["dropped"]}
    return summary, post, row, windows, phases


def make_report(output, state, seed):
    """只汇总通过末端核验的完整运行；未完成和未执行项目在状态中保留。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    public = output / "figures_and_data"
    public.mkdir(exist_ok=True)
    combined = output / "main_matrix"
    parent = combined / f"seed-{seed}"
    parent.mkdir(parents=True, exist_ok=True)
    available = {}
    for job in state["completed"]:
        label = job["label"]
        run = Path(job["output"]) / f"seed-{seed}" / label
        available[label] = run
        if label in ("Default", "AgenticDB", "AgentTune", "ADMTune"):
            for name in (label, f"{label}-postcheck.json"):
                link = parent / name
                if not link.exists():
                    link.symlink_to(run.parent / name, target_is_directory=name == label)
    full = all(m in available for m in ("Default", "AgenticDB", "AgentTune", "ADMTune"))
    if full and not (public / "main").exists():
        atomic_json(combined / "matrix.json", {"status": "completed", "origin": str(output),
                                             "note": "links to original independently reset runs"})
        plot(combined, public / "main", seed)
    rows, ws_all, ps_all, attainments = [], [], [], []
    reference = None
    for label, root in available.items():
        summary, post, row, ws, ps = evidence(root)
        manifest = json.loads((root / "manifest.json").read_text())
        signature = (summary["schedule_hash"], post["logical_checksums"], post["config_sha256"])
        if reference is None:
            reference = signature
        if signature != reference:
            raise ValueError("Paired runs differ in schedule, data or initial configuration")
        # 默认不调参；其他运行的参数目录必须一致。
        if label != "Default":
            if "space" in state and state["space"] != manifest["space"]:
                raise ValueError("Different knob catalog")
            state["space"] = manifest["space"]
        rows.append(row)
        ws_all.extend({"method": label, **w} for w in ws)
        ps_all.extend({"method": label, **p} for p in ps)
    if rows:
        write_csv(public / "overall_metrics.csv", rows, list(rows[0]))
        write_csv(public / "window_metrics.csv", ws_all, list(ws_all[0]))
        write_csv(public / "phase_metrics.csv", ps_all, list(ps_all[0]))
    for label in available:
        for i, phase in enumerate(("A1", "B", "A2")):
            ref = [w["p95_ms"] for w in ws_all if w["method"] == "Default" and w["phase"] == phase]
            if not ref:
                continue
            import numpy as np
            threshold = .9 * float(np.median(ref))
            ws = [w for w in ws_all if w["method"] == label and w["phase"] == phase]
            attainments.append({"method": label, "phase": phase, "target_p95_ms": threshold,
                                "attainment_seconds": adaptation(ws, 600 * i, threshold)})
    if attainments:
        write_csv(public / "adaptation_metrics.csv", attainments, list(attainments[0]))
    ablations = [r for r in rows if r["method"] == "ADMTune" or r["method"].startswith("ADMTune-")]
    if len(ablations) > 1:
        plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 9, "pdf.fonttype": 42})
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
        for row in ablations:
            label = row["method"].replace("ADMTune-", "w/o ").replace("no_", "")
            ws = [w for w in ws_all if w["method"] == row["method"]]
            axes[0].plot([w["end_s"] / 60 for w in ws], [w["p95_ms"] for w in ws],
                         label=label, lw=1.0)
        labels = [r["method"].replace("ADMTune-", "").replace("no_", "w/o ") for r in ablations]
        axes[1].bar(range(len(ablations)), [r["overall_p95_ms"] for r in ablations],
                    color=["#d62728"] + ["#999999"] * (len(ablations) - 1))
        axes[1].set(xticks=range(len(labels)), xticklabels=labels, ylabel="Whole-run P95 latency (ms)")
        axes[1].tick_params(axis="x", labelrotation=25)
        axes[0].set(xlabel="Workload time (min)", ylabel="P95 latency (ms; 20 s cohorts)")
        for at in (10, 20):
            axes[0].axvline(at, color="#777777", ls=":", lw=.7)
        axes[0].legend(frameon=False, fontsize=7)
        for ax in axes:
            ax.grid(axis="y", alpha=.18)
            ax.set_axisbelow(True)
        fig.tight_layout(rect=(0, .08, 1, 1))
        fig.text(.5, .01, f"Paired request seed {seed}; one run per variant; no inferred error bars.",
                 ha="center", fontsize=8)
        for ext in ("png", "pdf"):
            fig.savefig(public / f"ablations.{ext}", dpi=280, bbox_inches="tight")
        plt.close(fig)
    done = "\n".join(f"- {j['label']}" for j in state["completed"]) or "- 暂无完整运行"
    (public / "结果说明.md").write_text(
        "# 五小时动态优先验证\n\n本批次仅含真实测量，无文献比例曲线。\n\n"
        "## 已完成\n" + done + "\n\n"
        "每项只有一个新请求种子，不构成多种子统计结论。开发种子43不混入本批次。\n"
        "整体对照完成后生成main目录；消融至少两项完整结果后生成ablations图。"
        "NA表示未达标而非0秒；18个试验槽含失败，不以有效观测替代总成本。\n"
        "无文字反思消融不等于增量/全量反思比较。single_slice表示单时间片监测变体。\n"
        "性能不佳的完整运行同样保留，不按胜负挑选。TPC-C、JOB及独立重复未在本批次执行。\n",
        encoding="utf-8")
    atomic_json(public / "run_status.json", {k: v for k, v in state.items() if k != "space"})
    hashes = {str(p.relative_to(public)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in public.rglob("*") if p.is_file() and p.name != "checksums.json"}
    atomic_json(public / "checksums.json", hashes)
    with ZipFile(output / "public_figures.zip", "w", ZIP_DEFLATED) as z:
        for p in public.rglob("*"):
            if p.is_file():
                z.write(p, str(p.relative_to(public)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--deadline", required=True, help="带时区的截止时间")
    p.add_argument("--pilot", type=Path, required=True)
    p.add_argument("--seed", type=int, choices=(44, 45), default=44)
    p.add_argument("--dedicated-instance", action="store_true")
    p.add_argument("--allow-paid-llm", action="store_true")
    a = p.parse_args()
    if not a.dedicated_instance or not a.allow_paid_llm:
        p.error("Explicit dedicated-instance and paid-LLM authorization required")
    stamp = datetime.fromisoformat(a.deadline)
    if stamp.tzinfo is None:
        p.error("Deadline must include timezone")
    deadline = stamp.timestamp()
    if deadline <= time.time() or deadline - time.time() > 5 * 3600:
        p.error("Deadline must be in the next five hours")
    a.output.mkdir(parents=True, exist_ok=False, mode=0o700)
    root = Path(__file__).resolve().parents[1]
    state = {"status": "waiting_for_pilot", "deadline": a.deadline, "seed": a.seed,
             "started_at_utc": datetime.now(timezone.utc).isoformat(),
             "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
             "completed": [], "planned": [{"method": m, "ablation": v} for m, v in JOBS],
             "paid_runs_cap": 7, "llm_calls_cap": 252, "llm_tokens_cap": 700000,
             "note": "caps, not intended consumption; no retries; one seed; no fabricated results"}
    def save():
        atomic_json(a.output / "status.json", state)
    save()
    try:
        while True:
            pilot = json.loads((a.pilot / "matrix.json").read_text())
            if pilot["status"] == "completed":
                evidence(a.pilot / "seed-43/ADMTune")
                break
            if pilot["status"] != "running":
                raise RuntimeError("Pilot needs review; no new paid batch started")
            if not enough_time(deadline, time.time()):
                raise RuntimeError("Pilot did not finish with enough time left")
            time.sleep(15)
        state["status"] = "running"
        for index, (method, variant) in enumerate(JOBS):
            if not enough_time(deadline, time.time()):
                state["status"] = "budget_stopped"
                state["stop_reason"] = "Remaining time reserved for restoration/reporting; no shortened trials"
                break
            label = method if variant == "full" else f"ADMTune-{variant}"
            destination = a.output / f"{index + 1:02d}-{label}"
            state.update(current=label, current_output=str(destination))
            save()
            cmd = [sys.executable, "-u", "-m", "tools.run_dynamic_matrix",
                   "--output", str(destination), "--seeds", str(a.seed), "--methods", method,
                   "--ablation", variant, "--dedicated-instance", "--allow-paid-llm"]
            with (a.output / f"{label}.log").open("w") as log:
                child = subprocess.Popen(cmd, cwd=root, stdout=log, stderr=subprocess.STDOUT,
                                         start_new_session=True)
                try:
                    rc = child.wait(timeout=max(1, deadline - time.time() - 600))
                except subprocess.TimeoutExpired:
                    # 同时中断矩阵及其子进程，使动态执行器进入finally恢复，不使用SIGKILL。
                    os.killpg(child.pid, signal.SIGINT)
                    try:
                        child.wait(timeout=300)
                    except subprocess.TimeoutExpired:
                        pass
                    raise RuntimeError("Deadline interrupt sent; inspect restoration before any further run")
            if rc:
                raise RuntimeError(f"{label} failed; retained; no automatic rerun")
            evidence(destination / f"seed-{a.seed}" / label)
            state["completed"].append({"label": label, "output": str(destination)})
            save()
            make_report(a.output, state, a.seed)
        else:
            state["status"] = "completed"
    except Exception as exc:
        state.update(status="needs_review", error=str(exc))
    finally:
        state["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        save()
        try:
            make_report(a.output, state, a.seed)
        except Exception as exc:
            state["report_error"] = str(exc)
            save()
    return 0 if state["status"] in ("completed", "budget_stopped") else 1


if __name__ == "__main__":
    raise SystemExit(main())
