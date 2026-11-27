"""OnlineTune动态独立运行：复用共同只读重置/预热/请求轨迹，零LLM，不自动重试。"""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import sys

from admtune.continuous import atomic_json
from admtune.onlinetune_dynamic import DYNAMIC_ADAPTATION
from admtune.provenance import source_provenance
from tools.run_dynamic_matrix import POLICY


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--upstream", type=Path, required=True)
    p.add_argument("--seed", type=int, choices=[43, 44, 45], default=43)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--dedicated-instance", action="store_true")
    a = p.parse_args()
    if not a.dedicated_instance or not a.upstream.is_dir():
        p.error("Explicit dedicated instance and pinned upstream source required")
    root = Path(__file__).resolve().parents[1]
    binary = "/root/autodl-tmp/admtune-mysql/mysql-8.0.36-linux-glibc2.17-x86_64-minimal/bin"
    os.environ.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    os.environ["PATH"] = binary + os.pathsep + os.environ.get("PATH", "")
    a.output.mkdir(mode=0o700, parents=True, exist_ok=False)
    state = {"status": "running", "stage": "reset", "seed": a.seed, "smoke": a.smoke,
             "started_at_utc": datetime.now(timezone.utc).isoformat(), "llm_calls": 0}
    atomic_json(a.output / "protocol.json", {"common_dynamic_policy": POLICY,
        "adapter": DYNAMIC_ADAPTATION, "smoke": a.smoke,
        "measurement_seconds": 1 if a.smoke else 60, "warmup_seconds": 0 if a.smoke else 15,
        "phase_seconds": 160 if a.smoke else 600,
        "neutral_prewarm_seconds": 0 if a.smoke else 180, **source_provenance()})

    def save():
        atomic_json(a.output / "run_state.json", state)

    def run(module, args, log):
        with (a.output / log).open("w") as f:
            return subprocess.call([sys.executable, "-u", "-m", module, *map(str, args)],
                                   cwd=root, stdout=f, stderr=subprocess.STDOUT)

    reset = ["--root", root, "--dedicated-instance"]
    target = root / "configs/local/workset-v2-target.yaml"
    benchmark = root / "configs/local/timed-readonly-100min.yaml"
    try:
        save()
        if run("tools.reset_readonly_baseline", [*reset, "--audit", a.output / "reset.json"], "reset.log"):
            raise RuntimeError("Reset failed; no tuning started")
        if not a.smoke:
            state["stage"] = "neutral_prewarm"
            save()
            rc = run("admtune.replay", ["--target", target, "--scenario", "steady",
                "--phase-seconds", 60, "--query-profile", "range-mixed", "--table-size", 2000000,
                "--range-rows", 32768, "--top-k", 8192, "--seed", 1000 + a.seed,
                "--dedicated-instance", "--output", a.output / "warmup"], "warmup.log")
            if rc:
                raise RuntimeError("Neutral prewarm failed")
        state["stage"] = "continuous_run"
        save()
        args = ["--target", target, "--benchmark", benchmark,
                "--knobs", root / "configs/experiments/dynamic-online-knobs.yaml",
                "--output", a.output / "OnlineTune", "--backend", "onlinetune", "--method", "OnlineTune",
                "--onlinetune-source", a.upstream, "--dedicated-instance",
                "--query-profile", "range-mixed", "--range-rows", 32768, "--top-k", 8192,
                "--seed", a.seed - 42, "--llm-calls", 0, "--llm-tokens", 0]
        args += (["--smoke", "--phase-seconds", 160, "--warmup-seconds", 0, "--measurement-seconds", 1]
                 if a.smoke else ["--formal", "--phase-seconds", 600,
                                    "--warmup-seconds", 15, "--measurement-seconds", 60])
        rc = run("admtune.dynamic", args, "OnlineTune.log")
        state["stage"] = "postcheck"
        save()
        post = run("tools.reset_readonly_baseline",
                   [*reset, "--audit", a.output / "postcheck.json", "--verify-only"], "postcheck.log")
        if rc or post:
            raise RuntimeError(f"Run={rc}, postcheck={post}; originals retained, no automatic retry")
        state["status"], state["stage"] = "completed", "completed"
    except Exception as exc:
        state.update(status="needs_review", error=str(exc))
    finally:
        state["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        save()
    return 0 if state["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
