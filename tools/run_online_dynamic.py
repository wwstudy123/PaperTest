"""OnlineTune式动态评估先导：校准→Default→真实LLM ADMTune；共享只读请求计划。"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

from admtune.continuous import atomic_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--dedicated-instance", action="store_true")
    p.add_argument("--allow-paid-llm", action="store_true")
    a = p.parse_args()
    if not a.dedicated_instance or not a.allow_paid_llm:
        p.error("Explicit dedicated database and paid-LLM authorization required")
    root = Path(__file__).resolve().parents[1]
    os.environ.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    binary = "/root/autodl-tmp/admtune-mysql/mysql-8.0.36-linux-glibc2.17-x86_64-minimal/bin"
    os.environ["PATH"] = binary+os.pathsep+os.environ.get("PATH", "")
    a.output.mkdir(mode=0o700, parents=True, exist_ok=False)
    state = {"status": "running", "stage": "calibration", "phase_seconds": 600,
             "workload": "read-only scan/sort 9:1 → 1:9 → 9:1; 10 QPS, R32768, K8192",
             "caps": {"llm_calls": 12, "llm_tokens": 20000, "configuration_measurements": 18},
             "started_at_utc": datetime.now(timezone.utc).isoformat()}
    def save(): atomic_json(a.output/"batch.json", state)
    def run(module, args, label):
        state["stage"] = label; save()
        with (a.output/f"{label}.log").open("w") as log:
            code = subprocess.call([sys.executable, "-u", "-m", module, *map(str, args)],
                                   cwd=root, stdout=log, stderr=subprocess.STDOUT)
        if code:
            raise RuntimeError(f"{label} failed ({code}); no automatic paid retry")
    def reset(label, verify=False):
        run("tools.reset_readonly_baseline", ["--root", root, "--audit", a.output/f"{label}.json",
            "--dedicated-instance"]+(["--verify-only"] if verify else []), label)
    target = root/"configs/local/workset-v2-target.yaml"
    knobs = root/"configs/local/online-dynamic-knobs.yaml"
    benchmark = root/"configs/local/timed-readonly-100min.yaml"
    common = ["--target", target, "--dedicated-instance"]
    stream = ["--phase-seconds", 600, "--query-profile", "range-mixed",
              "--range-rows", 32768, "--top-k", 8192]
    try:
        reset("calibration-reset")
        run("admtune.online_dynamic_calibration", [*common, "--benchmark", benchmark,
            "--knobs", knobs, "--output", a.output/"calibration"], "calibration")
        reset("calibration-postcheck", True)
        calibration = json.loads((a.output/"calibration/summary.json").read_text())["result"]
        state["calibration"] = calibration; save()
        if not calibration["passed"]:
            state.update(status="signal_gate_not_passed", paid_llm_started=False)
        else:
            reset("Default-reset")
            run("admtune.replay", [*common, *stream, "--scenario", "composition",
                "--table-size", 2000000, "--output", a.output/"Default"], "Default")
            reset("Default-postcheck", True)
            reset("ADMTune-reset")
            state["paid_llm_started"] = True; save()
            run("admtune.dynamic", [*common, *stream, "--benchmark", benchmark, "--knobs", knobs,
                "--backend", "llm", "--allow-paid-llm", "--warmup-seconds", 15,
                "--measurement-seconds", 60, "--llm-calls", 12, "--llm-tokens", 20000,
                "--fixed-noise-log-std", calibration["fixed_noise_log_std"],
                "--seed", 0, "--output", a.output/"ADMTune"], "ADMTune")
            reset("ADMTune-postcheck", True)
            state.update(status="completed", stage="finished")
    except Exception as exc:
        state.update(status="failed", error=str(exc))
    finally:
        state["finished_at_utc"] = datetime.now(timezone.utc).isoformat(); save()
    return 0 if state["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
