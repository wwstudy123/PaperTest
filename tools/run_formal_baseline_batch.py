"""按固定预算串行补基线；任何一项失败即停，保留已产生的真实证据，不自动重跑收费。"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime, timezone

from admtune.continuous import atomic_json
from admtune.formal_baselines import POLICY


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--repeat", type=int, default=0)
    p.add_argument("--methods", nargs="+", default=["Default", "AgenticDB", "AgentTune"],
                   choices=["Default", "AgenticDB", "AgentTune", "ADMTune"])
    p.add_argument("--dedicated-instance", action="store_true")
    p.add_argument("--allow-paid-llm", action="store_true")
    a = p.parse_args()
    if not a.dedicated_instance or not a.allow_paid_llm or len(set(a.methods)) != len(a.methods):
        p.error("需要专用实例、收费授权和不重复的方法列表")
    root = Path(__file__).resolve().parents[1]
    mysql_bin = Path("/root/autodl-tmp/admtune-mysql/mysql-8.0.36-linux-glibc2.17-x86_64-minimal/bin")
    if not (mysql_bin/"mysql").is_file() or shutil.which("sysbench") is None:
        p.error("专用 MySQL 客户端或 sysbench 不可用；不能启动收费调用")
    os.environ["PATH"] = str(mysql_bin)+os.pathsep+os.environ.get("PATH", "")
    a.output.mkdir(mode=0o700, parents=True, exist_ok=False)
    state = {"status": "running", "policy": POLICY, "repeat": a.repeat, "methods": a.methods,
             "started_at_utc": datetime.now(timezone.utc).isoformat(), "finished_methods": []}
    atomic_json(a.output/"batch.json", state)
    os.environ.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    def execute(cmd, log):
        with log.open("w") as stream:
            return subprocess.run(cmd, cwd=root, stdout=stream, stderr=subprocess.STDOUT).returncode
    try:
        for method in a.methods:
            reset = a.output/f"{method}-reset.json"
            reset_args = [sys.executable, "-m", "tools.reset_readonly_baseline", "--root", str(root),
                          "--dedicated-instance"]
            if execute(reset_args+["--audit", str(reset)], a.output/f"{method}-reset.log"):
                raise RuntimeError(f"{method}: reset failed; paid run not started")
            cmd = [sys.executable, "-m", "admtune.formal_baselines", "--method", method,
                   "--target", str(root/"configs/local/workset-v2-target.yaml"),
                   "--benchmark", str(root/"configs/local/timed-readonly-100min.yaml"),
                   "--knobs", str(root/"configs/local/concurrency-knobs.yaml"),
                   "--output", str(a.output/method), "--reset-audit", str(reset),
                   "--repeat", str(a.repeat), "--dedicated-instance", "--allow-paid-llm"]
            rc = execute(cmd, a.output/f"{method}.log")
            post = execute(reset_args+["--audit", str(a.output/f"{method}-postcheck.json"),
                                      "--verify-only"], a.output/f"{method}-postcheck.log")
            if rc or post:
                raise RuntimeError(f"{method}: run={rc}, postcheck={post}; inspect retained logs, no automatic retry")
            state["finished_methods"].append(method)
            atomic_json(a.output/"batch.json", state)
        state["status"] = "completed"
    except Exception as exc:
        state.update(status="failed", error=str(exc))
    finally:
        state["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        atomic_json(a.output/"batch.json", state)
    return 0 if state["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
