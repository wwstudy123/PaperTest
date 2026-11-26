"""冻结的四方法/消融动态矩阵。默认只跑首个配对种子，不隐式启动全部重复。"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from admtune.continuous import atomic_json
from admtune.provenance import source_provenance


POLICY = {"version": "dynamic_matrix_v1", "phase_seconds": 600, "phases": ["A", "B", "A"],
    "offered_rate": 10, "query_profile": "range-mixed", "range_rows": 32768, "top_k": 8192,
    "scan_fractions": [.9, .1, .9], "neutral_prewarm_seconds": 180,
    "true_trial_cap": 18, "llm_call_cap": 36, "llm_token_cap": 100000,
    "measurement_warmup_seconds": 15, "measurement_seconds": 60, "fixed_noise_log_std": .02,
    "max_output_tokens": 1500,
    "independent_history_start": "empty per run; within-run facts retained unless ablated",
    "native_dynamic_adapter": "shared observed-change trigger; restart source search per phase; "
                              "keep accepted config and measured in-run history; no added GP/Reflexion",
    "attainment": {"reference": ".9 * paired Default median 20-second window P95 per phase",
                   "consecutive_complete_windows": 3, "window_seconds": 20},
    "primary_metrics": ["continuous end-to-end P95", "actual cumulative successful completions",
                        "first sustained target confirmation from phase start"],
    "provider_seed_controlled": False, "pilot_seed_42_not_reused": True}


def method_order(seed, methods):
    base = ["Default", "AgenticDB", "AgentTune", "ADMTune"]
    offset = (seed-42) % len(base)
    return [m for m in base[offset:]+base[:offset] if m in methods]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seeds", nargs="+", type=int, default=[43])
    p.add_argument("--methods", nargs="+", choices=["Default","AgenticDB","AgentTune","ADMTune"],
                   default=["Default","AgenticDB","AgentTune","ADMTune"])
    p.add_argument("--ablation", choices=["full","single_slice","no_history","no_gp","no_reflection"], default="full")
    p.add_argument("--dedicated-instance", action="store_true")
    p.add_argument("--allow-paid-llm", action="store_true")
    a = p.parse_args()
    if not a.dedicated_instance or not a.allow_paid_llm:
        p.error("Explicit dedicated instance and paid-LLM authorization required")
    if (len(set(a.seeds)) != len(a.seeds) or any(s not in (43,44,45) for s in a.seeds)
            or len(set(a.methods)) != len(a.methods)):
        p.error("Frozen paired request seeds 43/44/45; unique methods and seeds required")
    if a.ablation != "full" and a.methods != ["ADMTune"]:
        p.error("Ablation names are ADMTune variants, never renamed native baselines")
    root = Path(__file__).resolve().parents[1]
    os.environ.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    binary = "/root/autodl-tmp/admtune-mysql/mysql-8.0.36-linux-glibc2.17-x86_64-minimal/bin"
    os.environ["PATH"] = binary+os.pathsep+os.environ.get("PATH","")
    a.output.mkdir(mode=0o700, parents=True, exist_ok=False)
    private = root/"configs/local"
    target, benchmark, knobs = [private/n for n in (
        "workset-v2-target.yaml", "timed-readonly-100min.yaml", "online-dynamic-knobs.yaml")]
    atomic_json(a.output/"protocol.json", {"policy": POLICY, "seeds": a.seeds, "methods": a.methods,
        "ablation": a.ablation, "orders": {str(s): method_order(s,a.methods) for s in a.seeds},
        "private_config_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in (target,benchmark,knobs)}, **source_provenance()})
    state = {"status": "running", "completed": [], "ablation": a.ablation,
             "started_at_utc": datetime.now(timezone.utc).isoformat()}
    def save(): atomic_json(a.output/"matrix.json", state)
    def run(module, args, log):
        with log.open("w") as f:
            rc = subprocess.call([sys.executable,"-u","-m",module,*map(str,args)],
                                 cwd=root,stdout=f,stderr=subprocess.STDOUT)
        return rc
    try:
        for seed in a.seeds:
            parent = a.output/f"seed-{seed}"; parent.mkdir()
            for method in method_order(seed,a.methods):
                state.update(seed=seed, method=method, stage="reset"); save()
                label = method if a.ablation == "full" else f"ADMTune-{a.ablation}"
                reset = ["--root",root,"--dedicated-instance"]
                rc = run("tools.reset_readonly_baseline", [*reset,"--audit",parent/f"{label}-reset.json"],
                         parent/f"{label}-reset.log")
                if rc: raise RuntimeError(f"{seed}/{label} reset failed; no paid calls started")
                state["stage"] = "neutral_prewarm"; save()
                warm = ["--target",target,"--scenario","steady","--phase-seconds",60,
                        "--query-profile","range-mixed","--table-size",2000000,"--range-rows",32768,
                        "--top-k",8192,"--seed",1000+seed,"--dedicated-instance",
                        "--output",parent/f"{label}-warmup"]
                if run("admtune.replay",warm,parent/f"{label}-warmup.log"):
                    raise RuntimeError(f"{seed}/{label} warmup failed")
                state["stage"] = "continuous_run"; save(); started = time.monotonic()
                stream = ["--target",target,"--phase-seconds",600,"--query-profile","range-mixed",
                          "--range-rows",32768,"--top-k",8192,"--dedicated-instance",
                          "--output",parent/label]
                if method == "Default":
                    rc = run("admtune.replay",[*stream,"--scenario","composition","--table-size",2000000,
                             "--seed",seed],parent/f"{label}.log")
                else:
                    rc = run("admtune.dynamic",[*stream,"--benchmark",benchmark,"--knobs",knobs,
                        "--backend","llm","--allow-paid-llm","--formal","--method",method,
                        "--ablation",a.ablation,"--seed",seed-42,"--warmup-seconds",15,
                        "--measurement-seconds",60,"--llm-calls",36,"--llm-tokens",100000,
                        "--max-output-tokens",1500,
                        "--fixed-noise-log-std",.02],parent/f"{label}.log")
                elapsed = time.monotonic()-started
                post = run("tools.reset_readonly_baseline",
                           [*reset,"--audit",parent/f"{label}-postcheck.json","--verify-only"],
                           parent/f"{label}-postcheck.log")
                if rc or post:
                    raise RuntimeError(f"{seed}/{label}: run={rc}, postcheck={post}; retained, no automatic rerun")
                state["completed"].append({"seed": seed,"method":label,"run_wall_seconds":elapsed})
                save()
        state["status"] = "completed"
    except Exception as exc:
        state.update(status="needs_review",error=str(exc))
    finally:
        state["finished_at_utc"] = datetime.now(timezone.utc).isoformat(); save()
    return 0 if state["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
