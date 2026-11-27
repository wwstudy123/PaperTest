"""OnlineTune原安全搜索核心的静态执行器适配；不包含或替换为ADMTune的GP。

上游原件在忽略目录；这里仅含兼容补丁和接口胶水。静态单上下文不加载缺失的
SQL词表/离线默认预测器，不将此变体宣称为原动态工作负载编码器的复现。
"""
from __future__ import annotations

import ast
import hashlib
import importlib
import json
import math
from pathlib import Path
import sys

import numpy as np

REVISION = "88ba8c35f554bc4a4f7f1a88257b3e8928f87a39"
FILES = ("knobs.py", "utils/logger.py", "safe/algorithm.py", "safe/domain.py",
         "safe/model.py", "safe/models/gp.py", "safe/subdomain.py",
         "safe/subdomainbo.py", "safe/safeopt.py", "safe/utils.py")
ADAPTATION = {
    "name": "OnlineTune static adapter", "source_revision": REVISION,
    "context": "one constant 10-dimensional zero context; static workload only",
    "default_predictor": "actual initial measurement in this run; no offline labels",
    "context_regions": "single static region; no dynamic clustering claim",
    "importance_direction": "not provided; upstream random-direction fallback",
    "executor": "common restored-initial Sysbench executor; raw TPS and P95 retained",
    "objective": "same-measurement TPS/P95 maximize; objective GP models difference to initial",
    "safety": "upstream relative constraint GP, adaptive beta, safe acquisition/expansion",
    "offline_observations": 0, "llm_calls": 0,
    "dba_rules": "common hard catalog/constraints instead of upstream 40-knob DBA rules",
}


def prepare_core(source: Path, destination: Path):
    """固定原件校验后只复制所需算法模块；不执行上游数据库或shell入口。"""
    audit = json.loads((Path(__file__).resolve().parents[1] /
                        "docs/experiment_data/onlinetune_audit_20261128/source_audit.json").read_text())
    destination.mkdir(parents=True, exist_ok=False)
    hashes, changes = {}, []
    for name in FILES:
        relative = f"onlinetune/{name}"
        raw = (source / relative).read_text()
        if hashlib.sha256((source / relative).read_bytes()).hexdigest() != audit["source_sha256"][relative]:
            raise ValueError(f"Unpinned upstream file: {relative}")
        text = raw
        if name == "safe/subdomainbo.py":
            text = text.replace("\n)\n\n    def _tr_save_plot", "\n\n    def _tr_save_plot")
        if name == "safe/model.py":
            text = text.replace("from autotune.safe.algorithm import Greedy",
                                "from onlinetune.safe.algorithm import Greedy")
            text = text.replace("data['x'].flatten()[:40]", "data['x'].flatten()[:self.domain.d]")
        if name == "safe/models/gp.py":
            text = text.replace(" ctive_dims=", " active_dims=")
        if "np.asscalar(" in text:
            text = text.replace("np.asscalar(", "_scalar(")
            text = "import numpy as np\n\ndef _scalar(x):\n    return np.asarray(x).item()\n\n" + text
        # 软件异常必须抛出，不能在无人值守数据库试验中进入交互调试器。
        text = text.replace("pdb.set_trace()", '(_ for _ in ()).throw(RuntimeError("Upstream solver error"))')
        ast.parse(text, filename=relative)
        path = destination / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        if text != raw:
            changes.append(relative)
    report = {"revision": REVISION, "patched_modules": changes, "patched_sha256": hashes,
              "patches": ["syntax bracket", "autotune namespace", "active_dims keyword",
                          "parameter count from domain instead of 40",
                          "NumPy scalar compatibility", "interactive debugger becomes exception"]}
    (destination / "patch_manifest.json").write_text(json.dumps(report, indent=2))
    return report


class LinearCatalog:
    """保留上游线性连续动作坐标；按共同目录的步长和特殊值还原可执行配置。"""
    def __init__(self, space):
        self.space = space
        self.lower = np.array([min([k.lower, *k.special]) for k in
                               (space.knobs[n] for n in space.names)], dtype=float)
        self.upper = np.array([max([k.upper, *k.special]) for k in
                               (space.knobs[n] for n in space.names)], dtype=float)

    def encode(self, config):
        return (np.array([config[n] for n in self.space.names]) - self.lower) / (self.upper - self.lower)

    def decode(self, action):
        cfg = {}
        for i, name in enumerate(self.space.names):
            k = self.space.knobs[name]
            raw = self.lower[i] + float(np.clip(action[i], 0, 1)) * (self.upper[i] - self.lower[i])
            grid = min(k.upper, k.lower + math.ceil(max(0, int(raw) - k.lower) / k.step) * k.step)
            # 目前共同目录的唯一特殊值为并发0；其余参数均为整数规则网格。
            cfg[name] = min([grid, *k.special], key=lambda v: abs(v - raw))
        return self.space.merge(cfg, {})


class StaticRegion:
    """静态单上下文的历史接口；不假装执行过动态聚类或离线历史训练。"""
    current_id = 0
    def __init__(self, output, event):
        self.output, self.event, self.history = output, event, []

    def get_label(self, region):
        return -1

    def add_data_to_cluster(self, default, context, score, knobs, label):
        self.history.append({"default": float(default), "context": list(context),
                             "score": float(score), "action": list(knobs)})

    def array_to_file(self):
        (self.output / "static_region_history.json").write_text(json.dumps(self.history, indent=2))

    def get_important_knobs(self, label):
        self.event("onlinetune_original_direction_fallback",
                   {"reason": "static adapter without offline fANOVA; upstream fallback to random direction"})
        return False, None


class CatalogRule:
    def __init__(self, codec):
        self.codec = codec

    def check_safe_action(self, x, no_explore=True):
        result = np.asarray(x).copy()
        d = len(self.codec.space.names)
        result[:d] = self.codec.encode(self.codec.decode(result[:d]))
        return result


class OnlineTuneStrategy:
    """实际调用上游SafetyMixin/GP，不使用LLM，也不导入ADMTune代理模型。"""
    def __init__(self, space, source, output, event, seed=43):
        self.space, self.output, self.event = space, output, event
        self.codec = LinearCatalog(space)
        self.context = np.zeros(10)
        self.algorithm = None
        self.pending = None
        self.source_report = prepare_core(source, output / "upstream_core")
        sys.path.insert(0, str(output / "upstream_core"))
        np.random.seed(seed)
        knobs = importlib.import_module("onlinetune.knobs")
        knobs.gen_continuous, knobs.knob2action = self.codec.decode, self.codec.encode
        self.core = importlib.import_module("onlinetune.safe.safeopt")
        # 防止同进程重复验收时from-import缓存仍指向上一个参数目录。
        for name in ("onlinetune.safe.safeopt", "onlinetune.safe.subdomainbo"):
            sys.modules[name].gen_continuous = self.codec.decode

    def initialize(self, baseline):
        domain_module = importlib.import_module("onlinetune.safe.domain")
        d = len(self.space.names)
        domain = domain_module.ContinuousDomain(np.zeros(d), np.ones(d))
        all_domain = domain_module.ContinuousDomain(np.r_[np.zeros(d), -np.ones(10)],
                                                    np.ones(d + 10))
        self.baseline = baseline["value"]
        self.initial = dict(baseline["config"])
        algo = self.core.SafetyMixin()
        algo.initialize(domain=domain, domain_all=all_domain, x0=self.codec.encode(self.initial),
                        num_constraints=1, initial_data=[], parameter_names=self.space.names)
        algo.cluster = StaticRegion(self.output, self.event)
        algo.default_x = self.codec.encode(self.initial)
        algo.best_y, algo.context_change = 0, True
        algo.knobL, algo.ydeltaL, algo.xL, algo.knob_modelL = [], [], [], []
        algo.rule = CatalogRule(self.codec)
        self.algorithm = algo
        # 原算法第一个next选择x0；复用预算内已经真实测量的初始观测，不额外测量。
        x, additional = algo.next(self.context)
        if self.codec.decode(x) != self.initial:
            raise ValueError("Upstream initial point differs from the measured baseline")
        self.pending = (x, additional)

    def propose(self, records, incumbent):
        if self.algorithm is None:
            raise ValueError("Real baseline observation required before OnlineTune proposal")
        x, additional = self.algorithm.next(self.context)
        cfg = self.codec.decode(x)
        self.pending = (np.r_[self.codec.encode(cfg), self.context], additional)
        a = self.algorithm
        decision = {"rule": "upstream_OnlineTune_SafetyMixin", "source_phase": a._phase,
                    "additional_data": {k: str(v) for k, v in additional.items()},
                    "novel_configuration": cfg not in [r["config"] for r in records],
                    "objective_beta": float(a.model.beta), "constraint_beta": float(a.s[0].beta)}
        self.event("onlinetune_selection", decision)
        # 重复也占真实预算；不能把原算法回到中心的决策换成随机新配置。
        return cfg, "candidate", decision

    def observe(self, record):
        if record["metadata"]["kind"] == "verification":
            return
        if not (record["status"] == "measured" and record["complete"] and record["effective"]
                and record["constraints_ok"] and record["value"] is not None):
            self.event("onlinetune_invalid_observation", {"trial_id": record["trial_id"]})
            return
        if self.algorithm is None:
            self.initialize(record)
        if self.pending is None:
            raise ValueError("Observation without an upstream selection")
        _, additional = self.pending
        a = self.algorithm
        x = np.r_[self.codec.encode(record["config"]), self.context]
        delta = float(record["value"] - self.baseline)
        relative = delta / abs(self.baseline)
        data = {"x": x, "y": delta, "s": np.array([-relative]),
                "default": self.baseline, "lower_bound_objective": self.baseline,
                "context": self.context, **additional}
        if relative > 0:
            a.context_change = False
        a.knobL.append(record["config"])
        a.ydeltaL.append(relative)
        a.xL.append(x[:len(self.space.names)])
        a.s[0].x_added, a.s[0].y_added = x, relative
        previous = len(a.model.gp.X)
        is_new = record["config"] not in a.knob_modelL
        a.add_data(data)
        if is_new and (len(a.model.gp.X) <= previous or len(a.s[0].gp.X) <= previous):
            raise RuntimeError("Upstream GP training failed; no unreported fallback allowed")
        self.pending = None
        self.event("onlinetune_model_updated", {
            "trial_id": record["trial_id"], "phase": a._phase, "relative_gain": relative,
            "objective_training_points": len(a.model.gp.X),
            "constraint_training_points": len(a.s[0].gp.X),
            "unique_model_configurations": len(a.knob_modelL)})

    def accepts(self, record, incumbent):
        return record["value"] > incumbent["value"]

    def snapshot(self):
        return {"adaptation": ADAPTATION, "source_report": self.source_report,
                "phase": self.algorithm._phase if self.algorithm else None,
                "model_configurations": len(self.algorithm.knob_modelL) if self.algorithm else 0}
