"""OnlineTune动态接口：因果SQL上下文、DBSCAN/SVM路由及分区域原安全搜索模型。

用可观测SQL结构特征替代未完整发布的LSTM编码器；不声称原编码器逐位复现。
"""
from __future__ import annotations

from bisect import bisect_left
import json
from pathlib import Path

import numpy as np

from .controller import Controller, Policy
from .native_dynamic import native_record
from .onlinetune_adapter import OnlineTuneStrategy, StaticRegion, REVISION

DYNAMIC_ADAPTATION = {
    "name": "OnlineTune dynamic adapter",
    "source_revision": REVISION,
    "context": ["read_share", "write_share", "range_scan_share", "range_sort_share",
                "point_read_share", "mean_range_fraction", "mean_limit_fraction",
                "aggregate_fraction", "nested_select_share", "log_offered_rate"],
    "encoder": "observed SQL structure, not unavailable pretrained LSTM or future phase labels",
    "routing": "past windows only; StandardScaler/PCA3/DBSCAN(.7,8), RBF SVM gamma=auto",
    "novel_context": "provisional region when distance to observed region center exceeds .25",
    "default_reference": "budgeted measured phase anchor; historical raw utilities recentered on activation",
    "model_reuse": "region keyed raw observations and upstream GP hyperparameters retained on return",
    "cross_region_transfer": "objective-only transfer for >10% gains above observed 90th percentile",
    "objective": "inverse measured P95; original objective-difference and relative-constraint GPs",
    "llm_calls": 0,
}


def observed_sql_context(offers, start, end, *, table_size):
    """仅接收已经封闭窗口的实际到达；绝不读取请求的phase字段或性能结果。"""
    if end <= start or table_size <= 0 or not offers:
        raise ValueError("Nonempty complete external window required")
    if any(not start <= r["scheduled"] < end for r in offers):
        raise ValueError("Request outside the observed window (future leakage)")
    read = np.array([r["operation"] == "read" for r in offers], dtype=float)
    ranges = np.array([r.get("range_rows", 0) for r in offers], dtype=float)
    limits = np.array([r.get("top_k", 0) for r in offers], dtype=float)
    scan = read * (ranges > 0) * (limits == 0)
    sort = read * (limits > 0)
    return [float(x) for x in (
        read.mean(), 1 - read.mean(), scan.mean(), sort.mean(),
        (read * (ranges == 0)).mean(), np.minimum(ranges / table_size, 1).mean(),
        np.divide(limits, ranges, out=np.zeros_like(limits), where=ranges > 0).mean(),
        (scan + sort * (2 / 3)).mean(), sort.mean(),
        min(1., np.log1p(len(offers) / (end - start)) / np.log1p(1000)))]


class ContextRouter:
    """临时DBSCAN编号映射为稳定区域ID，避免重聚类时丢掉旧模型。"""
    def __init__(self, event):
        self.event = event
        self.centers = {}
        self.classifier = None
        self.label_regions = {}
        self.fitted_windows = 0
        self.svm_fits = 0

    def _region(self, center):
        center = np.asarray(center)
        if self.centers:
            region = min(self.centers, key=lambda r: np.linalg.norm(center - self.centers[r]))
            if np.linalg.norm(center - self.centers[region]) <= .25:
                return region
        region = len(self.centers)
        self.centers[region] = center.copy()
        return region

    def fit(self, contexts):
        from sklearn.cluster import DBSCAN
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC
        x = np.asarray(contexts, dtype=float)
        if x.ndim != 2 or x.shape[1] != 10 or not np.isfinite(x).all():
            raise ValueError("Finite 10-dimensional observed contexts required")
        self.fitted_windows = len(x)
        self.classifier, self.label_regions = None, {}
        if len(x) < 8:
            return
        scaled = StandardScaler().fit_transform(x)
        if np.max(np.ptp(scaled, axis=0)) < 1e-12:
            labels = np.zeros(len(x), dtype=int)
        else:
            projected = PCA(n_components=min(3, len(x), x.shape[1])).fit_transform(scaled)
            labels = DBSCAN(eps=.7, min_samples=8).fit_predict(projected)
        for label in sorted(set(labels) - {-1}):
            center = x[labels == label].mean(axis=0)
            region = self._region(center)
            self.centers[region] = center
            self.label_regions[int(label)] = region
        usable = labels >= 0
        if len(set(labels[usable])) >= 2:
            self.classifier = SVC(kernel="rbf", gamma="auto").fit(x[usable], labels[usable])
            self.svm_fits += 1
        self.event("onlinetune_context_clusters", self.snapshot())

    def route(self, context):
        context = np.asarray(context, dtype=float)
        if context.shape != (10,) or not np.isfinite(context).all():
            raise ValueError("Finite 10-dimensional current context required")
        if self.classifier is not None:
            label = int(self.classifier.predict(context.reshape(1, -1))[0])
            region = self.label_regions[label]
            if np.linalg.norm(context - self.centers[region]) <= .25:
                return region
        # 新负载不足8窗口时不强制把它归到旧区域；不用A/B标签做路由。
        return self._region(context)

    def snapshot(self):
        return {"centers": {str(k): v.tolist() for k, v in self.centers.items()},
                "windows": self.fitted_windows, "svm_fits": self.svm_fits,
                "dbscan_label_to_region": self.label_regions}


class DynamicOnlineTuneStrategy(OnlineTuneStrategy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.router = ContextRouter(self.event)
        self.bank, self.facts, self.anchors = {}, {}, {}
        self.region = None

    def make_region_history(self):
        path = self.output / f"region-{self.region}"
        path.mkdir(exist_ok=True)
        return StaticRegion(path, self.event)

    def _fact(self, record):
        return {"trial_id": record["trial_id"], "config": dict(record["config"]),
                "value": float(record["value"]), "context": self.context.tolist(),
                "phase_id": record.get("phase_id"), "origin_region": self.region,
                "anchor_at_observation": self.baseline}

    def _save_fact(self, record):
        self.facts[self.region][record["trial_id"]] = self._fact(record)

    def _rebase(self):
        """不丢掉历史观测；将原始效用重表达为当前已测基准下的差值/相对约束。"""
        a = self.algorithm
        facts = list(self.facts[self.region].values())
        x = np.array([np.r_[self.codec.encode(r["config"]), r["context"]] for r in facts])
        y = np.array([r["value"] - self.baseline for r in facts])
        local = [r for r in facts if r.get("origin_region", self.region) == self.region]
        sx = np.array([np.r_[self.codec.encode(r["config"]), r["context"]] for r in local])
        sy = np.array([r["value"] - self.baseline for r in local])
        a.model.train_gp(x, y)
        a.s[0].train_gp(sx, -sy / abs(self.baseline))
        self.event("onlinetune_region_rebased", {
            "region": self.region, "anchor": self.baseline,
            "training_ids": [r["trial_id"] for r in facts], "records": len(facts),
            "constraint_training_ids": [r["trial_id"] for r in local]})

    def start_phase(self, record, context, past_contexts):
        self.router.fit(past_contexts)
        region = self.router.route(context)
        reused = region in self.bank
        self.region, self.context = region, np.asarray(context, dtype=float)
        self.anchors[region] = float(record["value"])
        if not reused:
            self.algorithm = None
            self.facts[region] = {}
            super().observe(record)
            self.bank[region] = self.algorithm
        else:
            self.algorithm = self.bank[region]
            self.baseline = float(record["value"])
            self.initial = dict(record["config"])
            self._save_fact(record)
            self._rebase()
            a = self.algorithm
            a.default_x = self.codec.encode(self.initial)
            a.context_change, a.context = True, self.context
            a.knobL, a.ydeltaL, a.xL, a.knob_modelL = [], [], [], []
            a._phase, a._iteration, a._best_x = "best", 0, a.default_x.copy()
            a.best_x_evauated, a.best_y_evaluated = {}, {}
            for fact in self.facts[region].values():
                key = tuple(fact["context"])
                value = fact["value"] - self.baseline
                if key not in a.best_y_evaluated or value > a.best_y_evaluated[key]:
                    a.best_y_evaluated[key] = value
                    a.best_x_evauated[key] = np.r_[self.codec.encode(fact["config"]), fact["context"]]
            # 不沿用上一阶段的子空间缓存来决定新阶段起点。
            for attr in ("_tr_domain", "_line_domain"):
                if hasattr(a, attr):
                    delattr(a, attr)
            self.pending = None
        self._save_fact(record)
        self.event("onlinetune_region_activated", {
            "phase_id": record.get("phase_id"), "region": region, "reused_model": reused,
            "context": self.context.tolist(), "measured_anchor": self.baseline,
            "retained_ids": list(self.facts[region]), "routing": self.router.snapshot()})
        self.persist()

    def observe(self, record):
        if record["metadata"]["kind"] == "verification":
            if (record["status"] == "measured" and record["constraints_ok"] and record["complete"]
                    and record["effective"] and record["value"] is not None):
                self._save_fact(record)
                self._rebase()
                self.persist()
            return
        super().observe(record)
        if record["status"] == "measured" and record["complete"] and record["effective"] and record["constraints_ok"]:
            self._save_fact(record)
            self.share_good_observation(record)
            self.persist()

    def share_good_observation(self, record):
        """沿用上游高收益跨区域目标模型迁移，不伪称已在其他区域测过安全性。"""
        fact = self.facts[self.region][record["trial_id"]]
        gain = (fact["value"] - self.baseline) / abs(self.baseline)
        observed = {r["trial_id"]: r for fs in self.facts.values() for r in fs.values()}
        gains = [(r["value"] - r["anchor_at_observation"]) / abs(r["anchor_at_observation"])
                 for r in observed.values() if "anchor_at_observation" in r]
        if not gains or gain <= .1 or gain <= float(np.percentile(gains, 90)):
            return
        point = np.r_[self.codec.encode(fact["config"]), fact["context"]]
        for region, model in self.bank.items():
            if region == self.region or fact["trial_id"] in self.facts[region]:
                continue
            self.facts[region][fact["trial_id"]] = dict(fact)
            model.model.add_data(point, fact["value"] - self.anchors[region])
            self.event("onlinetune_cross_region_transfer", {
                "trial_id": fact["trial_id"], "from_region": self.region, "to_region": region,
                "objective_only": True, "relative_gain": gain})

    def persist(self):
        snapshot = {"policy": DYNAMIC_ADAPTATION, "active_region": self.region, "anchors": self.anchors,
                    "router": self.router.snapshot(), "regions": {
                        str(k): {"raw_observations": list(self.facts[k].values()),
                                 "objective_X": a.model.gp.X.tolist(),
                                 "objective_Y": a.model.gp.Y.tolist(),
                                 "constraint_Y": a.s[0].gp.Y.tolist(),
                                 "kernel_parameters": a.model.gp.param_array.tolist()}
                        for k, a in self.bank.items()}}
        (self.output / "dynamic_regions.json").write_text(json.dumps(snapshot, indent=2))

    def snapshot(self):
        return {"adaptation": DYNAMIC_ADAPTATION, "source_report": self.source_report,
                "active_region": self.region, "regions": len(self.bank),
                "router": self.router.snapshot()}


class OnlineTuneDynamicController(Controller):
    def __init__(self, *, telemetry, upstream, output, table_size, **kwargs):
        kwargs.update(team=None, reflector=None, policy=Policy(surrogate_gate=False, min_gain=0,
                                                             final_repeats=1))
        super().__init__(**kwargs)
        self.telemetry, self.table_size = telemetry, table_size
        self.contexts, self.context_ends = [], []
        self.strategy = DynamicOnlineTuneStrategy(self.space, upstream, output, self.store.event, self.seed)

    def sync_contexts(self):
        with self.telemetry.condition:
            ends = [w["window_end"] for w in self.telemetry.windows
                    if not self.context_ends or w["window_end"] > self.context_ends[-1]]
            offers = sorted(self.telemetry.offered.values(), key=lambda r: r["scheduled"])
        times = [r["scheduled"] for r in offers]
        for end in ends:
            start = end - self.telemetry.fence.window_seconds
            rows = offers[bisect_left(times, start):bisect_left(times, end)]
            if not rows:
                continue
            self.contexts.append(observed_sql_context(rows, start, end, table_size=self.table_size))
            self.context_ends.append(end)
        if not self.contexts:
            raise ValueError("No complete observed SQL context")
        path = self.strategy.output / "observed_contexts.json"
        path.write_text(json.dumps({"window_ends": self.context_ends, "contexts": self.contexts}))
        return self.contexts[-1]

    def begin_phase(self, phase_id, context):
        self.phase_id, self.last_context, self.active = phase_id, list(context), False
        self.baseline = self.best_value = None
        self.best_config = dict(self.current)
        record = self.trial(self.current, context, kind="baseline")
        if not (record.trainable and record.constraints_ok):
            return
        self.baseline = self.best_value = record.value
        features = self.sync_contexts()
        self.strategy.start_phase(native_record(record), features, self.contexts)
        self.active = True

    def step(self):
        if not self.active or self.detector.state != "stable" or self.detector.changed_count:
            return
        self.ledger.ensure_time(self.executor.estimated_trial_seconds)
        self.strategy.context = np.asarray(self.sync_contexts())
        cfg, kind, decision = self.strategy.propose(
            [native_record(t) for t in self.store.trials()], self.current)
        self.store.event("onlinetune_dynamic_selection", {
            "phase_id": self.phase_id, "region": self.strategy.region,
            "context": self.strategy.context.tolist(), "observed_until": self.context_ends[-1],
            "decision": decision})
        record = self.trial(cfg, self.last_context, kind=kind)
        if record.metadata.get("accepted"):
            self.best_value, self.best_config = record.value, dict(record.config)
        self.strategy.observe(native_record(record))
        if not record.trainable or not record.constraints_ok:
            self.active = False
        return record

    def fit(self, context):
        self.gp.fitted = False  # 绝不调用ADMTune的GP；原GP在策略内更新。

    def reflect(self, events, *, force=False):
        if force:
            records = self.store.trials()
            if (records and records[-1].phase_id == self.phase_id
                    and records[-1].metadata.get("kind") == "verification"):
                self.strategy.observe(native_record(records[-1]))
            self.store.event("onlinetune_no_llm_reflection", {"phase_id": self.phase_id})
