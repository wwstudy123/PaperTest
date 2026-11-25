from __future__ import annotations

import json
import math
import random
from pathlib import Path
import time
from typing import Protocol

from .budget import Ledger
from .contracts import Candidate, fingerprint
from .space import Space
from .security import redact


PROMPTS = {
    "selector": (
        "你负责选择数据库参数。依据负载、诊断与实测历史，从 catalog 选择不超过 "
        "max_selected 个唯一参数名。运行指标只是诊断，不是因果证据。保持持久性和压测语义不变。"
        "诊断中的 objective_key 定义目标；minimize 表示越低越好。"
        '仅返回 JSON：{"selected":["目录中的参数名"],"rationale":"简短依据"}。'
    ),
    "pruner": (
        "你负责建议优先搜索范围。selected 中每个参数必须且只能出现一次。硬边界与步长不能改变。"
        "ranges 的键集合必须与 selected 完全一致；未选中的参数必须省略，不能用 null 占位。"
        "数值上下界必须位于 catalog 边界内，且与 lower 加整数倍 step 对齐；单位沿用目录，"
        "字节值必须输出数字，不能输出 GB 等字符串。枚举值只能来自目录。"
        '仅返回一个 JSON 对象：{"ranges":{"参数名":{"lower":数值或null,"upper":数值或null,"choices":[],"special":[]}}}。'
        "每项严格保留这四个字段；数值参数的 choices 为空数组，枚举参数的 lower/upper 为 null。"
        "special 只能选目录中已有的特殊值。不要增加 step、解释或其他字段。"
        "特殊值（如0表示不限制）可以位于普通数值区间外；只放在special，"
        "不要为容纳特殊值而把lower/upper改到目录数值边界之外。"
        "整数/浮点参数的lower和upper永远必须是数字，即使special只有[0]也不能为null。"
        "本接口不支持仅特殊值范围；仍需提供合法普通数值区间，不能用null禁用它。"
    ),
    "recommender": (
        "你负责围绕实测父配置 parent 生成数据库配置变更，最多 batch_size 个。"
        "仅使用 selected 参数与 ranges 范围，严格遵循 catalog 的类型、特殊值和步长。"
        "未改参数继承父配置；每个候选至少一项实际变化，候选之间不同。"
        "结合 feedback 避免重复被拒绝的配置。历史只证明观测到的组合效果，不是因果规律。"
        "以 diagnostics.objective_key 为准；minimize 表示越低越好，不默认最大化历史数值。"
        '仅返回一个 JSON 对象：{"candidates":[{"参数名":具体数值或枚举值}]}。'
        "每项直接是变更字典，不嵌套 config，不增加说明字段，不虚构实测收益。"
        "ranges.special包含0时可以直接推荐0；未改参数的特殊默认值照常继承。"
    ),
    "reflection": (
        "你负责归纳新增执行证据，旧摘要只作背景。仅输出一个 JSON 对象，含 facts、hypotheses、conditions、actions、"
        "evidence_ids 五个字段。facts 必须完整复制输入 facts 的 trial_id/status/value，"
        "不得遗漏、修改数字、改变状态或添加文字事实。conditions 原样复制输入条件。"
        "hypotheses 是待验证解释的字符串数组，不是确定性规则。"
        "record_context 给出配置与试验类型；同配置复测的差异是波动证据，不能解释成搜索空间变化。"
        "以 conditions.objective_key 为准，minimize 表示越低越好。"
        "actions 是字符串数组，只能选 resample/adjust_range/reselect/verify/stop。"
        "evidence_ids 列出全部所引用的试验编号。不要改变硬约束，不将未执行配置标记为成功。"
    ),
}


class Backend(Protocol):
    def complete(self, role: str, payload: dict) -> dict: ...


class HeuristicBackend:
    """Free deterministic fixture, explicitly NOT an LLM or a research baseline."""

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)
        self.calls: list[str] = []

    def complete(self, role: str, payload: dict) -> dict:
        self.calls.append(role)
        if role == "selector":
            return {"selected": [k["name"] for k in payload["catalog"]][:payload["max_selected"]],
                    "rationale": "Offline fixture selects fixed catalog order."}
        if role == "pruner":
            return {"ranges": {
                k["name"]: {key: k[key] for key in ["lower", "upper", "choices", "special"]}
                for k in payload["catalog"] if k["name"] in payload["selected"]
            }}
        if role == "reflection":
            return {
                "facts": payload["facts"], "hypotheses": ["Combination effects need controlled confirmation."],
                "conditions": payload["conditions"], "actions": ["resample"],
                "evidence_ids": [f["trial_id"] for f in payload["facts"]],
            }
        if role != "recommender":
            raise ValueError("Unknown role")
        result = []
        for _ in range(payload["batch_size"]):
            delta = {}
            names = self.rng.sample(payload["selected"], self.rng.randint(1, len(payload["selected"])))
            for name in names:
                spec = next(k for k in payload["catalog"] if k["name"] == name)
                preferred = payload["ranges"][name]
                if spec["kind"] in {"enum", "boolean"}:
                    delta[name] = self.rng.choice(preferred["choices"])
                elif preferred["special"] and self.rng.random() < 0.1:
                    delta[name] = self.rng.choice(preferred["special"])
                else:
                    # A mixed local/global proposal avoids a fixture that knows the optimum.
                    low, high = preferred["lower"], preferred["upper"]
                    if self.rng.random() < 0.6:
                        parent = payload["parent"][name]
                        center = parent if low <= parent <= high else (low + high) / 2
                        value = self.rng.gauss(center, (high - low) / 5)
                    else:
                        value = self.rng.uniform(low, high)
                    value = low + round((min(high, max(low, value)) - low) / spec["step"]) * spec["step"]
                    value = min(high, max(low, value))
                    delta[name] = int(value) if spec["kind"] == "integer" else float(value)
            result.append(delta)
        return {"candidates": result}


class AgenticDBBackend:
    """Reuse the upstream API client with no silent heuristic fallback or retries."""

    def __init__(self, client, ledger: Ledger, *, max_output_tokens: int = 1500, trace_dir: Path | None = None):
        if client.dry_run or not client.api_key or not client.model:
            raise ValueError("Explicit real LLM client settings required")
        self.client, self.ledger, self.max_output_tokens = client, ledger, max_output_tokens
        self.client.response_retries = 0
        self.cache: dict[str, dict] = {}
        self.trace_dir = trace_dir

    def complete(self, role: str, payload: dict) -> dict:
        # Reserve against the exact compact bytes sent; do not lower the safety
        # estimate or replace actual provider usage with a guessed token count.
        text = json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False,
                          separators=(",", ":"))
        cache_key = fingerprint([role, payload])
        if cache_key in self.cache:
            return self.cache[cache_key]
        # Byte-length is a conservative reservation, not a reported tokenizer count.
        reservation = len((PROMPTS[role] + text).encode()) + self.max_output_tokens + 64
        remaining_before = self.ledger.limits.llm_tokens - self.ledger.llm_tokens
        self.ledger.reserve_llm(reservation)
        started = time.monotonic()
        trace = {
            "role": role, "model": self.client.model, "system_prompt": PROMPTS[role], "payload": payload,
            "reservation": {"input_bytes": len((PROMPTS[role] + text).encode()),
                            "max_output_tokens": self.max_output_tokens, "protocol_margin": 64,
                            "reserved_tokens": reservation, "remaining_tokens_before": remaining_before},
        }
        try:
            result = self.client._call_openai_compatible(
                text, operation=f"admtune_{role}", system_prompt=PROMPTS[role],
                max_output_tokens=self.max_output_tokens,
                timeout_s=max(1, min(self.client.timeout_s, int(
                    self.ledger.remaining_seconds() - self.ledger.limits.recovery_reserve_seconds
                ))),
            )
            trace["response"] = result
        except Exception as exc:
            trace["error"] = redact(str(exc))
            raise
        finally:
            usage = self.client.consume_last_usage()
            trace.update(usage=usage, elapsed_seconds=time.monotonic() - started)
            if self.trace_dir is not None:
                self.trace_dir.mkdir(parents=True, exist_ok=True)
                path = self.trace_dir / f"{self.ledger.llm_calls:03d}-{role}.json"
                serialized = redact(json.dumps(trace, ensure_ascii=False, indent=2))
                serialized = serialized.replace(self.client.api_key, "<redacted>")
                path.write_text(serialized + "\n", encoding="utf-8")
                path.chmod(0o600)
            self.ledger.settle_llm(reservation, usage.get("total_tokens") if usage else None)
        if not isinstance(result, dict):
            raise ValueError("Agent output must be a JSON object")
        self.cache[cache_key] = result
        return result


class Team:
    def __init__(self, space: Space, backend: Backend, *, batch_size: int = 4, max_selected: int = 6):
        self.space, self.backend = space, backend
        self.batch_size, self.max_selected = batch_size, max_selected
        self.selected: list[str] = []
        self.ranges: dict = {}
        self.generation = 0
        self.feedback: dict = {}
        self.diagnostics: dict = {}

    def initialize(self, context: list[float], experiences: list[dict], *,
                   diagnostics: dict | None = None, reuse_selection: bool = False):
        if diagnostics is not None:
            self.diagnostics = diagnostics
        payload = {"catalog": self.space.describe(), "context": context,
                   "experiences": experiences[:5], "max_selected": self.max_selected,
                   "feedback": self.feedback, "diagnostics": self.diagnostics,
                   "fixed_resource_context": self.space.fixed_resources,
                   "host_memory_budget_bytes": self.space.memory_bytes}
        selected = self.selected if reuse_selection else self.backend.complete("selector", payload).get("selected")
        if (
            not isinstance(selected, list) or not selected or len(set(selected)) != len(selected)
            or len(selected) > self.max_selected or set(selected) - set(self.space.names)
        ):
            raise ValueError("Invalid selected parameter set")
        self.selected = selected
        output = self.backend.complete("pruner", {**payload, "selected": selected,
            "catalog": [k for k in payload["catalog"] if k["name"] in selected],
            "numeric_range_contract": {
                k["name"]: {"lower_upper_must_be_numeric": True, "null_allowed": False,
                            "special_is_separate": True}
                for k in payload["catalog"] if k["name"] in selected
                and k["kind"] in {"integer", "float"}}})
        self.set_ranges(output.get("ranges"))

    def set_ranges(self, ranges: dict):
        if not isinstance(ranges, dict) or set(ranges) != set(self.selected):
            raise ValueError("Preferred ranges must cover selected knobs exactly")
        for name, preferred in ranges.items():
            knob = self.space.knobs[name]
            if set(preferred) != {"lower", "upper", "choices", "special"}:
                raise ValueError("Malformed preferred range")
            if any(value not in knob.special for value in preferred["special"]):
                raise ValueError("Invented special value")
            if knob.kind in {"enum", "boolean"}:
                if not preferred["choices"] or any(c not in knob.choices for c in preferred["choices"]):
                    raise ValueError("Invalid preferred enum options")
            else:
                low, high = preferred["lower"], preferred["upper"]
                if (isinstance(low, bool) or isinstance(high, bool)
                        or not isinstance(low, (float, int)) or not isinstance(high, (float, int))):
                    raise ValueError(
                        f"{name}: Numeric preferred bounds required; lower/upper cannot be null or bool, "
                        f"including with special={preferred['special']}; ordinary bounds must be within "
                        f"[{knob.lower}, {knob.upper}], step={knob.step}. Special-only ranges unsupported.")
                if not math.isfinite(low + high) or not knob.lower <= low <= high <= knob.upper:
                    raise ValueError("Preferred range exceeded hard bounds")
                knob.validate(low)
                knob.validate(high)
        self.ranges = ranges

    def relax(self):
        self.ranges = {
            n: {key: getattr(self.space.knobs[n], key) for key in ["lower", "upper", "choices", "special"]}
            for n in self.selected
        }

    def recommend(self, parent: dict, phase_id: str, context: list[float], experiences: list[dict]) -> list[Candidate]:
        self.generation += 1
        output = self.backend.complete("recommender", {
            "catalog": self.space.describe(), "selected": self.selected, "ranges": self.ranges,
            "parent": parent, "context": context, "experiences": experiences[:5],
            "batch_size": self.batch_size,
            "generation": self.generation, "feedback": self.feedback,
            "diagnostics": self.diagnostics,
            "fixed_resource_context": self.space.fixed_resources,
            "host_memory_budget_bytes": self.space.memory_bytes,
        })
        deltas = output.get("candidates")
        if not isinstance(deltas, list) or not 1 <= len(deltas) <= self.batch_size:
            raise ValueError("Invalid candidate batch")
        candidates, seen = [], set()
        for delta in deltas:
            try:
                if not isinstance(delta, dict) or set(delta) - set(self.selected):
                    continue
                full = self.space.merge(parent, delta)
                for name, value in delta.items():
                    bounds = self.ranges[name]
                    knob = self.space.knobs[name]
                    if knob.kind in {"enum", "boolean"}:
                        if value not in bounds["choices"]:
                            raise ValueError("Outside preferred choices")
                    elif value not in bounds["special"] and not bounds["lower"] <= value <= bounds["upper"]:
                        raise ValueError("Outside preferred range")
                item = Candidate(full, phase_id, fingerprint(parent))
                if item.key != fingerprint(parent) and item.key not in seen:
                    candidates.append(item)
                    seen.add(item.key)
            except (ValueError, TypeError):
                continue
        return candidates
