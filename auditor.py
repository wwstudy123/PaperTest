from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from models import BenchmarkResult

DB_PHASE = "db"
STOP_PHASE = "stop"
TUNING_PHASES = {DB_PHASE}


@dataclass(slots=True)
class AuditorSettings:
    enabled: bool = True
    initial_phase: str = DB_PHASE
    min_db_rounds: int = 5
    db_plateau_patience: int = 3
    repeat_benchmark_patience: int = 2
    respect_model_phase_recommendation: bool = True
    model_recommendation_min_rounds: int = 3
    use_workload_direction_policy: bool = True

    def __post_init__(self):
        if self.initial_phase != DB_PHASE:
            raise ValueError("只允许数据库调优阶段 db")
        if min(self.min_db_rounds, self.db_plateau_patience, self.repeat_benchmark_patience,
               self.model_recommendation_min_rounds) < 1:
            raise ValueError("审计轮数与耐心参数必须为正整数")


@dataclass(slots=True)
class AuditDecision:
    current_phase: str
    next_phase: str
    action: str
    reason: str
    stop: bool = False
    plateau_detected: bool = False
    phase_round_count: int = 0
    consecutive_non_improving: int = 0
    consecutive_repeats: int = 0
    best_score: float | None = None
    model_recommendation: dict[str, Any] | None = None
    model_recommendation_used: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def allowed_actions_for_phase(phase: str) -> set[str]:
    return {"db_config", "repeat_benchmark"} if phase == DB_PHASE else set()


class TuningAuditor:
    def __init__(self, settings: AuditorSettings) -> None:
        self.settings = settings

    def audit(self, *, rounds: list[dict[str, Any]], current_phase: str,
              best_result: BenchmarkResult,
              workload_interpretation: dict[str, Any] | None = None) -> AuditDecision:
        if current_phase != DB_PHASE:
            return AuditDecision(current_phase, STOP_PHASE, "stop", "不支持的调优阶段，停止执行。", stop=True)
        phase_rounds = [entry for entry in rounds if self._round_phase(entry, current_phase) == DB_PHASE]
        count = len(phase_rounds)
        non_improving = self._consecutive_non_improving(phase_rounds, count_repeat_benchmarks=False)
        repeats = self._consecutive_repeats(phase_rounds)
        recommendation = self._latest_model_recommendation(phase_rounds)
        plateau = self.settings.enabled and self._plateau(
            phase_round_count=count,
            min_rounds=self._effective_phase_setting(workload_interpretation, "min_db_rounds_hint", self.settings.min_db_rounds),
            consecutive_non_improving=non_improving,
            patience=self._effective_phase_setting(workload_interpretation, "db_plateau_patience_hint", self.settings.db_plateau_patience),
            consecutive_repeats=repeats,
        )
        model_stop = self.settings.enabled and self._should_follow_model_recommendation(
            recommendation=recommendation, current_phase=current_phase,
            target_phase=STOP_PHASE, phase_round_count=count,
        )
        stop = bool(plateau or model_stop)
        return AuditDecision(
            current_phase=current_phase, next_phase=STOP_PHASE if stop else DB_PHASE,
            action="stop" if stop else "continue_db",
            reason="数据库调优达到停止条件，保留已验证配置。" if stop else "继续数据库参数搜索。",
            stop=stop, plateau_detected=bool(plateau), phase_round_count=count,
            consecutive_non_improving=non_improving, consecutive_repeats=repeats,
            best_score=best_result.score, model_recommendation=recommendation,
            model_recommendation_used=bool(model_stop and not plateau),
        )

    def _round_phase(self, entry: dict[str, Any], fallback_phase: str) -> str:
        phase = (entry.get("metadata") or {}).get("phase", fallback_phase)
        action = (entry.get("proposal") or {}).get("action_type")
        return DB_PHASE if phase == DB_PHASE and action in {"db_config", "repeat_benchmark", None} else "unsupported"

    def _latest_model_recommendation(self, rounds: list[dict[str, Any]]) -> dict[str, Any] | None:
        for entry in reversed(rounds):
            recommendation = (entry.get("proposal") or {}).get("auditor_recommendation")
            if isinstance(recommendation, dict) and recommendation:
                return recommendation
        return None

    def _should_follow_model_recommendation(self, *, recommendation: dict[str, Any] | None,
            current_phase: str, target_phase: str, phase_round_count: int) -> bool:
        if not self.settings.respect_model_phase_recommendation or not recommendation:
            return False
        if phase_round_count < self.settings.model_recommendation_min_rounds:
            return False
        requested = str(recommendation.get("next_phase") or recommendation.get("recommended_next_phase")
                        or recommendation.get("phase") or "").lower()
        if requested in {"stop_run", "finish"}:
            requested = STOP_PHASE
        confidence = str(recommendation.get("confidence", "medium")).lower()
        return requested == target_phase == STOP_PHASE and current_phase == DB_PHASE and confidence not in {"low", "weak"}

    def _plateau(
        self,
        *,
        phase_round_count: int,
        min_rounds: int,
        consecutive_non_improving: int,
        patience: int,
        consecutive_repeats: int,
    ) -> bool:
        if phase_round_count < min_rounds:
            return False
        if consecutive_repeats >= self.settings.repeat_benchmark_patience:
            return True
        return consecutive_non_improving >= patience

    def _consecutive_non_improving(
        self,
        rounds: list[dict[str, Any]],
        *,
        count_repeat_benchmarks: bool = True,
    ) -> int:
        count = 0
        for entry in reversed(rounds):
            if entry.get("decision") == "accepted":
                break
            proposal = entry.get("proposal", {}) or {}
            if not count_repeat_benchmarks and proposal.get("action_type") == "repeat_benchmark":
                continue
            count += 1
        return count

    def _consecutive_repeats(self, rounds: list[dict[str, Any]]) -> int:
        count = 0
        for entry in reversed(rounds):
            proposal = entry.get("proposal", {}) or {}
            if proposal.get("action_type") != "repeat_benchmark":
                break
            count += 1
        return count

    def _effective_phase_setting(
        self,
        workload_interpretation: dict[str, Any] | None,
        key: str,
        configured_value: int,
    ) -> int:
        if not self.settings.use_workload_direction_policy:
            return configured_value
        policy = (workload_interpretation or {}).get("auditor_policy") or {}
        try:
            hinted_value = int(policy.get(key, configured_value))
        except (TypeError, ValueError):
            return configured_value
        return max(configured_value, hinted_value)
