"""只用于离线时间切分的观测方差校准；不修改在线GP、均值或候选筛选。"""
from collections import defaultdict
from dataclasses import replace
import math
import statistics

from .contracts import fingerprint
from .surrogate import GaussianProcess


def validate_calibration_rows(rows):
    if len(rows) != 6 or [r["seq"] for r in rows] != list(range(1, 7)):
        raise ValueError("校准只能使用预先固定的前6条，禁止检验点进入拟合")
    groups = defaultdict(list)
    for row in rows:
        value = row["objective_value"]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError("需要有限正数规范目标")
        groups[fingerprint(row["config"])].append(math.log(value))
    if len(groups) != 3 or any(len(values) != 2 for values in groups.values()):
        raise ValueError("需要三个完整配置、各两次观测")
    return groups


def pooled_log_variance(rows):
    groups = validate_calibration_rows(rows)
    df = len(rows)-len(groups)
    sse = sum(sum((y-statistics.mean(ys))**2 for y in ys) for ys in groups.values())
    return {"variance": sse/df, "std": math.sqrt(sse/df), "sse": sse,
            "degrees_of_freedom": df, "training_sequences": [r["seq"] for r in rows]}


def refit_first_six(space, rows):
    validate_calibration_rows(rows)
    model = GaussianProcess(config_dimensions=len(space.encode(rows[0]["config"])), seed=42)
    model.fit([space.encode(r["config"])+[0.0] for r in rows],
              [math.log(r["objective_value"]) for r in rows], optimize=True)
    return model


def with_noise_floor(prediction, variance):
    values = (variance, prediction.mean, prediction.latent_std, prediction.observation_std)
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values):
        raise ValueError("方差与预测必须有限")
    if variance < 0 or prediction.latent_std < 0 or prediction.observation_std < prediction.latent_std:
        raise ValueError("方差或标准差不合法")
    noise_var = max(0.0, prediction.observation_std**2-prediction.latent_std**2)
    return replace(prediction, observation_std=math.sqrt(
        prediction.latent_std**2+max(noise_var, variance)))


def interval_row(prediction, actual):
    if not math.isfinite(actual) or actual <= 0:
        raise ValueError("检验目标必须有限为正")
    return {"actual_objective": actual, "predicted_objective": math.exp(prediction.mean),
            "latent_log_std": prediction.latent_std,
            "observation_log_std": prediction.observation_std,
            "lower": math.exp(prediction.mean-2*prediction.observation_std),
            "upper": math.exp(prediction.mean+2*prediction.observation_std),
            "covered": abs(math.log(actual)-prediction.mean) <= 2*prediction.observation_std,
            "absolute_percent_error": abs(math.exp(prediction.mean)/actual-1)*100}
