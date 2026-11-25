from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


@dataclass(frozen=True)
class Prediction:
    mean: float
    latent_std: float
    observation_std: float
    model_version: int

    def lower(self, beta: float) -> float:
        return self.mean - beta * self.latent_std

    def upper(self, beta: float) -> float:
        return self.mean + beta * self.latent_std


class GaussianProcess:
    """Small exact GP with joint Matérn or an explicit additive-kernel ablation.

    Targets are raw log-performance, not stage-normalized ratios. Optimization
    uses bounded LML, Cholesky solves, separate noise and numerical jitter.
    """

    def __init__(self, *, kernel: str = "joint", config_dimensions: int = 1, seed: int = 0):
        if kernel not in {"joint", "additive"}:
            raise ValueError("kernel must be joint or additive")
        self.kernel, self.config_dimensions, self.seed = kernel, config_dimensions, seed
        self.theta = np.log([0.5, 1.0, 0.05])  # length, amplitude, observation std
        self.version = 0
        self.fitted = False

    @staticmethod
    def matern(a, b, length):
        r = np.sqrt(np.maximum(((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=2), 0)) / length
        return (1 + math.sqrt(5) * r + 5 * r**2 / 3) * np.exp(-math.sqrt(5) * r)

    def covariance(self, a, b, theta=None):
        length, amplitude, _ = np.exp(self.theta if theta is None else theta)
        if self.kernel == "joint":
            return amplitude**2 * self.matern(a, b, length)
        d = self.config_dimensions
        return amplitude**2 * (self.matern(a[:, :d], b[:, :d], length) + a[:, d:] @ b[:, d:].T)

    def fit(self, x, y, *, optimize: bool = True, fixed_noise_log_std: float = 0.0):
        # 显式可选：在训练协方差中加入固定的观测方差，而非事后扩大误差条。
        # 参数单位与输入y一致（本项目为原始对数目标），不随y标准化尺度漂移。
        if (isinstance(fixed_noise_log_std, bool) or not isinstance(fixed_noise_log_std, (int, float))
                or not math.isfinite(fixed_noise_log_std) or fixed_noise_log_std < 0):
            raise ValueError("Fixed observation noise must be finite, nonnegative log-target units")
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        if (
            x.ndim != 2 or y.shape != (len(x),) or len(x) < 2
            or not np.isfinite(x).all() or not np.isfinite(y).all()
        ):
            raise ValueError("At least two finite compatible observations required")
        self.x = x
        self.y = y.copy()
        self.y_mean, self.y_scale = float(y.mean()), max(float(y.std()), 0.01)
        self.fixed_noise_log_std = float(fixed_noise_log_std)
        fixed_scaled = self.fixed_noise_log_std/self.y_scale
        if not math.isfinite(fixed_scaled) or fixed_scaled > math.sqrt(np.finfo(float).max)/2:
            raise ValueError("Fixed noise variance is not numerically representable")
        centered = (y - self.y_mean) / self.y_scale

        def decompose(theta):
            noise_var = np.exp(theta[2])**2 + fixed_scaled**2
            k = self.covariance(x, x, theta) + (noise_var + 1e-9) * np.eye(len(x))
            lower = np.linalg.cholesky(k)
            alpha = np.linalg.solve(lower.T, np.linalg.solve(lower, centered))
            return lower, alpha

        def nll(theta):
            try:
                lower, alpha = decompose(theta)
                return float(0.5 * centered @ alpha + np.log(np.diag(lower)).sum() + len(x) * math.log(2 * math.pi) / 2)
            except np.linalg.LinAlgError:
                return 1e30

        if optimize:
            rng = np.random.default_rng(self.seed)
            starts = [self.theta, self.theta + rng.normal(0, 0.25, 3)]
            bounds = [(-4, 4), (-4, 4), (-7, 0)]
            fits = [minimize(nll, start, method="L-BFGS-B", bounds=bounds, options={"maxiter": 60}) for start in starts]
            usable = [f for f in fits if np.isfinite(f.fun) and f.fun < 1e29]
            if usable:
                best = min(usable, key=lambda f: f.fun)
                self.theta = best.x
                self.optimizer = {"optimized": True, "success": bool(best.success),
                                  "negative_log_marginal_likelihood": float(best.fun),
                                  "iterations": int(best.nit)}
            else:
                self.optimizer = {"optimized": True, "success": False}
        else:
            self.optimizer = {"optimized": False, "reused_hyperparameters": True}
        self.lower, self.alpha = decompose(self.theta)
        self.version += 1
        self.fitted = True

    def snapshot(self, training_ids: list[str]) -> dict:
        if not self.fitted or len(training_ids) != len(self.x):
            raise ValueError("Snapshot requires an exact ordered training-ID list")
        payload = {
            "version": self.version, "kernel": self.kernel, "config_dimensions": self.config_dimensions,
            "input_dimensions": self.x.shape[1], "training_ids": training_ids,
            "x": self.x.tolist(), "target": self.y.tolist(),
            "target_mean": self.y_mean, "target_scale": self.y_scale,
            "log_hyperparameters": self.theta.tolist(),
            "hyperparameters": dict(zip(["length_scale", "amplitude", "noise_std"], np.exp(self.theta).tolist())),
            "jitter": 1e-9, "seed": self.seed, "optimizer": self.optimizer,
        }
        if self.fixed_noise_log_std > 0:
            payload["noise_policy"] = {
                "parameterization": "residual_plus_fixed_variance_v1",
                "fixed_noise_log_std": self.fixed_noise_log_std,
                "effective_noise_log_std": math.hypot(
                    self.y_scale*math.exp(self.theta[2]), self.fixed_noise_log_std)}
        return payload

    @classmethod
    def from_snapshot(cls, payload: dict):
        theta = np.asarray(payload["log_hyperparameters"], dtype=float)
        if theta.shape != (3,) or not np.isfinite(theta).all():
            raise ValueError("Invalid GP hyperparameters")
        model = cls(kernel=payload["kernel"], config_dimensions=int(payload["config_dimensions"]),
                    seed=int(payload["seed"]))
        model.theta = theta
        noise = payload.get("noise_policy")
        if noise is not None and (
                not isinstance(noise, dict) or set(noise) != {
                    "parameterization", "fixed_noise_log_std", "effective_noise_log_std"}
                or noise["parameterization"] != "residual_plus_fixed_variance_v1"
                or isinstance(noise["effective_noise_log_std"], bool)
                or not isinstance(noise["effective_noise_log_std"], (int, float))
                or not math.isfinite(noise["effective_noise_log_std"])):
            raise ValueError("Unknown snapshot noise parameterization")
        model.fit(payload["x"], payload["target"], optimize=False,
                  fixed_noise_log_std=0.0 if noise is None else noise["fixed_noise_log_std"])
        if noise is not None and not math.isclose(noise["effective_noise_log_std"],
                math.hypot(model.y_scale*math.exp(model.theta[2]), model.fixed_noise_log_std),
                rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("Snapshot observation noise mismatch")
        if (len(payload["training_ids"]) != len(model.x)
                or model.x.shape[1] != payload["input_dimensions"]
                or not math.isclose(model.y_mean, payload["target_mean"], abs_tol=1e-12)
                or not math.isclose(model.y_scale, payload["target_scale"], abs_tol=1e-12)):
            raise ValueError("Snapshot training schema or normalization mismatch")
        model.version = int(payload["version"])
        model.optimizer = payload["optimizer"]
        return model

    def predict(self, x) -> list[Prediction]:
        if not self.fitted:
            raise ValueError("GP not bootstrapped")
        x = np.atleast_2d(np.asarray(x, dtype=float))
        if x.shape[1] != self.x.shape[1] or not np.isfinite(x).all():
            raise ValueError("Incompatible GP input encoding")
        cross = self.covariance(self.x, x)
        mean = self.y_mean + self.y_scale * (cross.T @ self.alpha)
        projection = np.linalg.solve(self.lower, cross)
        variance = np.maximum(np.diag(self.covariance(x, x)) - (projection**2).sum(axis=0), 0)
        noise_var = np.exp(self.theta[2])**2 + (self.fixed_noise_log_std/self.y_scale)**2
        return [
            Prediction(float(m), float(self.y_scale * np.sqrt(v)),
                       float(self.y_scale * np.sqrt(v + noise_var)), self.version)
            for m, v in zip(mean, variance)
        ]
