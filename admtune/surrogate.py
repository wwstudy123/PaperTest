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

    def fit(self, x, y, *, optimize: bool = True):
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        if (
            x.ndim != 2 or y.shape != (len(x),) or len(x) < 2
            or not np.isfinite(x).all() or not np.isfinite(y).all()
        ):
            raise ValueError("At least two finite compatible observations required")
        self.x = x
        self.y_mean, self.y_scale = float(y.mean()), max(float(y.std()), 0.01)
        centered = (y - self.y_mean) / self.y_scale

        def decompose(theta):
            noise = np.exp(theta[2])
            k = self.covariance(x, x, theta) + (noise**2 + 1e-9) * np.eye(len(x))
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
                self.theta = min(usable, key=lambda f: f.fun).x
        self.lower, self.alpha = decompose(self.theta)
        self.version += 1
        self.fitted = True

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
        noise = np.exp(self.theta[2])
        return [
            Prediction(float(m), float(self.y_scale * np.sqrt(v)),
                       float(self.y_scale * np.sqrt(v + noise**2)), self.version)
            for m, v in zip(mean, variance)
        ]
