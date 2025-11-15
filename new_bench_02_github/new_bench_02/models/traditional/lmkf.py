"""Levenberg-Marquardt Kalman Filter baseline."""
from __future__ import annotations

import numpy as np

from ..base import BaseModel, ModelOutput
from ..registry import register_model
from .ekf import ExtendedKalmanFilterModel


@register_model("LMKF")
class LevenbergMarquardtKalmanModel(BaseModel):
    """Simplified LM-KF that reuses EKF with adaptive damping."""

    supports_gpu = False

    def __init__(self, damping: float = 1.0, damping_decay: float = 0.95, **kwargs) -> None:
        super().__init__(kwargs)
        self.damping = damping
        self.damping_decay = damping_decay
        self._ekf = ExtendedKalmanFilterModel(process_noise=0.05, measurement_noise=None, dt=1.0)

    def fit(self, train, val=None) -> None:  # noqa: D401
        self._ekf.fit(train, val)
        if self.damping <= 0:
            self.damping = 1.0

    def predict(self, features: np.ndarray) -> ModelOutput:
        if features.size == 0:
            return ModelOutput(position=np.zeros((0, 2), dtype=features.dtype))
        X = np.asarray(features, dtype=np.float64)
        if np.isnan(X).any():
            col_means = np.nanmean(X, axis=0)
            col_means = np.where(np.isnan(col_means), 0.0, col_means)
            inds = np.where(np.isnan(X))
            X[inds] = col_means[inds[1]]
        base = self._ekf._linear_measurement(X)
        corrections = np.zeros_like(base)
        damping = self.damping
        for _ in range(3):
            corrections += (base - corrections) * (1.0 / (1.0 + damping))
            damping *= self.damping_decay
        return ModelOutput(position=base - corrections)
