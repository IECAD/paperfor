"""Extended Kalman Filter baseline."""
from __future__ import annotations

import numpy as np

from ..base import BaseModel, ModelOutput
from ..registry import register_model


@register_model("EKF")
class ExtendedKalmanFilterModel(BaseModel):
    supports_gpu = False
    requires_fit = True

    def __init__(
        self,
        process_noise: float = 0.05,
        measurement_noise: float | None = None,
        dt: float = 1.0,
        ridge: float = 1e-6,
        **kwargs,
    ) -> None:
        super().__init__(kwargs)
        self.dt = dt
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.ridge = ridge
        self.linear_coef: np.ndarray | None = None
        self.linear_bias: np.ndarray | None = None
        self.R: np.ndarray | None = None

    def fit(self, train, val=None) -> None:  # noqa: D401
        if train.X.size == 0:
            raise ValueError("Training data is empty.")
        features = np.asarray(train.X, dtype=np.float64)
        if np.isnan(features).any():
            col_means = np.nanmean(features, axis=0)
            col_means = np.where(np.isnan(col_means), 0.0, col_means)
            inds = np.where(np.isnan(features))
            features[inds] = col_means[inds[1]]
        X = np.hstack([features, np.ones((features.shape[0], 1), dtype=np.float64)])
        y = np.asarray(train.y_pos, dtype=np.float64)
        try:
            theta, *_ = np.linalg.lstsq(X, y, rcond=self.ridge)
        except np.linalg.LinAlgError:
            pinv = np.linalg.pinv(X, rcond=self.ridge)
            theta = pinv @ y
        if np.isnan(theta).any():
            pinv = np.linalg.pinv(X, rcond=self.ridge)
            theta = pinv @ y
        self.linear_coef = theta[:-1]
        self.linear_bias = theta[-1]

        residuals = y - (features @ self.linear_coef + self.linear_bias)
        cov = residuals.T @ residuals / max(1, residuals.shape[0] - 1)
        base_noise = np.eye(2) * 1e-2
        self.R = cov + base_noise if self.measurement_noise is None else np.eye(2) * self.measurement_noise

    def _linear_measurement(self, features: np.ndarray) -> np.ndarray:
        if self.linear_coef is None or self.linear_bias is None:
            raise RuntimeError("Model is not fitted.")
        return features @ self.linear_coef + self.linear_bias

    def predict(self, features: np.ndarray) -> ModelOutput:
        X = np.asarray(features, dtype=np.float64)
        if np.isnan(X).any():
            col_means = np.nanmean(X, axis=0)
            col_means = np.where(np.isnan(col_means), 0.0, col_means)
            inds = np.where(np.isnan(X))
            X[inds] = col_means[inds[1]]
        measurements = self._linear_measurement(X)
        n_samples = measurements.shape[0]
        if n_samples == 0:
            return ModelOutput(position=np.zeros((0, 2), dtype=features.dtype))

        F = np.array(
            [
                [1.0, 0.0, self.dt, 0.0],
                [0.0, 1.0, 0.0, self.dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
        H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=float)
        Q = np.eye(4) * self.process_noise
        R = self.R if self.R is not None else np.eye(2) * 0.05

        state = np.zeros(4, dtype=float)
        state[:2] = measurements[0]
        P = np.eye(4)

        outputs = np.zeros((n_samples, 2), dtype=float)
        for idx, z in enumerate(measurements):
            state = F @ state
            P = F @ P @ F.T + Q

            y_res = z - H @ state
            S = H @ P @ H.T + R
            K = P @ H.T @ np.linalg.inv(S)
            state = state + K @ y_res
            P = (np.eye(4) - K @ H) @ P
            outputs[idx] = state[:2]

        return ModelOutput(position=outputs)
