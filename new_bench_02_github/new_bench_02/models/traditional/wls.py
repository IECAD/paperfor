"""Weighted least squares baseline."""
from __future__ import annotations

import numpy as np

from ..base import BaseModel, ModelOutput
from ..registry import register_model


@register_model("WLS")
class WeightedLeastSquaresModel(BaseModel):
    supports_gpu = False
    requires_fit = True

    def __init__(self, snr_index: int = 7, ridge: float = 1e-6, **kwargs) -> None:
        super().__init__(kwargs)
        self.snr_index = snr_index
        self.ridge = ridge
        self.theta_: np.ndarray | None = None

    def fit(self, train, val=None) -> None:  # noqa: D401
        if train.X.size == 0:
            raise ValueError("Training data is empty.")
        features = np.asarray(train.X, dtype=np.float64)
        if np.isnan(features).any():
            col_means = np.nanmean(features, axis=0)
            col_means = np.where(np.isnan(col_means), 0.0, col_means)
            inds = np.where(np.isnan(features))
            features[inds] = col_means[inds[1]]
        snr = features[:, self.snr_index]
        snr = np.where(np.isnan(snr), np.nanmean(snr), snr)
        snr = np.where(np.isnan(snr), 1.0, snr)
        snr = np.clip(snr, 0.05, None)
        weights = np.sqrt(snr)[:, None]
        X = np.hstack([features, np.ones((features.shape[0], 1), dtype=features.dtype)])
        Xw = X * weights
        y = np.asarray(train.y_pos, dtype=np.float64)
        yw = y * weights
        try:
            self.theta_, *_ = np.linalg.lstsq(Xw, yw, rcond=self.ridge)
        except np.linalg.LinAlgError:
            pinv = np.linalg.pinv(Xw, rcond=self.ridge)
            self.theta_ = pinv @ yw
        if np.isnan(self.theta_).any():
            pinv = np.linalg.pinv(Xw, rcond=self.ridge)
            self.theta_ = pinv @ yw

    def predict(self, features: np.ndarray) -> ModelOutput:
        if self.theta_ is None:
            raise RuntimeError("Model is not fitted.")
        X = np.asarray(features, dtype=np.float64)
        if np.isnan(X).any():
            col_means = np.nanmean(X, axis=0)
            col_means = np.where(np.isnan(col_means), 0.0, col_means)
            inds = np.where(np.isnan(X))
            X[inds] = col_means[inds[1]]
        X = np.hstack([X, np.ones((X.shape[0], 1), dtype=X.dtype)])
        preds = X @ self.theta_
        return ModelOutput(position=preds)
