"""Ordinary least squares baseline."""
from __future__ import annotations

import numpy as np

from ..base import BaseModel, ModelOutput
from ..registry import register_model


@register_model("LS")
class LeastSquaresModel(BaseModel):
    supports_gpu = False
    requires_fit = True

    def __init__(self, ridge: float = 1e-6, **kwargs) -> None:
        super().__init__(kwargs)
        self.coef_: np.ndarray | None = None
        self.ridge = ridge

    def fit(self, train, val=None) -> None:  # noqa: D401
        if train.X.size == 0:
            raise ValueError("Training data is empty.")
        X = np.asarray(train.X, dtype=np.float64)
        if np.isnan(X).any():
            col_means = np.nanmean(X, axis=0)
            col_means = np.where(np.isnan(col_means), 0.0, col_means)
            inds = np.where(np.isnan(X))
            X[inds] = col_means[inds[1]]
        ones = np.ones((X.shape[0], 1), dtype=np.float64)
        X = np.hstack([X, ones])
        y = np.asarray(train.y_pos, dtype=np.float64)
        try:
            self.coef_, *_ = np.linalg.lstsq(X, y, rcond=self.ridge)
        except np.linalg.LinAlgError:
            pinv = np.linalg.pinv(X, rcond=self.ridge)
            self.coef_ = pinv @ y
        if np.isnan(self.coef_).any():
            pinv = np.linalg.pinv(X, rcond=self.ridge)
            self.coef_ = pinv @ y

    def predict(self, features: np.ndarray) -> ModelOutput:
        if self.coef_ is None:
            raise RuntimeError("Model is not fitted.")
        X = np.asarray(features, dtype=np.float64)
        if np.isnan(X).any():
            col_means = np.nanmean(X, axis=0)
            col_means = np.where(np.isnan(col_means), 0.0, col_means)
            inds = np.where(np.isnan(X))
            X[inds] = col_means[inds[1]]
        X = np.hstack([X, np.ones((X.shape[0], 1), dtype=X.dtype)])
        preds = X @ self.coef_
        return ModelOutput(position=preds)
