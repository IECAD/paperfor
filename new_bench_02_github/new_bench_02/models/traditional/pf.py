"""Particle filter baseline."""
from __future__ import annotations

import numpy as np

from ..base import BaseModel, ModelOutput
from ..registry import register_model


@register_model("PF")
class ParticleFilterModel(BaseModel):
    supports_gpu = False

    def __init__(self, n_particles: int = 500, process_noise: float = 0.05, **kwargs) -> None:
        super().__init__(kwargs)
        self.n_particles = n_particles
        self.process_noise = process_noise
        self._particles: np.ndarray | None = None
        self._weights: np.ndarray | None = None
        self._reference: np.ndarray | None = None

    def fit(self, train, val=None) -> None:  # noqa: D401
        if train.X.size == 0:
            raise ValueError("Training data is empty.")
        features = np.asarray(train.X, dtype=np.float64)
        if np.isnan(features).any():
            col_means = np.nanmean(features, axis=0)
            col_means = np.where(np.isnan(col_means), 0.0, col_means)
            inds = np.where(np.isnan(features))
            features[inds] = col_means[inds[1]]
        feature_means = features.mean(axis=1).ravel()
        global_mean = feature_means.mean()
        self._reference = np.column_stack([feature_means, np.repeat(global_mean, features.shape[0])])
        mins = np.nanmin(train.y_pos, axis=0)
        maxs = np.nanmax(train.y_pos, axis=0)
        self._particles = np.random.uniform(mins, maxs, size=(self.n_particles, 2))
        self._weights = np.ones(self.n_particles) / self.n_particles

    def predict(self, features: np.ndarray) -> ModelOutput:
        if self._particles is None or self._weights is None:
            raise RuntimeError("Particle filter not fitted.")
        if features.size == 0:
            return ModelOutput(position=np.zeros((0, 2), dtype=features.dtype))

        X = np.asarray(features, dtype=np.float64)
        if np.isnan(X).any():
            col_means = np.nanmean(X, axis=0)
            col_means = np.where(np.isnan(col_means), 0.0, col_means)
            inds = np.where(np.isnan(X))
            X[inds] = col_means[inds[1]]

        n_samples = X.shape[0]
        outputs = np.zeros((n_samples, 2), dtype=float)
        for idx in range(n_samples):
            measurement = np.array([
                X[idx, 0],
                X[idx, 1] if X.shape[1] > 1 else X[idx, 0],
            ])
            noise = np.random.normal(scale=self.process_noise, size=self._particles.shape)
            self._particles = self._particles + noise
            dists = np.linalg.norm(self._particles - measurement, axis=1)
            likelihood = np.exp(-0.5 * (dists / (self.process_noise + 1e-6)) ** 2)
            self._weights *= likelihood
            self._weights += 1e-12
            total = self._weights.sum()
            if not np.isfinite(total) or total <= 0:
                self._weights = np.ones(self.n_particles) / self.n_particles
            else:
                self._weights /= total
            outputs[idx] = np.average(self._particles, axis=0, weights=self._weights)
            indices = np.random.choice(self.n_particles, size=self.n_particles, p=self._weights)
            self._particles = self._particles[indices]
            self._weights = np.ones(self.n_particles) / self.n_particles
        return ModelOutput(position=outputs)
