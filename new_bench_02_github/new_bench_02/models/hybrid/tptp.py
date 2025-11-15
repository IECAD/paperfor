"""TPTP baseline implementation for the benchmark."""
from __future__ import annotations

import itertools
import math
from typing import Dict, Iterable

import numpy as np

from ..base import BaseModel, ModelOutput
from ..registry import register_model


@register_model("TPTP")
class TPTPBaseline(BaseModel):
    supports_gpu = False

    def __init__(
        self,
        anchors: Iterable[int] = (1, 2, 3, 4),
        lambda_thresh: float = 0.9,
        num_particles: int = 1500,
        sabo_iterations: int = 3,
        sample_sigma: float = 0.35,
        process_noise: float = 0.15,
        **kwargs,
    ) -> None:
        super().__init__(kwargs)
        self.anchors = tuple(int(a) for a in anchors)
        self.lambda_thresh = float(lambda_thresh)
        self.num_particles = int(num_particles)
        self.sabo_iterations = int(sabo_iterations)
        self.sample_sigma = float(sample_sigma)
        self.process_noise = float(process_noise)

        self.anchor_positions: Dict[int, np.ndarray] = {}
        self.measurement_noise: Dict[int, float] = {}
        self.bundle = None
        self.scaler_mean: np.ndarray | None = None
        self.scaler_scale: np.ndarray | None = None
        self.inference_meta: Dict[str, np.ndarray] | None = None

    def configure_from_bundle(self, bundle) -> None:  # type: ignore[override]
        self.bundle = bundle
        scaler = bundle.scaler
        mean = np.asarray(getattr(scaler, "mean_", np.zeros(20)), dtype=np.float64)
        scale = np.asarray(getattr(scaler, "scale_", np.ones_like(mean)), dtype=np.float64)
        scale = np.where(scale == 0, 1.0, scale)
        self.scaler_mean = mean
        self.scaler_scale = scale

    def set_inference_split(self, split) -> None:  # type: ignore[override]
        self.inference_meta = getattr(split, "meta", {}) or {}

    def fit(self, train, val=None) -> None:  # type: ignore[override]
        if train.X.size == 0:
            raise ValueError("Training data is empty.")
        if self.scaler_mean is None or self.scaler_scale is None:
            raise RuntimeError("Scaler statistics are not configured. Call configure_from_bundle first.")

        raw_features = self._inverse_scale(train.X)
        anchor_ids = (train.meta or {}).get("anchor_id", np.full(train.X.shape[0], -1, dtype=np.int32))
        self.anchor_positions.clear()
        self.measurement_noise.clear()

        for anchor in self.anchors:
            mask = anchor_ids == anchor
            if not np.any(mask):
                continue
            dists = raw_features[mask, 0].astype(np.float64)
            positions = train.y_pos[mask].astype(np.float64)
            anchor_pos = self._estimate_anchor_position(positions, dists)
            self.anchor_positions[anchor] = anchor_pos
            residuals = np.linalg.norm(positions - anchor_pos, axis=1) - dists
            sigma = float(np.std(residuals)) if residuals.size else 0.25
            self.measurement_noise[anchor] = max(sigma, 0.1)

        if not self.anchor_positions:
            raise RuntimeError("Unable to estimate anchor positions from training data.")

    def predict(self, features: np.ndarray) -> ModelOutput:  # noqa: D401
        if features.size == 0:
            return ModelOutput(position=np.zeros((0, 2), dtype=np.float32))
        if self.scaler_mean is None or self.scaler_scale is None:
            raise RuntimeError("Model not configured with scaler statistics.")
        if not self.anchor_positions:
            raise RuntimeError("Model must be fitted before inference.")

        raw = self._inverse_scale(features)
        meta = self.inference_meta or {}
        anchor_ids = meta.get("anchor_id", np.full(raw.shape[0], -1, dtype=np.int32))
        group_ids = meta.get("group_index", np.arange(raw.shape[0], dtype=np.int32))

        predictions = np.zeros((raw.shape[0], 2), dtype=np.float32)
        cache: Dict[int, np.ndarray] = {}

        for idx, row in enumerate(raw):
            group_id = int(group_ids[idx]) if idx < len(group_ids) else idx
            if group_id in cache:
                predictions[idx] = cache[group_id]
                continue

            main_anchor = int(anchor_ids[idx]) if idx < len(anchor_ids) and anchor_ids[idx] >= 0 else self.anchors[0]
            measurements = self._decode_measurements(row, main_anchor)
            coarse = self._mrwgh_estimate(measurements)
            adjusted = self._nlos_adjust(measurements, coarse)
            refined = self._refine(adjusted, coarse)
            cache[group_id] = refined.astype(np.float32)
            predictions[idx] = cache[group_id]

        return ModelOutput(position=predictions)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _inverse_scale(self, features: np.ndarray) -> np.ndarray:
        return (features * self.scaler_scale) + self.scaler_mean

    def _estimate_anchor_position(self, positions: np.ndarray, distances: np.ndarray) -> np.ndarray:
        A = np.column_stack((-2.0 * positions[:, 0], -2.0 * positions[:, 1], np.ones_like(distances)))
        b = distances ** 2 - (positions[:, 0] ** 2 + positions[:, 1] ** 2)
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
        return sol[:2]

    def _decode_measurements(self, row: np.ndarray, main_anchor: int) -> Dict[int, Dict[str, float]]:
        measurements: Dict[int, Dict[str, float]] = {}
        aux_idx = 0
        for anchor in self.anchors:
            if anchor == main_anchor:
                measurements[anchor] = {
                    "distance": float(max(row[0], 0.0)),
                    "snr": float(row[7]) if row.size > 7 else 0.0,
                    "cir": float(row[4]) if row.size > 4 else 0.0,
                }
                continue
            base = 8 + aux_idx * 4
            if base + 3 < row.size:
                measurements[anchor] = {
                    "distance": float(max(row[base], 0.0)),
                    "snr": float(row[base + 3]),
                    "cir": float(row[base + 2]),
                }
            aux_idx += 1
        return measurements

    def _mrwgh_estimate(self, measurements: Dict[int, Dict[str, float]]) -> np.ndarray:
        available = [anchor for anchor in measurements if anchor in self.anchor_positions]
        if not available:
            return np.zeros(2, dtype=np.float64)

        candidates = []
        weights = []
        if len(available) >= 3:
            for combo in itertools.combinations(available, 3):
                positions = np.array([self.anchor_positions[a] for a in combo], dtype=np.float64)
                dists = np.array([measurements[a]["distance"] for a in combo], dtype=np.float64)
                candidate = self._estimate_anchor_position(positions, dists)
                residuals = []
                for anchor, pos, dist in zip(combo, positions, dists):
                    residuals.append(abs(np.linalg.norm(candidate - pos) - dist))
                residual = np.mean(residuals) if residuals else 1.0
                weights.append(1.0 / (residual + 1e-6) ** 2)
                candidates.append(candidate)
        else:
            positions = np.array([self.anchor_positions[a] for a in available], dtype=np.float64)
            dists = np.array([measurements[a]["distance"] for a in available], dtype=np.float64)
            candidate = self._estimate_anchor_position(positions, dists)
            candidates.append(candidate)
            weights.append(1.0)

        weights = np.asarray(weights, dtype=np.float64)
        weights = weights / weights.sum()
        return np.average(np.array(candidates, dtype=np.float64), axis=0, weights=weights)

    def _nlos_adjust(self, measurements: Dict[int, Dict[str, float]], coarse: np.ndarray) -> Dict[int, Dict[str, float]]:
        adjusted = {}
        for anchor, info in measurements.items():
            if anchor not in self.anchor_positions:
                continue
            anchor_pos = self.anchor_positions[anchor]
            expected = np.linalg.norm(coarse - anchor_pos)
            sigma = self.measurement_noise.get(anchor, 0.25)
            if sigma <= 0:
                sigma = 0.25
            residual = info["distance"] - expected
            z = residual / (sigma * math.sqrt(2.0))
            los_prob = 0.5 * (1.0 + math.erf(z))
            is_nlos = los_prob > self.lambda_thresh
            adjusted_distance = expected if is_nlos else info["distance"]
            adjusted[anchor] = {
                "distance": adjusted_distance,
                "sigma": sigma * (2.0 if is_nlos else 1.0),
            }
        return adjusted

    def _sample_particles(self, coarse: np.ndarray, measurement_info: Dict[int, Dict[str, float]]) -> np.ndarray:
        particles = coarse + np.random.normal(scale=self.sample_sigma, size=(self.num_particles, 2))
        mask = np.ones(self.num_particles, dtype=bool)
        for anchor, info in measurement_info.items():
            anchor_pos = self.anchor_positions.get(anchor)
            if anchor_pos is None:
                continue
            radius = info["distance"] + 2.0 * info.get("sigma", self.process_noise)
            dist = np.linalg.norm(particles - anchor_pos, axis=1)
            mask &= dist <= (radius + 0.05)
        if not np.any(mask):
            mask = slice(None)
        filtered = particles[mask]
        if filtered.shape[0] < self.num_particles:
            extras = coarse + np.random.normal(scale=self.sample_sigma, size=(self.num_particles - filtered.shape[0], 2))
            filtered = np.vstack([filtered, extras])
        return filtered

    def _likelihood(self, particles: np.ndarray, measurement_info: Dict[int, Dict[str, float]]) -> np.ndarray:
        weights = np.ones(particles.shape[0], dtype=np.float64)
        for anchor, info in measurement_info.items():
            anchor_pos = self.anchor_positions.get(anchor)
            if anchor_pos is None:
                continue
            expected = np.linalg.norm(particles - anchor_pos, axis=1)
            residual = info["distance"] - expected
            sigma = info.get("sigma", self.process_noise) + 1e-6
            weights *= np.exp(-0.5 * (residual / sigma) ** 2)
        total = weights.sum()
        if not np.isfinite(total) or total <= 0:
            return np.full(particles.shape[0], 1.0 / particles.shape[0], dtype=np.float64)
        return weights / total

    def _refine(self, measurement_info: Dict[int, Dict[str, float]], coarse: np.ndarray) -> np.ndarray:
        particles = self._sample_particles(coarse, measurement_info)
        if particles.shape[0] == 0:
            return coarse.astype(np.float64)
        for _ in range(max(1, self.sabo_iterations)):
            weights = self._likelihood(particles, measurement_info)
            elite_count = max(int(0.1 * len(particles)), 1)
            elite_idx = np.argsort(weights)[-elite_count:]
            elite = particles[elite_idx]
            mean_elite = elite.mean(axis=0)
            perturb = elite - mean_elite
            particles = elite + perturb * 0.5 + np.random.normal(scale=self.process_noise, size=elite.shape)
        weights = self._likelihood(particles, measurement_info)
        return np.average(particles, axis=0, weights=weights)

