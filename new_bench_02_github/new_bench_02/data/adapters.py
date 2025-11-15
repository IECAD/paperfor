"""Input adaptation utilities for multi-phase experiments."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection


@dataclass
class AdapterConfig:
    name: str
    target_dim: int
    method: str = "select"
    kwargs: Dict[str, float] | None = None


class AdapterArtifact:
    def __init__(self, config: AdapterConfig) -> None:
        self.config = config
        self.transformer: Optional[object] = None

    def fit(self, X: np.ndarray) -> None:
        method = self.config.method
        target_dim = self.config.target_dim
        if method == "pca":
            self.transformer = PCA(n_components=min(target_dim, X.shape[1]))
            self.transformer.fit(X)
        elif method == "random_proj":
            self.transformer = GaussianRandomProjection(
                n_components=min(target_dim, X.shape[1]), random_state=42
            )
            self.transformer.fit(X)
        else:
            self.transformer = None

    def transform(self, X: np.ndarray) -> np.ndarray:
        method = self.config.method
        target_dim = self.config.target_dim
        if method == "identity":
            return X.astype(np.float32, copy=True)
        if method == "pca" and isinstance(self.transformer, PCA):
            transformed = self.transformer.transform(X)
            if transformed.shape[1] < target_dim:
                return np.pad(
                    transformed,
                    ((0, 0), (0, target_dim - transformed.shape[1])),
                    mode="edge",
                )
            return transformed.astype(np.float32, copy=False)
        if method == "random_proj" and isinstance(self.transformer, GaussianRandomProjection):
            return self.transformer.transform(X).astype(np.float32, copy=False)
        if method == "tile":
            reps = max(1, target_dim // max(1, X.shape[1]) + 1)
            tiled = np.tile(X, (1, reps))
            return tiled[:, :target_dim].astype(np.float32, copy=False)
        # default: select/pad
        if target_dim <= X.shape[1]:
            return X[:, :target_dim].astype(np.float32, copy=False)
        return np.pad(X, ((0, 0), (0, target_dim - X.shape[1])), mode="edge").astype(
            np.float32, copy=False
        )


class InputAdapterManager:
    """Registry for feature adapters used across experiment phases."""

    def __init__(self, profiles: Dict[str, AdapterConfig] | None = None) -> None:
        if profiles is None:
            profiles = {
                "low_dim": AdapterConfig(name="low_dim", target_dim=4, method="select"),
                "mid_dim": AdapterConfig(name="mid_dim", target_dim=10, method="select"),
                "high_dim": AdapterConfig(name="high_dim", target_dim=20, method="identity"),
                "cir_496": AdapterConfig(name="cir_496", target_dim=20, method="identity"),
                "cir_991": AdapterConfig(name="cir_991", target_dim=20, method="identity"),
                "cir_1016": AdapterConfig(name="cir_1016", target_dim=20, method="identity"),
                "base_8d": AdapterConfig(name="base_8d", target_dim=20, method="identity"),
                "base_no_snr": AdapterConfig(name="base_no_snr", target_dim=20, method="identity"),
            }
        self.artifacts: Dict[str, AdapterArtifact] = {
            name: AdapterArtifact(config) for name, config in profiles.items()
        }

    def fit(self, X: np.ndarray) -> None:
        for artifact in self.artifacts.values():
            artifact.fit(X)

    def transform(self, X: np.ndarray, profile: str) -> np.ndarray:
        if profile not in self.artifacts:
            raise KeyError(f"Unknown adapter profile '{profile}'.")
        result = self.artifacts[profile].transform(X)
        if profile == "base_no_snr":
            result = result.copy()
            snr_indices = [7, 11, 15, 19]
            for idx in snr_indices:
                if idx < result.shape[1]:
                    result[:, idx] = 0.0
        elif profile == "base_8d":
            result = result.copy()
            if result.shape[1] > 8:
                result[:, 8:] = 0.0
        return result.astype(np.float32, copy=False)

    def available_profiles(self) -> Dict[str, AdapterConfig]:
        return {name: artifact.config for name, artifact in self.artifacts.items()}


def build_default_manager() -> InputAdapterManager:
    return InputAdapterManager()
