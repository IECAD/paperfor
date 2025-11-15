"""Filtering utilities for UWB ranging data."""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats


class UWBFilter:
    """Collection of reusable filtering strategies for raw UWB measurements."""

    @staticmethod
    def snr_weighted_filter(data: pd.DataFrame, window_size: int = 3) -> pd.DataFrame:
        """Apply an SNR weighted moving average per anchor/point trajectory."""
        if window_size < 1:
            raise ValueError("window_size must be >= 1")
        filtered = data.copy()
        if filtered.empty:
            return filtered

        required_cols = {"point_id", "ID", "DIST", "SNR"}
        missing = required_cols - set(filtered.columns)
        if missing:
            raise KeyError(f"Missing columns for SNR filter: {sorted(missing)}")

        time_col = "timestamp" if "timestamp" in filtered.columns else None

        for point_id in filtered["point_id"].unique():
            point_slice = filtered[filtered["point_id"] == point_id]
            for anchor_id in point_slice["ID"].unique():
                mask = (filtered["point_id"] == point_id) & (filtered["ID"] == anchor_id)
                subset = filtered.loc[mask]
                if time_col:
                    subset = subset.sort_values(time_col)

                if len(subset) < window_size:
                    continue

                weights = np.clip(subset["SNR"].to_numpy() / 8.0, 0.1, 1.0)
                distances = subset["DIST"].to_numpy()
                filtered_values = np.empty_like(distances)

                for idx in range(len(distances)):
                    start = max(0, idx - window_size // 2)
                    end = min(len(distances), idx + window_size // 2 + 1)
                    window_dist = distances[start:end]
                    window_weights = weights[start:end]
                    filtered_values[idx] = np.average(window_dist, weights=window_weights)

                filtered.loc[mask, "DIST"] = filtered_values
        return filtered

    @staticmethod
    def combo_filter(
        data: pd.DataFrame,
        window_size: int = 3,
        z_threshold: float = 2.5,
        min_snr: float = 1.0,
        min_cir: float = 30.0,
    ) -> pd.DataFrame:
        """Quality-gated filter chaining denoising heuristics for UWB ranges."""
        if data.empty:
            return data.copy()

        filtered = data.copy()
        quality_mask = (filtered["SNR"] >= min_snr) & (filtered["CIR"] >= min_cir)

        for point_id in filtered["point_id"].unique():
            point_slice = filtered[filtered["point_id"] == point_id]
            for anchor_id in point_slice["ID"].unique():
                mask = (filtered["point_id"] == point_id) & (filtered["ID"] == anchor_id)
                subset = filtered.loc[mask]
                if subset.empty:
                    continue

                hq_mask = quality_mask.loc[mask]
                if hq_mask.any():
                    replacement = subset.loc[hq_mask, "DIST"].mean()
                    filtered.loc[mask & ~quality_mask, "DIST"] = replacement

                if len(subset) > 3:
                    distances = subset["DIST"].to_numpy(dtype=float)
                    if np.all(np.isnan(distances)):
                        continue
                    z_scores = np.abs(stats.zscore(distances, nan_policy="omit"))
                    if np.isnan(z_scores).all():
                        continue
                    outlier_mask = np.nan_to_num(z_scores, nan=0.0) > z_threshold
                    if np.any(outlier_mask):
                        clean_values = distances[~outlier_mask]
                        clean_values = clean_values[~np.isnan(clean_values)]
                        if clean_values.size == 0:
                            continue
                        clean_mean = float(clean_values.mean())
                        filtered.loc[subset.index[outlier_mask], "DIST"] = clean_mean

        return UWBFilter.snr_weighted_filter(filtered, window_size=window_size)
