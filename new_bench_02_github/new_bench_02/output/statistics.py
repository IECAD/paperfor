"""Statistical testing utilities."""
from __future__ import annotations

import argparse
from math import sqrt
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from ..paths import package_path

REFERENCE_MODEL = "Proposed"
REFERENCE_PROFILE = "base"
DEFAULT_RESULTS_FILE = package_path("3_output", "results", "experiment_metrics.csv")


def cliffs_delta(proposed: np.ndarray, baseline: np.ndarray) -> float:
    if proposed.shape != baseline.shape:
        raise ValueError("Cliff's delta requires paired samples of equal length.")
    diff = proposed - baseline
    n = len(diff)
    if n == 0:
        raise ValueError("No samples provided for Cliff's delta.")
    pos = float((diff > 0).sum())
    neg = float((diff < 0).sum())
    return (pos - neg) / n


def wilcoxon_test(proposed: np.ndarray, baseline: np.ndarray) -> Dict[str, float]:
    if proposed.shape != baseline.shape:
        raise ValueError("Wilcoxon test requires paired samples of equal length.")
    if proposed.size == 0:
        raise ValueError("No samples provided for Wilcoxon test.")

    statistic, p_value = stats.wilcoxon(
        proposed, baseline, zero_method="wilcox", alternative="two-sided"
    )
    diff = proposed - baseline
    effect_size = np.mean(diff)
    std_diff = np.std(diff, ddof=1)
    cohen_d = effect_size / std_diff if std_diff > 0 else float("inf")
    z_score = stats.norm.ppf(1 - p_value / 2)
    r = z_score / sqrt(len(diff))
    delta = cliffs_delta(proposed, baseline)
    return {
        "statistic": float(statistic),
        "p_value": float(p_value),
        "mean_diff": float(effect_size),
        "cohen_d": float(cohen_d),
        "r_effect": float(r),
        "cliffs_delta": float(delta),
    }


def _load_metric_table(
    path: Path, metric: str, proposed_label: str, baseline_label: str
) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    if not {"model", "seed", metric}.issubset(df.columns):
        raise KeyError(f"CSV must contain columns model, seed, and {metric}.")

    prop = df[df["model"].str.lower() == proposed_label.lower()][["seed", metric]].copy()
    base = df[df["model"].str.lower() == baseline_label.lower()][["seed", metric]].copy()

    merged = prop.merge(base, on="seed", suffixes=("_proposed", "_baseline"))
    if merged.empty:
        raise ValueError("No paired samples found for the selected models.")

    return merged[f"{metric}_proposed"].to_numpy(), merged[f"{metric}_baseline"].to_numpy()


def _load_results(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "phase" not in df.columns:
        df["phase"] = "phase1_apple"
    df["phase"] = df["phase"].fillna("phase1_apple")
    df["phase"] = df["phase"].replace(
        {
            "phase1": "phase1_apple",
            "phase2": "phase2_native",
            "ablation": "phase3_ablation",
            "phase3": "phase3_ablation",
        }
    )
    df["profile"] = df.get("profile", "base").fillna("base")
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.sort_values("timestamp")
    df = df.drop_duplicates(
        subset=["phase", "model", "profile", "seed"], keep="last"
    )
    return df


def _metric_series(
    df: pd.DataFrame, phase: str, model: str, profile: str, metric: str
) -> pd.Series:
    subset = df[
        (df["phase"] == phase)
        & (df["model"] == model)
        & (df["profile"] == profile)
    ]
    if subset.empty:
        return pd.Series(dtype=float)
    return subset.groupby("seed")[metric].mean()


def compute_statistics(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    records = []
    for phase in sorted(df["phase"].unique()):
        reference_series = _metric_series(
            df, phase, REFERENCE_MODEL, REFERENCE_PROFILE, metric
        )
        if reference_series.empty:
            continue
        phase_slice = df[df["phase"] == phase]
        for (model, profile), group in phase_slice.groupby(["model", "profile"]):
            series = group.groupby("seed")[metric].mean()
            if series.empty:
                continue
            stats: Dict[str, float] = {
                "phase": phase,
                "model": model,
                "profile": profile,
                "p_value_vs_proposed": float("nan"),
                "effect_size": float("nan"),
                "cliffs_delta": float("nan"),
                "mean_diff": float("nan"),
                "significant": False,
            }
            if model == REFERENCE_MODEL and profile == REFERENCE_PROFILE:
                records.append(stats)
                continue
            aligned_proposed, aligned_baseline = reference_series.align(
                series, join="inner"
            )
            if aligned_proposed.empty:
                continue
            res = wilcoxon_test(
                aligned_proposed.to_numpy(), aligned_baseline.to_numpy()
            )
            stats.update(
                {
                    "p_value_vs_proposed": res["p_value"],
                    "effect_size": res["cohen_d"],
                    "cliffs_delta": res.get("cliffs_delta", float("nan")),
                    "mean_diff": res["mean_diff"],
                    "significant": res["p_value"] < 0.05,
                }
            )
            records.append(stats)
    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run statistical tests on experiment results."
    )
    parser.add_argument(
        "--file",
        default=str(DEFAULT_RESULTS_FILE),
        help="Metrics CSV path",
    )
    parser.add_argument("--metric", default="MAE", help="Metric column to compare")
    parser.add_argument("--proposed", default="Proposed", help="Label for proposed model")
    parser.add_argument(
        "--baseline", default="LS", help="Label for baseline model"
    )
    args = parser.parse_args()

    proposed, baseline = _load_metric_table(
        Path(args.file), args.metric, args.proposed, args.baseline
    )
    result = wilcoxon_test(proposed, baseline)
    print(f"Wilcoxon test ({args.proposed} vs {args.baseline}) on {args.metric}:")
    for key, value in result.items():
        print(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    main()
