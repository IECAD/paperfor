"""Visualization helpers for benchmark results."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..paths import package_path

matplotlib.use("Agg")


LOWER_BETTER = {"MAE", "RMSE", "CEP50", "CEP95", "TrainTimeSec", "InferTimeMS"}
DEFAULT_RESULTS = package_path("3_output", "results", "experiment_metrics.csv")
DEFAULT_FIGURE_DIR = package_path("3_output", "figures")


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _filter_phase(df: pd.DataFrame, phase: str | None) -> pd.DataFrame:
    if phase and "phase" in df.columns:
        return df[df["phase"] == phase] if phase in set(df["phase"]) else df
    return df


def plot_box(df: pd.DataFrame, metric: str, output: Path) -> None:
    _ensure_dir(output.parent)
    plt.figure(figsize=(8, 5))
    df.boxplot(column=metric, by="model")
    plt.title(f"{metric} distribution")
    plt.suptitle("")
    plt.xlabel("Model")
    plt.ylabel(metric)
    plt.tight_layout()
    plt.savefig(output)
    plt.close()


def plot_seed_trend(df: pd.DataFrame, metric: str, output: Path) -> None:
    _ensure_dir(output.parent)
    plt.figure(figsize=(8, 5))
    for model, group in df.groupby("model"):
        group = group.sort_values("seed")
        plt.plot(group["seed"], group[metric], marker="o", label=model)
    plt.title(f"{metric} per seed")
    plt.xlabel("Seed")
    plt.ylabel(metric)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output)
    plt.close()


def plot_critical_difference(df: pd.DataFrame, metric: str, output: Path) -> None:
    _ensure_dir(output.parent)
    agg = df.groupby("model")[metric].mean()
    agg = agg.dropna()
    if agg.empty:
        return
    agg = agg.sort_values()
    ranks = agg.rank(method="dense")
    plt.figure(figsize=(8, 2 + len(agg) * 0.1))
    plt.scatter(agg, [1] * len(agg), c=ranks, cmap="viridis", s=80)
    for x, model in zip(agg, agg.index):
        plt.text(x, 1.05, f"{model}\n(rank {int(ranks[model])})", ha="center", va="bottom", fontsize=8)
    plt.yticks([])
    plt.xlabel(metric)
    plt.title(f"Critical-difference style view ({metric})")
    plt.tight_layout()
    plt.savefig(output)
    plt.close()


def _normalize_for_radar(values: pd.DataFrame) -> pd.DataFrame:
    normed = values.copy().astype(float)
    for col in normed.columns:
        col_vals = normed[col]
        denom = col_vals.max() - col_vals.min()
        if denom == 0:
            normed[col] = 0.5
            continue
        if col in LOWER_BETTER:
            normed[col] = (col_vals.max() - col_vals) / denom
        else:
            normed[col] = (col_vals - col_vals.min()) / denom
    return normed


def plot_radar(df: pd.DataFrame, metrics: List[str], output: Path) -> None:
    _ensure_dir(output.parent)
    agg = df.groupby("model")[metrics].mean()
    if agg.empty:
        return
    top = agg.sort_values(metrics[0]).head(5)
    normed = _normalize_for_radar(top)
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False)
    angles = np.concatenate([angles, [angles[0]]])
    plt.figure(figsize=(6, 6))
    ax = plt.subplot(111, polar=True)
    for model, row in normed.iterrows():
        values = np.concatenate([row.values, [row.values[0]]])
        ax.plot(angles, values, label=model)
        ax.fill(angles, values, alpha=0.1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_ylim(0, 1.0)
    ax.set_title("Radar chart (normalized)")
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.savefig(output)
    plt.close()


def plot_efficiency(df: pd.DataFrame, output: Path) -> None:
    _ensure_dir(output.parent)
    if not {"TrainTimeSec", "Accuracy"}.issubset(df.columns):
        return
    agg = df.groupby("model").agg({"TrainTimeSec": "mean", "Accuracy": "mean"})
    plt.figure(figsize=(7, 5))
    plt.scatter(agg["TrainTimeSec"], agg["Accuracy"], s=60)
    for model, row in agg.iterrows():
        plt.text(row["TrainTimeSec"], row["Accuracy"], model, fontsize=8, ha="left", va="bottom")
    plt.xlabel("Training time (s)")
    plt.ylabel("Accuracy (%)")
    plt.title("Efficiency vs Accuracy")
    plt.tight_layout()
    plt.savefig(output)
    plt.close()


def plot_environment_heatmap(df: pd.DataFrame, output: Path) -> None:
    _ensure_dir(output.parent)
    metrics = [col for col in ["Accuracy", "NLOSAccuracy", "F1Macro"] if col in df.columns]
    if not metrics:
        return
    agg = df.groupby("model")[metrics].mean()
    plt.figure(figsize=(max(6, len(metrics) + 2), 0.4 * len(agg) + 2))
    plt.imshow(agg, aspect="auto", cmap="viridis")
    plt.colorbar(label="Score")
    plt.xticks(range(len(metrics)), metrics, rotation=45, ha="right")
    plt.yticks(range(len(agg.index)), agg.index)
    plt.title("Environment-style performance heatmap")
    plt.tight_layout()
    plt.savefig(output)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot benchmark metrics.")
    parser.add_argument(
        "--file",
        default=str(DEFAULT_RESULTS),
        help="Metrics CSV path",
    )
    parser.add_argument("--metric", default="MAE")
    parser.add_argument(
        "--plots",
        nargs="*",
        choices=["box", "trend", "critical", "radar", "efficiency", "heatmap", "all"],
        default=["all"],
    )
    parser.add_argument("--phase", default="phase1_apple")
    parser.add_argument("--out", default=str(DEFAULT_FIGURE_DIR))
    args = parser.parse_args()

    df = pd.read_csv(args.file)
    if "phase" in df.columns:
        df["phase"] = df["phase"].fillna(args.phase)
    phase_df = _filter_phase(df, args.phase)

    out_dir = Path(args.out)
    metrics_dir = out_dir / args.metric / args.phase
    plots: Iterable[str] = set(args.plots)

    if "all" in plots:
        plots = {"box", "trend", "critical", "radar", "efficiency", "heatmap"}

    if "box" in plots:
        plot_box(phase_df, args.metric, metrics_dir / f"{args.metric}_box.png")
    if "trend" in plots and "seed" in phase_df.columns:
        plot_seed_trend(phase_df, args.metric, metrics_dir / f"{args.metric}_trend.png")
    if "critical" in plots:
        plot_critical_difference(phase_df, args.metric, metrics_dir / f"{args.metric}_critical.png")
    if "radar" in plots:
        radar_metrics = [m for m in ["MAE", "RMSE", "CEP95", "Accuracy", "TrainTimeSec"] if m in phase_df.columns]
        if radar_metrics:
            plot_radar(phase_df, radar_metrics, metrics_dir / "radar.png")
    if "efficiency" in plots:
        plot_efficiency(phase_df, metrics_dir / "efficiency_tradeoff.png")
    if "heatmap" in plots:
        plot_environment_heatmap(phase_df, metrics_dir / "environment_heatmap.png")


if __name__ == "__main__":
    main()
