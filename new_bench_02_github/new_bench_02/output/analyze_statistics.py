"""Aggregate statistics for experiment results."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import pandas as pd

from ..paths import package_path
from .summarize import summarize_results
from .statistics import compute_statistics, _load_results

DEFAULT_RESULTS = package_path("3_output", "results", "experiment_metrics.csv")
DEFAULT_OUTPUT = package_path("3_output", "tables", "summary_with_stats.csv")

def generate_statistics(results_path: Path, metric: str, output_path: Path) -> pd.DataFrame:
    df = _load_results(results_path)
    output_dir = output_path.parent
    summary = summarize_results(results_path, output_dir)
    stats_df = compute_statistics(df, metric)
    merged = summary.merge(stats_df, on=["phase", "model", "profile"], how="left")
    merged["rank"] = merged.groupby("phase")[f"{metric}_mean"].rank(method="dense")
    output_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="Augment summary statistics with hypothesis tests.")
    parser.add_argument("--results", default=str(DEFAULT_RESULTS))
    parser.add_argument("--metric", default="MAE")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    merged = generate_statistics(Path(args.results), args.metric, Path(args.output))
    pd.set_option("display.max_columns", None)
    print(merged.to_string(index=False))


if __name__ == "__main__":
    main()
