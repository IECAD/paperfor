"""Summarize experiment metrics."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ..paths import package_path


GROUP_KEYS = ["phase", "model", "profile"]
DROP_COLS = {"seed", "timestamp", "split"}
DEFAULT_RESULTS = package_path("3_output", "results", "experiment_metrics.csv")
DEFAULT_TABLE_DIR = package_path("3_output", "tables")


def summarize_results(path: Path, output_dir: Path | None = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    available_groups = [c for c in GROUP_KEYS if c in df.columns]
    if not available_groups:
        available_groups = ["model"]

    numeric_cols = [c for c in df.select_dtypes(include="number").columns if c not in DROP_COLS]
    summary = df.groupby(available_groups)[numeric_cols].agg(["mean", "std"])
    summary.columns = [f"{col}_{stat}" for col, stat in summary.columns]
    summary = summary.reset_index()

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / "summary.csv"
        summary.to_csv(csv_path, index=False)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize benchmark results.")
    parser.add_argument("--file", default=str(DEFAULT_RESULTS))
    parser.add_argument("--out", default=str(DEFAULT_TABLE_DIR))
    args = parser.parse_args()

    summary = summarize_results(Path(args.file), Path(args.out))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
