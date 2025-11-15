"""Produce formatted tables from summary metrics."""
from __future__ import annotations

import argparse
from pathlib import Path

from ..paths import package_path
from .summarize import summarize_results

DEFAULT_RESULTS = package_path("3_output", "results", "experiment_metrics.csv")
DEFAULT_TABLE_DIR = package_path("3_output", "tables")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate result tables in various formats.")
    parser.add_argument("--file", default=str(DEFAULT_RESULTS))
    parser.add_argument("--format", choices=["csv", "latex", "markdown"], default="latex")
    parser.add_argument("--out", default=str(DEFAULT_TABLE_DIR))
    args = parser.parse_args()

    output_dir = Path(args.out)
    summary = summarize_results(Path(args.file), output_dir)

    if args.format == "csv":
        print(summary.to_csv(index=False))
    elif args.format == "latex":
        print(summary.to_latex(index=False, float_format=lambda x: f"{x:.2f}"))
    else:
        print(summary.to_markdown(index=False, floatfmt=".2f"))


if __name__ == "__main__":
    main()
