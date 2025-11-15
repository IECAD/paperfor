#!/usr/bin/env python
"""Generate summary tables and visualizations for benchmarking results."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "new_bench_02" / "3_output" / "results" / "experiment_metrics.csv"
TABLE_DIR = ROOT / "new_bench_02" / "3_output" / "tables"
FIGURES_DIR = ROOT / "new_bench_02" / "3_output" / "figures"


def run_cmd(cmd):
    print("[generate_all_plots]", " ".join(str(c) for c in cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    py = sys.executable
    if not RESULTS.exists():
        raise FileNotFoundError(f"Results file not found: {RESULTS}")

    df = pd.read_csv(RESULTS)
    phases = sorted(df["phase"].dropna().unique()) if "phase" in df.columns else [None]

    run_cmd([py, "-m", "new_bench_02.output.summarize", "--file", str(RESULTS)])
    run_cmd([py, "-m", "new_bench_02.output.analyze_statistics", "--results", str(RESULTS)])

    metrics = ["MAE", "RMSE"]
    for phase in phases:
        phase_arg = ["--phase", phase] if phase else []
        for metric in metrics:
            run_cmd(
                [
                    py,
                    "-m",
                    "new_bench_02.output.visualizer",
                    "--file",
                    str(RESULTS),
                    "--metric",
                    metric,
                    "--plots",
                    "all",
                    "--out",
                    str(FIGURES_DIR),
                    *phase_arg,
                ]
            )


if __name__ == "__main__":
    main()
