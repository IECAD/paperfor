#!/usr/bin/env bash
set -euo pipefail

python -m new_bench_02.runner.main --phase phase1_apple "$@"
python -m new_bench_02.runner.main --phase phase2_native "$@"
python -m new_bench_02.runner.main --phase phase3_ablation "$@"
python generate_all_plots.py
python analyze_statistics.py
