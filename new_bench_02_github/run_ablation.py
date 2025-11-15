#!/usr/bin/env python
"""Convenience runner for the proposed-model ablation phase."""
from __future__ import annotations

import sys

from new_bench_02.runner.main import main as runner_main


if __name__ == "__main__":
    sys.argv = [sys.argv[0], "--phase", "phase3_ablation"] + sys.argv[1:]
    runner_main()
