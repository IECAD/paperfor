# UWB Localization Benchmark

This workspace packages the full pipeline needed to compare the proposed multi-task attention model against classical, machine-learning, deep-learning, and hybrid baselines for UWB-based positioning. The layout implements the performance-comparison plan in `doc/성능비교계획.txt` and supports reproducible, GPU-accelerated experiments.

## Environment

A project-local Conda environment specification is provided under `conda_env/environment.yml`. Create and activate it inside the repository root (no global installs required):

```powershell
conda env create -f conda_env/environment.yml -p ./conda_env/.uwb-benchmark
conda activate ./conda_env/.uwb-benchmark
```

The spec installs PyTorch with CUDA support (when available) alongside the scientific stack used across the benchmark.

## Directory Overview

- `new_bench/data`: raw source spreadsheets (`raw/`) and processed caches (`processed/`) used by the `DataProvider`.
- `new_bench/models`: full model zoo — traditional (LS, WLS, EKF, LMKF, PF), ML (SVM, FC-SVM, BO-FDT), deep models (LSTM, CNN-MLP, FCN-Att, F-BERT, Att-LSTM) and hybrid (DNN-EKF) alongside the proposed multi-task attention architecture.
- `new_bench/output`: metric evaluation, statistics, summarization, and visualization helpers; runtime artefacts live under `new_bench/results`.
- `new_bench/runner`: configurable experiment driver plus the active YAML config.
- `new_bench/logs`: runtime logs (`experiment.log`).

## Usage

1. Place the raw UWB `.xlsx` files under `new_bench/data/raw/` (fallback: `uwb_data/`).
2. Activate the Conda environment.
3. Run a quick smoke test (Phase 1, two models, single seed):
   ```powershell
   python -m new_bench.runner.main --test --models LS,Proposed --seeds 42
   ```
4. Launch the full Phase 1 benchmark defined in `runner/config.yaml` (all models consume the common 20D feature windows):
   ```powershell
   python -m new_bench.runner.main
   ```
5. Switch phases (e.g. ablation) by editing the `phase` key in `new_bench/runner/config.yaml` and rerunning:
   ```powershell
   python -m new_bench.runner.main --config new_bench/runner/config.yaml
   ```
6. Post-process consolidated results:
   ```powershell
   python new_bench/output/summarize.py
   python new_bench/output/statistics.py --baseline LSTM
   python new_bench/output/table_maker.py --format latex
   python new_bench/output/visualizer.py --plots all --metric MAE
   ```

Configuration tweaks (model subsets, hyperparameters, output locations) can be applied via CLI flags or by editing `new_bench/runner/config.yaml`.

### Limiting training data / CAD-only evaluation

- Choose a dataset subset with `--data-group`. The stock config defines:
  - `merged` (default): all XLSX files under `new_bench/data/raw`.
  - `legacy3`: only the original `los`, `nlos_static`, `nlos_dynamic` files.
  - `cad_only`: CAD-specific captures.
- Example - train Proposed on the original three sources:
  ```powershell
  python -m new_bench.runner.main --models Proposed --data-group legacy3 --force-data
  ```
  The best checkpoint is written to `new_bench/models/proposed/checkpoints/ProposedUWBModel/proposed_best.pth`.
- Evaluate that trained model on CAD data without retraining:
  ```powershell
  python -m new_bench.runner.main `
    --models Proposed `
    --data-group cad_only `
    --skip-fit `
    --pretrained Proposed=new_bench/models/proposed/checkpoints/ProposedUWBModel/proposed_best.pth `
    --results new_bench/results/cad_eval.csv
  ```
  (Add `--force-data` the first time to build the CAD-only cache.)

## Notes

- `DataProvider` caches processed tensors, scalers, label encoders, and the derived feature profiles (all rooted in the 20D windowed representation) in `new_bench/data/processed`.
- Every run logs per-seed metrics and efficiency indicators (training time, inference latency, parameter counts) to `new_bench/results/experiment_metrics.csv`, with raw error distributions stored under `new_bench/results/errors/`.
- Model checkpoints for the proposed architecture reside in `new_bench/models/proposed/checkpoints`.

Refer to the `doc/` folder for the detailed comparison plan, model summaries, and supporting literature.
