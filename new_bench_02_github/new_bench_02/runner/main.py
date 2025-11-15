"""Main execution entrypoint for the UWB benchmark."""
from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import yaml

from ..data.provider import DataProvider, DataBundle, SplitData
from ..models.base import BaseModel
from ..models.registry import create_model
from ..output.evaluator import Evaluator
from ..paths import package_path


LEGACY_CONFIG_PATH = Path("uwb_benchmark/4_runner/config.yaml")
DEFAULT_CONFIG_PATH = Path(__file__).resolve().with_name("config.yaml")
if not DEFAULT_CONFIG_PATH.exists():
    DEFAULT_CONFIG_PATH = LEGACY_CONFIG_PATH
DEFAULT_RESULTS_FILE = package_path("3_output", "results", "experiment_metrics.csv")
DEFAULT_LOG_DIR = package_path("logs")
PACKAGE_NAME = __package__.split(".")[0] if __package__ else "new_bench_02"


RESULT_COLUMNS = [
    "timestamp",
    "phase",
    "model",
    "profile",
    "seed",
    "MAE",
    "RMSE",
    "CEP50",
    "CEP95",
    "Accuracy",
    "NLOSAccuracy",
    "F1Macro",
    "ConfidenceMean",
    "TrainTimeSec",
    "InferTimeMS",
    "ParamK",
    "ModelSizeMB",
    "FLOPsM",
]


def _setup_logging(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "experiment.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler()],
    )


def _set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run UWB localization benchmark")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to YAML config")
    parser.add_argument("--models", help="Comma-separated subset of models to run")
    parser.add_argument("--seeds", help="Comma-separated list of seeds to use")
    parser.add_argument("--phase", help="Phase key to execute")
    parser.add_argument("--test", action="store_true", help="Run quick smoke test")
    parser.add_argument("--force-data", action="store_true", help="Force dataset regeneration")
    parser.add_argument("--results", help="Override results CSV path")
    return parser.parse_args()


def _load_config(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def _select_models(config: Dict, override: Optional[str], phase_cfg: Dict) -> List[str]:
    if override:
        candidates = override.split(",")
    else:
        candidates = phase_cfg.get("models") or config.get("models", [])
    return [m.strip() for m in candidates if m and m.strip()]


def _select_seeds(config: Dict, override: Optional[str], is_test: bool) -> List[int]:
    if override:
        return [int(s.strip()) for s in override.split(",") if s.strip()]
    seeds = config.get("seeds", [42])
    return seeds[:1] if is_test else seeds


def _resolve_model_config(config: Dict, name: str) -> Dict:
    model_cfg = config.get("model_configs", {})
    return model_cfg.get(name, {})


def _resolve_profiles(
    config: Dict, phase: str, phase_cfg: Dict
) -> Tuple[str, Dict[str, str]]:
    default_profile = "base"
    overrides: Dict[str, str] = {}

    if phase_cfg:
        default_profile = phase_cfg.get("default_profile", default_profile)
        for model_name, profile in phase_cfg.get("model_inputs", {}).items():
            if model_name == "default":
                default_profile = _normalize_profile(profile, default_profile)
            else:
                overrides[model_name] = _normalize_profile(profile, default_profile)

    model_inputs = config.get("model_inputs", {})
    if isinstance(model_inputs, dict):
        default_profile = model_inputs.get("default", default_profile)
        for model_name, profile in model_inputs.items():
            if model_name == "default":
                continue
            overrides.setdefault(model_name, _normalize_profile(profile, default_profile))

    return default_profile, overrides


def _normalize_profile(value, fallback: str) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return value.get("profile") or value.get("name") or fallback
    return fallback


def _resolve_profile_for_model(model_name: str, default_profile: str, overrides: Dict[str, str]) -> str:
    return overrides.get(model_name, default_profile)


def _prepare_results_file(path: Path) -> None:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(RESULT_COLUMNS)


def _save_run(path: Path, row: Dict[str, float]) -> None:
    with open(path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=RESULT_COLUMNS)
        writer.writerow(row)


def _store_errors(directory: Path, model: str, seed: int, errors: np.ndarray) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / f"{model}_seed{seed}_errors.npy", errors)


def _extract_features(bundle: DataBundle, split: str, profile: str) -> np.ndarray:
    split_store = bundle.feature_store.get(split, {})
    if profile == "base":
        target = getattr(bundle, split)
        if target is None:
            return np.zeros((0, 0), dtype=np.float32)
        return target.X
    if profile in split_store:
        return split_store[profile]
    logging.warning("Profile '%s' not found for split '%s'; falling back to base features.", profile, split)
    target = getattr(bundle, split)
    return target.X if target is not None else np.zeros((0, 0), dtype=np.float32)


def _with_features(original: SplitData, features: np.ndarray) -> SplitData:
    meta = {}
    if hasattr(original, "meta") and original.meta:
        meta = {k: np.copy(v) for k, v in original.meta.items()}
    return SplitData(features.astype(np.float32), original.y_pos, original.y_class, meta)


def _count_parameters(model: BaseModel) -> Optional[int]:
    candidates = [model]
    for attr in ("state", "artifacts"):
        candidate = getattr(model, attr, None)
        if candidate is not None:
            candidates.append(candidate)
    for candidate in candidates:
        if isinstance(candidate, torch.nn.Module):
            return sum(p.numel() for p in candidate.parameters())
        if hasattr(candidate, "model") and isinstance(candidate.model, torch.nn.Module):
            return sum(p.numel() for p in candidate.model.parameters())
    return None


def run_model(
    name: str,
    bundle: DataBundle,
    seed: int,
    model_config: Dict,
    evaluator: Evaluator,
    error_dir: Path,
    phase: str,
    default_profile: str,
    profile_overrides: Dict[str, str],
) -> Dict[str, float]:
    logging.info("Running model %s (seed %s)", name, seed)
    _set_global_seed(seed)
    profile = _resolve_profile_for_model(name, default_profile, profile_overrides)

    train_features = _extract_features(bundle, "train", profile)
    train_split = _with_features(bundle.train, train_features)

    if bundle.val is not None:
        val_features = _extract_features(bundle, "val", profile)
        val_split = _with_features(bundle.val, val_features)
    else:
        val_split = None

    test_features = _extract_features(bundle, "test", profile)
    test_split = _with_features(bundle.test, test_features)

    cfg = dict(model_config or {})
    if name.lower().startswith("proposed"):
        cfg.setdefault("input_dim", int(train_features.shape[1]))
        num_classes = int(train_split.y_class.max() + 1) if train_split.y_class.size else 1
        cfg.setdefault("num_classes", num_classes)

    model: BaseModel = create_model(name, **cfg)
    if hasattr(model, "configure_from_bundle"):
        model.configure_from_bundle(bundle)

    train_time = 0.0
    if model.requires_fit:
        start = time.perf_counter()
        model.fit(train_split, val_split)
        train_time = time.perf_counter() - start

    if hasattr(model, "set_inference_split"):
        model.set_inference_split(test_split)

    start = time.perf_counter()
    output = model.predict(test_features)
    infer_time = time.perf_counter() - start
    per_sample_ms = (infer_time / max(1, len(test_features))) * 1000.0

    result = evaluator.evaluate(output, test_split.y_pos, test_split.y_class)
    _store_errors(error_dir, name, seed, result.errors_cm)

    metrics = dict(result.metrics)
    metrics.update(
        {
            "TrainTimeSec": train_time,
            "InferTimeMS": per_sample_ms,
            "ParamK": float("nan"),
            "ModelSizeMB": float("nan"),
            "FLOPsM": float("nan"),
            "Profile": profile,
            "Phase": phase,
        }
    )

    param_count = _count_parameters(model)
    if param_count is not None:
        metrics["ParamK"] = param_count / 1_000.0
        metrics["ModelSizeMB"] = (param_count * 4) / (1024.0 * 1024.0)

    logging.info("%s metrics (%s): %s", name, profile, json.dumps(metrics, ensure_ascii=False))
    return metrics

def _run_post_processing(results_path: Path, metrics: List[str], phase: str) -> None:
    unique_metrics = list(dict.fromkeys(metrics)) or ['MAE']
    output_module = f"{PACKAGE_NAME}.output"
    commands = [
        [sys.executable, "-m", f"{output_module}.summarize", "--file", str(results_path)],
        [sys.executable, "-m", f"{output_module}.analyze_statistics", "--results", str(results_path), "--metric", unique_metrics[0]],
    ]
    for metric in unique_metrics:
        commands.append([
            sys.executable,
            "-m",
            f"{output_module}.visualizer",
            "--file",
            str(results_path),
            "--metric",
            metric,
            "--phase",
            phase,
            "--plots",
            "all",
        ])
    for cmd in commands:
        try:
            subprocess.run(cmd, check=True)
        except FileNotFoundError:
            logging.warning("Post-processing skipped (missing command): %s", cmd)
        except subprocess.CalledProcessError as exc:
            logging.warning("Post-processing command failed (%s): %s", exc.returncode, cmd)



def main() -> None:
    args = _parse_args()
    config = _load_config(Path(args.config))

    phase = args.phase or config.get("phase", "phase1_apple")
    phase_cfg = config.get("phases", {}).get(phase, {})
    if not phase_cfg:
        logging.warning("Phase '%s' not found in config; using defaults.", phase)
        phase_cfg = {}

    default_profile, profile_overrides = _resolve_profiles(config, phase, phase_cfg)

    results_path = Path(args.results) if args.results else Path(
        config.get("results_file", str(DEFAULT_RESULTS_FILE))
    )
    log_dir = Path(config.get("log_dir", str(DEFAULT_LOG_DIR)))
    _setup_logging(log_dir)

    models = _select_models(config, args.models, phase_cfg)
    seeds = _select_seeds(config, args.seeds, args.test)

    logging.info("Selected models: %s", models)
    logging.info("Selected seeds: %s", seeds)
    logging.info("Phase: %s | default profile: %s | overrides: %s", phase, default_profile, profile_overrides)

    provider = DataProvider()
    bundle = provider.prepare_once(force=args.force_data)
    bundle.metadata["phase"] = phase
    logging.info("Dataset prepared: %s", bundle.metadata)

    evaluator = Evaluator()
    _prepare_results_file(results_path)
    error_dir = results_path.parent / "errors"

    for seed in seeds:
        for model_name in models:
            model_cfg = _resolve_model_config(config, model_name)
            metrics = run_model(
                model_name,
                bundle,
                seed,
                model_cfg,
                evaluator,
                error_dir,
                phase,
                default_profile,
                profile_overrides,
            )
            row = {key: float("nan") for key in RESULT_COLUMNS}
            row.update(
                {
                    "timestamp": datetime.utcnow().isoformat(),
                    "phase": metrics.get("Phase", phase),
                    "model": model_name,
                    "profile": metrics.get("Profile", default_profile),
                    "seed": seed,
                    "MAE": metrics.get("MAE", float("nan")),
                    "RMSE": metrics.get("RMSE", float("nan")),
                    "CEP50": metrics.get("CEP50", float("nan")),
                    "CEP95": metrics.get("CEP95", float("nan")),
                    "Accuracy": metrics.get("Accuracy", float("nan")),
                    "NLOSAccuracy": metrics.get("NLOSAccuracy", float("nan")),
                    "F1Macro": metrics.get("F1Macro", float("nan")),
                    "ConfidenceMean": metrics.get("ConfidenceMean", float("nan")),
                    "TrainTimeSec": metrics.get("TrainTimeSec", float("nan")),
                    "InferTimeMS": metrics.get("InferTimeMS", float("nan")),
                    "ParamK": metrics.get("ParamK", float("nan")),
                    "ModelSizeMB": metrics.get("ModelSizeMB", float("nan")),
                    "FLOPsM": metrics.get("FLOPsM", float("nan")),
                }
            )
            _save_run(results_path, row)

    if not args.test:
        metrics_cfg = config.get('metrics', ['MAE'])
        _run_post_processing(results_path, metrics_cfg, phase)


if __name__ == "__main__":
    main()

