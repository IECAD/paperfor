"""Metric evaluation for benchmark runs."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from sklearn.metrics import f1_score

from ..models.base import ModelOutput


@dataclass
class EvaluationResult:
    metrics: Dict[str, float]
    errors_cm: np.ndarray


class Evaluator:
    def __init__(self, distance_unit: float = 100.0) -> None:
        self.distance_unit = distance_unit

    def evaluate(
        self,
        output: ModelOutput,
        true_position: np.ndarray,
        true_class: Optional[np.ndarray] = None,
    ) -> EvaluationResult:
        if output.position.shape != true_position.shape:
            raise ValueError("Prediction and ground-truth shapes do not match.")

        errors = np.linalg.norm(output.position - true_position, axis=1)
        errors_cm = errors * self.distance_unit

        metrics: Dict[str, float] = {
            "MAE": float(np.mean(errors_cm)),
            "RMSE": float(np.sqrt(np.mean(errors_cm ** 2))),
            "CEP50": float(np.percentile(errors_cm, 50)),
            "CEP95": float(np.percentile(errors_cm, 95)),
        }

        if true_class is not None and output.classification is not None:
            accuracy = float(np.mean(output.classification == true_class)) if len(true_class) else 0.0
            metrics["Accuracy"] = accuracy * 100.0
            metrics["NLOSAccuracy"] = metrics["Accuracy"]
            if len(true_class) and len(output.classification):
                metrics["F1Macro"] = float(f1_score(true_class, output.classification, average="macro", zero_division=0))
            else:
                metrics["F1Macro"] = 0.0

        if output.confidence is not None:
            metrics["ConfidenceMean"] = float(np.mean(output.confidence))

        return EvaluationResult(metrics=metrics, errors_cm=errors_cm)
