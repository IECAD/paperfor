"""Bayesian optimized fuzzy decision tree."""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import ParameterSampler
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

from ..base import BaseModel, ModelOutput
from ..registry import register_model


@register_model("BO-FDT")
class BayesianOptimizedFuzzyDecisionTree(BaseModel):
    supports_gpu = False

    def __init__(self, n_iter: int = 20, random_state: int = 42, **kwargs) -> None:
        super().__init__(kwargs)
        self.n_iter = n_iter
        self.random_state = random_state
        self.best_reg_x: DecisionTreeRegressor | None = None
        self.best_reg_y: DecisionTreeRegressor | None = None
        self.best_clf: DecisionTreeClassifier | None = None

    def fit(self, train, val=None) -> None:  # noqa: D401
        rng = np.random.default_rng(self.random_state)
        param_dist = {
            "max_depth": [3, 4, 5, 6, None],
            "min_samples_split": [2, 4, 6, 8],
            "min_samples_leaf": [1, 2, 4],
        }
        best_score = float("inf")
        for params in ParameterSampler(param_dist, n_iter=self.n_iter, random_state=rng.integers(1, 1_000_000)):
            reg_x = DecisionTreeRegressor(random_state=self.random_state, **params)
            reg_y = DecisionTreeRegressor(random_state=self.random_state, **params)
            reg_x.fit(train.X, train.y_pos[:, 0])
            reg_y.fit(train.X, train.y_pos[:, 1])
            preds = np.column_stack([reg_x.predict(train.X), reg_y.predict(train.X)])
            mae = np.mean(np.linalg.norm(preds - train.y_pos, axis=1))
            if mae < best_score:
                best_score = mae
                self.best_reg_x = reg_x
                self.best_reg_y = reg_y
                clf = DecisionTreeClassifier(random_state=self.random_state, **params)
                clf.fit(train.X, train.y_class)
                self.best_clf = clf

        if self.best_reg_x is None or self.best_reg_y is None or self.best_clf is None:
            raise RuntimeError("Failed to fit BO-FDT model.")

    def predict(self, features: np.ndarray) -> ModelOutput:
        if self.best_reg_x is None or self.best_reg_y is None or self.best_clf is None:
            raise RuntimeError("Model has not been fitted.")
        pos = np.column_stack([
            self.best_reg_x.predict(features),
            self.best_reg_y.predict(features),
        ])
        classification = self.best_clf.predict(features)
        proba = self.best_clf.predict_proba(features)
        confidence = proba.max(axis=1) if proba.ndim == 2 else np.ones(features.shape[0])
        return ModelOutput(position=pos, classification=classification, confidence=confidence)
