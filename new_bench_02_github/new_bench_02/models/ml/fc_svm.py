"""Fuzzy credibility SVM baseline."""
from __future__ import annotations

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC, SVR

from ..base import BaseModel, ModelOutput
from ..registry import register_model


@register_model("FC-SVM")
class FuzzyCredibilitySVM(BaseModel):
    supports_gpu = False

    def __init__(
        self,
        regressor_kernel: str = "rbf",
        classifier_kernel: str = "rbf",
        C: float = 5.0,
        gamma: str | float = "scale",
        **kwargs,
    ) -> None:
        super().__init__(kwargs)
        self.reg_x = SVR(kernel=regressor_kernel, C=C, gamma=gamma)
        self.reg_y = SVR(kernel=regressor_kernel, C=C, gamma=gamma)
        self.clf = SVC(kernel=classifier_kernel, C=C, gamma=gamma, probability=True)
        self.scaler = MinMaxScaler()
        self.feature_means: np.ndarray | None = None

    def fit(self, train, val=None) -> None:  # noqa: D401
        features = np.asarray(train.X, dtype=np.float64)
        self.feature_means = np.nanmean(features, axis=0)
        self.feature_means = np.where(np.isnan(self.feature_means), 0.0, self.feature_means)
        inds = np.where(np.isnan(features))
        features[inds] = self.feature_means[inds[1]]
        scaled = self.scaler.fit_transform(features)
        credibility = np.clip(features.std(axis=1, keepdims=True), 0.0, 1.0)
        augmented = np.hstack([scaled, credibility])
        self.reg_x.fit(augmented, train.y_pos[:, 0])
        self.reg_y.fit(augmented, train.y_pos[:, 1])
        self.clf.fit(augmented, train.y_class)

    def predict(self, features: np.ndarray) -> ModelOutput:
        X = np.asarray(features, dtype=np.float64)
        means = self.feature_means
        if means is None:
            means = np.nanmean(X, axis=0)
            means = np.where(np.isnan(means), 0.0, means)
        inds = np.where(np.isnan(X))
        if inds[0].size:
            X[inds] = means[inds[1]]
        scaled = self.scaler.transform(X)
        credibility = np.clip(X.std(axis=1, keepdims=True), 0.0, 1.0)
        augmented = np.hstack([scaled, credibility])
        pos = np.column_stack([self.reg_x.predict(augmented), self.reg_y.predict(augmented)])
        classification = self.clf.predict(augmented)
        confidence = self.clf.predict_proba(augmented).max(axis=1)
        return ModelOutput(position=pos, classification=classification, confidence=confidence)
