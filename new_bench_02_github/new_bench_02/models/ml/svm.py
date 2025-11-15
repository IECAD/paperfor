"""Support Vector Machine hybrid baseline."""
from __future__ import annotations

import numpy as np
from sklearn.svm import SVC, SVR

from ..base import BaseModel, ModelOutput
from ..registry import register_model


@register_model("SVM")
class SVMHybridModel(BaseModel):
    supports_gpu = False
    requires_fit = True

    def __init__(
        self,
        regressor_kernel: str = "rbf",
        classifier_kernel: str = "rbf",
        C: float = 10.0,
        epsilon: float = 0.1,
        gamma: str | float = "scale",
        **kwargs,
    ) -> None:
        super().__init__(kwargs)
        self.reg_x = SVR(kernel=regressor_kernel, C=C, epsilon=epsilon, gamma=gamma)
        self.reg_y = SVR(kernel=regressor_kernel, C=C, epsilon=epsilon, gamma=gamma)
        self.clf = SVC(kernel=classifier_kernel, C=C, gamma=gamma)

    def fit(self, train, val=None) -> None:  # noqa: D401
        if train.X.size == 0:
            raise ValueError("Training data is empty.")
        self.reg_x.fit(train.X, train.y_pos[:, 0])
        self.reg_y.fit(train.X, train.y_pos[:, 1])
        self.clf.fit(train.X, train.y_class)

    def predict(self, features: np.ndarray) -> ModelOutput:
        pos_x = self.reg_x.predict(features)
        pos_y = self.reg_y.predict(features)
        position = np.column_stack([pos_x, pos_y])
        classification = self.clf.predict(features)
        return ModelOutput(position=position, classification=classification)
