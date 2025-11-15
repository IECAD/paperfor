"""DNN-enhanced EKF baseline."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ..base import BaseModel, ModelOutput
from ..registry import register_model


def _to_pairs(X: np.ndarray) -> np.ndarray:
    return X[:, :2]


class _DNNEstimator(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class _State:
    model: _DNNEstimator
    device: torch.device
    covariance: np.ndarray


@register_model("DNN-EKF")
class DNNEKFModel(BaseModel):
    supports_gpu = True

    def __init__(self, batch_size: int = 128, epochs: int = 80, lr: float = 1e-3, process_noise: float = 0.05, **kwargs) -> None:
        super().__init__(kwargs)
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.process_noise = process_noise
        self.state: Optional[_State] = None

    def fit(self, train, val=None) -> None:  # noqa: D401
        dataset = TensorDataset(
            torch.from_numpy(train.X),
            torch.from_numpy(train.y_pos),
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = _DNNEstimator(train.X.shape[1]).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs, eta_min=1e-5)
        criterion = nn.MSELoss()

        for _ in range(self.epochs):
            model.train()
            for features, target in loader:
                features = features.to(device)
                target = target.to(device)
                optimizer.zero_grad()
                preds = model(features)
                loss = criterion(preds, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            scheduler.step()

        model.eval()
        with torch.no_grad():
            preds = model(torch.from_numpy(train.X).to(device)).cpu().numpy()
        residuals = train.y_pos - preds
        covariance = residuals.T @ residuals / max(1, residuals.shape[0] - 1)
        self.state = _State(model=model, device=device, covariance=covariance)

    def predict(self, features: np.ndarray) -> ModelOutput:
        if self.state is None:
            raise RuntimeError("Model not trained.")
        state = self.state
        state.model.eval()
        with torch.no_grad():
            tensor = torch.from_numpy(features).to(state.device)
            est = state.model(tensor).cpu().numpy()

        if est.shape[0] == 0:
            return ModelOutput(position=est)

        P = state.covariance.copy()
        Q = np.eye(2) * self.process_noise
        positions = np.zeros_like(est)
        current = est[0]
        for idx, measurement in enumerate(est):
            current = current
            P = P + Q
            K = P @ np.linalg.inv(P + state.covariance + np.eye(2) * 1e-3)
            current = current + K @ (measurement - current)
            P = (np.eye(2) - K) @ P
            positions[idx] = current
        return ModelOutput(position=positions)
