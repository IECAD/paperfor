"""CNN-MLP hybrid baseline."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ..base import BaseModel, ModelOutput
from ..registry import register_model


def _pad_features(X: np.ndarray, target_len: int = 128) -> np.ndarray:
    if X.shape[1] >= target_len:
        return X[:, :target_len]
    pad_width = target_len - X.shape[1]
    return np.pad(X, ((0, 0), (0, pad_width)), mode="edge")


class _CNNMLP(nn.Module):
    def __init__(self, input_len: int, num_classes: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        conv_out = (input_len // 4) * 32
        self.mlp = nn.Sequential(
            nn.Linear(conv_out, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.pos_head = nn.Linear(64, 2)
        self.cls_head = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x.unsqueeze(1)
        features = self.conv(x)
        features = features.reshape(features.size(0), -1)
        features = self.mlp(features)
        pos = self.pos_head(features)
        cls = self.cls_head(features)
        return pos, cls


@dataclass
class _State:
    model: _CNNMLP
    device: torch.device


@register_model("CNN-MLP")
class CNNMLPBaseline(BaseModel):
    supports_gpu = True

    def __init__(self, input_len: int = 128, batch_size: int = 128, epochs: int = 80, lr: float = 1e-3, **kwargs) -> None:
        super().__init__(kwargs)
        self.input_len = input_len
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.state: Optional[_State] = None
        self.num_classes: Optional[int] = None

    def _build(self, num_classes: int) -> _State:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = _CNNMLP(self.input_len, num_classes).to(device)
        return _State(model=model, device=device)

    def fit(self, train, val=None) -> None:  # noqa: D401
        inputs = _pad_features(train.X, self.input_len)
        dataset = TensorDataset(
            torch.from_numpy(inputs),
            torch.from_numpy(train.y_pos),
            torch.from_numpy(train.y_class),
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        self.num_classes = int(train.y_class.max() + 1)
        state = self._build(self.num_classes)
        self.state = state

        optimizer = torch.optim.AdamW(state.model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs, eta_min=1e-5)
        pos_loss_fn = nn.SmoothL1Loss()
        cls_loss_fn = nn.CrossEntropyLoss()

        for _ in range(self.epochs):
            state.model.train()
            for batch in loader:
                x, pos, cls = [t.to(state.device) for t in batch]
                optimizer.zero_grad()
                pos_pred, cls_pred = state.model(x)
                loss = pos_loss_fn(pos_pred, pos) + 0.3 * cls_loss_fn(cls_pred, cls)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(state.model.parameters(), max_norm=1.0)
                optimizer.step()
            scheduler.step()

    def predict(self, features: np.ndarray) -> ModelOutput:
        if self.state is None:
            raise RuntimeError("Model not trained.")
        state = self.state
        state.model.eval()
        padded = _pad_features(features, self.input_len)
        with torch.no_grad():
            tensor = torch.from_numpy(padded).to(state.device)
            pos, cls = state.model(tensor)
            position = pos.cpu().numpy()
            classification = cls.argmax(dim=1).cpu().numpy()
            confidence = torch.softmax(cls, dim=1).max(dim=1).values.cpu().numpy()
        return ModelOutput(position=position, classification=classification, confidence=confidence)
