"""Fuzzy BERT-like transformer baseline."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ..base import BaseModel, ModelOutput
from ..registry import register_model


def _prepare_tokens(X: np.ndarray) -> np.ndarray:
    if X.ndim != 2:
        raise ValueError("Expected 2D feature matrix for token preparation.")
    return X.astype(np.float32, copy=False)[:, :, None]


class _TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dim_feedforward: int) -> None:
        super().__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            activation="relu",
        )
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=2)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        cls_tokens = self.cls.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        return self.transformer(x)[:, 0]


class _FBERT(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.project = nn.Linear(1, 32)
        self.block = _TransformerBlock(d_model=32, n_heads=4, dim_feedforward=64)
        self.head = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.pos_head = nn.Linear(64, 2)
        self.cls_head = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.project(x)
        pooled = self.block(x)
        features = self.head(pooled)
        return self.pos_head(features), self.cls_head(features)


@dataclass
class _State:
    model: _FBERT
    device: torch.device


@register_model("F-BERT")
class FBERTBaseline(BaseModel):
    supports_gpu = True

    def __init__(self, batch_size: int = 128, epochs: int = 60, lr: float = 3e-4, **kwargs) -> None:
        super().__init__(kwargs)
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.state: Optional[_State] = None
        self.num_classes: Optional[int] = None

    def fit(self, train, val=None) -> None:  # noqa: D401
        tokens = _prepare_tokens(train.X)
        dataset = TensorDataset(
            torch.from_numpy(tokens),
            torch.from_numpy(train.y_pos),
            torch.from_numpy(train.y_class),
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        self.num_classes = int(train.y_class.max() + 1)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = _FBERT(self.num_classes).to(device)
        self.state = _State(model=model, device=device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs, eta_min=1e-5)
        pos_loss_fn = nn.SmoothL1Loss()
        cls_loss_fn = nn.CrossEntropyLoss()

        for _ in range(self.epochs):
            model.train()
            for batch in loader:
                x, pos, cls = [b.to(device) for b in batch]
                optimizer.zero_grad()
                pos_pred, cls_pred = model(x)
                loss = pos_loss_fn(pos_pred, pos) + 0.3 * cls_loss_fn(cls_pred, cls)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            scheduler.step()

    def predict(self, features: np.ndarray) -> ModelOutput:
        if self.state is None:
            raise RuntimeError("Model not trained.")
        tokens = _prepare_tokens(features)
        state = self.state
        state.model.eval()
        with torch.no_grad():
            tensor = torch.from_numpy(tokens).to(state.device)
            pos, cls = state.model(tensor)
            position = pos.cpu().numpy()
            classification = cls.argmax(dim=1).cpu().numpy()
            confidence = torch.softmax(cls, dim=1).max(dim=1).values.cpu().numpy()
        return ModelOutput(position=position, classification=classification, confidence=confidence)
