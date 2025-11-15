"""Attention-augmented LSTM baseline."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ..base import BaseModel, ModelOutput
from ..registry import register_model


def _to_sequence(X: np.ndarray, seq_len: int = 10) -> np.ndarray:
    input_dim = X.shape[1] // seq_len + (1 if X.shape[1] % seq_len else 0)
    padded = np.pad(X, ((0, 0), (0, seq_len * input_dim - X.shape[1])), mode="edge")
    return padded.reshape(-1, seq_len, input_dim)


class _AttentionLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, num_layers: int, num_classes: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_size, num_layers=num_layers, batch_first=True, dropout=0.2)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads=2, batch_first=True)
        self.norm = nn.LayerNorm(hidden_size)
        self.pos_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 2),
        )
        self.cls_head = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        outputs, _ = self.lstm(x)
        attn_output, _ = self.attn(outputs, outputs, outputs)
        context = self.norm(outputs + attn_output)
        pooled = context.mean(dim=1)
        pos = self.pos_head(pooled)
        cls = self.cls_head(pooled)
        return pos, cls


@dataclass
class _State:
    model: _AttentionLSTM
    device: torch.device


@register_model("Att-LSTM")
class AttentionLSTMBaseline(BaseModel):
    supports_gpu = True

    def __init__(self, seq_len: int = 10, hidden_size: int = 128, num_layers: int = 2, batch_size: int = 128, epochs: int = 80, lr: float = 1e-3, **kwargs) -> None:
        super().__init__(kwargs)
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.state: Optional[_State] = None
        self.num_classes: Optional[int] = None

    def fit(self, train, val=None) -> None:  # noqa: D401
        sequences = _to_sequence(train.X, self.seq_len)
        dataset = TensorDataset(
            torch.from_numpy(sequences),
            torch.from_numpy(train.y_pos),
            torch.from_numpy(train.y_class),
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        self.num_classes = int(train.y_class.max() + 1)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = _AttentionLSTM(sequences.shape[2], self.hidden_size, self.num_layers, self.num_classes).to(device)
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
        sequences = _to_sequence(features, self.seq_len)
        state = self.state
        state.model.eval()
        with torch.no_grad():
            tensor = torch.from_numpy(sequences).to(state.device)
            pos, cls = state.model(tensor)
            position = pos.cpu().numpy()
            classification = cls.argmax(dim=1).cpu().numpy()
            confidence = torch.softmax(cls, dim=1).max(dim=1).values.cpu().numpy()
        return ModelOutput(position=position, classification=classification, confidence=confidence)
