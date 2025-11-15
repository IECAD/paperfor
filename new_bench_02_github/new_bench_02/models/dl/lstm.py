"""LSTM-based deep learning baseline."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ..base import BaseModel, ModelOutput
from ..registry import register_model


def _reshape_sequences(X: np.ndarray, seq_len: int, input_dim: int) -> np.ndarray:
    if X.shape[1] != seq_len * input_dim:
        raise ValueError(f"Expected feature dimension {seq_len * input_dim}, got {X.shape[1]}")
    return X.reshape(-1, seq_len, input_dim)


class _LSTMHead(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, num_layers: int, dropout: float, num_classes: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.norm = nn.LayerNorm(hidden_size)
        self.pos_head = nn.Sequential(nn.Linear(hidden_size, hidden_size // 2), nn.ReLU(), nn.Linear(hidden_size // 2, 2))
        self.cls_head = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        outputs, _ = self.lstm(x)
        last = self.norm(outputs[:, -1])
        pos = self.pos_head(last)
        cls = self.cls_head(last)
        return pos, cls


@dataclass
class _TrainingState:
    model: _LSTMHead
    device: torch.device


@register_model("LSTM")
class LSTMSequenceModel(BaseModel):
    supports_gpu = True
    requires_fit = True

    def __init__(
        self,
        seq_len: int = 5,
        input_dim: int = 4,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        batch_size: int = 128,
        epochs: int = 80,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        **kwargs,
    ) -> None:
        super().__init__(kwargs)
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.state: Optional[_TrainingState] = None
        self.num_classes: Optional[int] = None

    def _build_state(self, num_classes: int) -> _TrainingState:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = _LSTMHead(self.input_dim, self.hidden_size, self.num_layers, self.dropout, num_classes)
        model.to(device)
        return _TrainingState(model=model, device=device)

    def fit(self, train, val=None) -> None:  # noqa: D401
        inputs = _reshape_sequences(train.X, self.seq_len, self.input_dim)
        train_dataset = TensorDataset(
            torch.from_numpy(inputs),
            torch.from_numpy(train.y_pos),
            torch.from_numpy(train.y_class),
        )
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        val_loader = None
        if val is not None and val.X.size:
            val_inputs = _reshape_sequences(val.X, self.seq_len, self.input_dim)
            val_dataset = TensorDataset(
                torch.from_numpy(val_inputs),
                torch.from_numpy(val.y_pos),
                torch.from_numpy(val.y_class),
            )
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        self.num_classes = int(train.y_class.max() + 1)
        state = self._build_state(self.num_classes)
        self.state = state

        optimizer = torch.optim.AdamW(state.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs, eta_min=1e-5)
        pos_criterion = nn.SmoothL1Loss()
        cls_criterion = nn.CrossEntropyLoss()

        for epoch in range(self.epochs):
            state.model.train()
            epoch_loss = 0.0
            for batch in train_loader:
                batch_x, batch_pos, batch_cls = [b.to(state.device) for b in batch]
                optimizer.zero_grad()
                pos_pred, cls_pred = state.model(batch_x)
                pos_loss = pos_criterion(pos_pred, batch_pos)
                cls_loss = cls_criterion(cls_pred, batch_cls)
                loss = pos_loss + 0.3 * cls_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(state.model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item() * batch_x.size(0)

            scheduler.step()

            if val_loader:
                state.model.eval()
                with torch.no_grad():
                    _ = sum(batch[0].size(0) for batch in val_loader)
            if epoch % 20 == 0 or epoch == self.epochs - 1:
                _ = epoch_loss / len(train_loader.dataset)

    def predict(self, features: np.ndarray) -> ModelOutput:
        if self.state is None or self.num_classes is None:
            raise RuntimeError("Model has not been trained.")
        seq_data = _reshape_sequences(features, self.seq_len, self.input_dim)
        state = self.state
        state.model.eval()

        with torch.no_grad():
            tensor_x = torch.from_numpy(seq_data).to(state.device)
            pos_pred, cls_pred = state.model(tensor_x)
            position = pos_pred.cpu().numpy()
            classification = cls_pred.argmax(dim=1).cpu().numpy()
            confidence = torch.softmax(cls_pred, dim=1).max(dim=1).values.cpu().numpy()
        return ModelOutput(position=position, classification=classification, confidence=confidence)
