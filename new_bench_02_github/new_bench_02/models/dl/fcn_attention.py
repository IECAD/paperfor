"""Fully convolutional attention baseline."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ..base import BaseModel, ModelOutput
from ..registry import register_model


def _reshape_for_attention(X: np.ndarray, seq_len: int = 20) -> np.ndarray:
    seq_len = max(1, min(seq_len, X.shape[1]))
    if X.shape[1] % seq_len == 0:
        input_dim = X.shape[1] // seq_len
        return X.reshape(-1, seq_len, input_dim)
    input_dim = X.shape[1] // seq_len + (1 if X.shape[1] % seq_len else 0)
    padded = np.pad(X, ((0, 0), (0, seq_len * input_dim - X.shape[1])), mode="edge")
    return padded.reshape(-1, seq_len, input_dim)


def _select_heads(embed_dim: int, requested: int) -> int:
    heads = min(requested, embed_dim)
    while heads > 1 and embed_dim % heads != 0:
        heads -= 1
    return max(1, heads)


class _AttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads=n_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_output, _ = self.attn(x, x, x)
        x = self.norm(x + attn_output)
        x = self.norm(x + self.ff(x))
        return x


class _FCNAttention(nn.Module):
    def __init__(self, seq_len: int, input_dim: int, num_classes: int, requested_heads: int = 2) -> None:
        super().__init__()
        d_model = max(input_dim, 8)
        if requested_heads > 1 and d_model % requested_heads != 0:
            d_model += requested_heads - (d_model % requested_heads)
        self.proj = nn.Linear(input_dim, d_model)
        heads = _select_heads(d_model, requested_heads)
        self.block = _AttentionBlock(d_model=d_model, n_heads=heads)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.pos_head = nn.Linear(128, 2)
        self.cls_head = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.proj(x)
        x = self.block(x)
        x = x.transpose(1, 2)
        pooled = self.pool(x).squeeze(-1)
        features = self.head(pooled)
        pos = self.pos_head(features)
        cls = self.cls_head(features)
        return pos, cls


@dataclass
class _State:
    model: _FCNAttention
    device: torch.device


@register_model("FCN-Att")
class FCNAttentionBaseline(BaseModel):
    supports_gpu = True

    def __init__(self, seq_len: int = 20, batch_size: int = 128, epochs: int = 80, lr: float = 1e-3, **kwargs) -> None:
        super().__init__(kwargs)
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.state: Optional[_State] = None
        self.num_classes: Optional[int] = None

    def fit(self, train, val=None) -> None:  # noqa: D401
        tensor_data = _reshape_for_attention(train.X, self.seq_len)
        input_dim = tensor_data.shape[2]
        dataset = TensorDataset(
            torch.from_numpy(tensor_data),
            torch.from_numpy(train.y_pos),
            torch.from_numpy(train.y_class),
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        self.num_classes = int(train.y_class.max() + 1)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = _FCNAttention(self.seq_len, input_dim, self.num_classes).to(device)
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
            raise RuntimeError("Model not fitted.")
        tensor_data = _reshape_for_attention(features, self.seq_len)
        state = self.state
        state.model.eval()
        with torch.no_grad():
            tensor = torch.from_numpy(tensor_data).to(state.device)
            pos, cls = state.model(tensor)
            position = pos.cpu().numpy()
            classification = cls.argmax(dim=1).cpu().numpy()
            confidence = torch.softmax(cls, dim=1).max(dim=1).values.cpu().numpy()
        return ModelOutput(position=position, classification=classification, confidence=confidence)
