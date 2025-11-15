"""Multi-task attention based proposed model."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from ...paths import package_path
from ...data.dataset import UWBDataset
from ...data.provider import SplitData
from ..base import BaseModel, ModelOutput
from ..registry import register_model


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _distance_error(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(((pred - target) ** 2).sum(dim=1) + 1e-8)


class UWBHyperTunedModel(nn.Module):
    def __init__(
        self,
        input_dim: int = 20,
        num_classes: int = 3,
        main_dim: int = 8,
        other_dim: int = 12,
        use_attention: bool = True,
        use_confidence: bool = True,
    ) -> None:
        super().__init__()
        if main_dim + other_dim != input_dim:
            raise ValueError(f"Expected input_dim {main_dim + other_dim}, got {input_dim}")
        self.main_dim = main_dim
        self.other_dim = other_dim
        self.use_attention = use_attention
        self.use_confidence = use_confidence

        self.main_anchor = nn.Sequential(
            nn.Linear(main_dim, 160),
            nn.LayerNorm(160),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(160, 80),
            nn.LayerNorm(80),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(80, 40),
        )
        self.other_anchors = None
        if other_dim > 0:
            self.other_anchors = nn.Sequential(
                nn.Linear(other_dim, 160),
                nn.LayerNorm(160),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(160, 80),
                nn.LayerNorm(80),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(80, 40),
            )
        if use_attention and self.other_anchors is not None:
            self.attention = nn.Sequential(
                nn.Linear(80, 80),
                nn.LayerNorm(80),
                nn.Tanh(),
                nn.Linear(80, 40),
                nn.ReLU(),
                nn.Linear(40, 2),
                nn.Softmax(dim=1),
            )
        else:
            self.attention = None
        self.shared = nn.Sequential(
            nn.Linear(80, 640),
            nn.LayerNorm(640),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(640, 320),
            nn.LayerNorm(320),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(320, 160),
            nn.LayerNorm(160),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.position_head = nn.Sequential(
            nn.Linear(160, 80),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(80, 40),
            nn.ReLU(),
            nn.Linear(40, 20),
            nn.ReLU(),
            nn.Linear(20, 2),
        )
        self.class_head = nn.Sequential(
            nn.Linear(160, 40),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(40, 20),
            nn.ReLU(),
            nn.Linear(20, num_classes),
        )
        self.conf_head = None
        if use_confidence:
            self.conf_head = nn.Sequential(
                nn.Linear(160, 20),
                nn.ReLU(),
                nn.Linear(20, 10),
                nn.ReLU(),
                nn.Linear(10, 1),
                nn.Sigmoid(),
            )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        main = self.main_anchor(x[:, : self.main_dim])
        if self.other_anchors is not None and self.other_dim > 0:
            other = self.other_anchors(x[:, self.main_dim :])
        else:
            other = torch.zeros_like(main)
        combined = torch.cat([main, other], dim=1)

        if self.attention is not None:
            attention = self.attention(combined)
            fused = torch.cat(
                [main * attention[:, 0:1], other * attention[:, 1:2]], dim=1
            )
            att_weights = attention
        else:
            fused = torch.cat([main, other], dim=1)
            att_weights = torch.ones(x.size(0), 2, device=x.device)
        shared = self.shared(fused)
        position = self.position_head(shared)
        classification = self.class_head(shared)
        if self.conf_head is not None:
            confidence = self.conf_head(shared)
        else:
            confidence = torch.ones(position.size(0), 1, device=x.device)
        return position, classification, confidence, att_weights


class UWBMultiTaskLoss(nn.Module):
    def __init__(
        self,
        pos_weight: float = 2.5,
        cls_weight: float = 2.0,
        conf_weight: float = 0.5,
        conf_norm: float = 0.5,
        nlos_boost: float = 0.0,
    ) -> None:
        super().__init__()
        self.pos_weight = pos_weight
        self.cls_weight = cls_weight
        self.conf_weight = conf_weight
        self.conf_norm = conf_norm
        self.nlos_boost = nlos_boost
        self.bce = nn.BCELoss()
        self.class_weights: Optional[torch.Tensor] = None

    def set_class_weights(self, weights: torch.Tensor) -> None:
        self.class_weights = weights

    def forward(
        self,
        pos_pred: torch.Tensor,
        pos_true: torch.Tensor,
        cls_pred: torch.Tensor,
        cls_true: torch.Tensor,
        conf_pred: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        smooth = nn.functional.smooth_l1_loss(
            pos_pred, pos_true, reduction="none"
        ).mean(dim=1)
        mse = nn.functional.mse_loss(
            pos_pred, pos_true, reduction="none"
        ).mean(dim=1)
        pos_errors = 0.6 * smooth + 0.4 * mse
        if self.nlos_boost > 0.0:
            mask = (cls_true != 0).float()
            pos_errors = pos_errors * (1.0 + self.nlos_boost * mask)
        pos_loss = pos_errors.mean()
        if self.cls_weight > 0:
            if self.class_weights is not None:
                cls_loss = nn.functional.cross_entropy(
                    cls_pred, cls_true, weight=self.class_weights
                )
            else:
                cls_loss = nn.functional.cross_entropy(cls_pred, cls_true)
        else:
            cls_loss = torch.zeros(1, device=pos_pred.device)
        if self.conf_weight > 0:
            pos_error = _distance_error(pos_pred, pos_true)
            target_conf = 1.0 - torch.clamp(pos_error / self.conf_norm, 0.0, 1.0)
            target_conf = torch.nan_to_num(target_conf, nan=0.5, posinf=0.0, neginf=1.0).detach()
            target_conf = torch.clamp(target_conf, 0.0, 1.0)
            confidence = torch.nan_to_num(
                conf_pred.squeeze(), nan=0.5, posinf=1.0, neginf=0.0
            )
            confidence = torch.clamp(confidence, 1e-6, 1 - 1e-6)
            conf_loss = self.bce(confidence, target_conf)
        else:
            conf_loss = torch.zeros(1, device=pos_pred.device)
        total = self.pos_weight * pos_loss + self.cls_weight * cls_loss + self.conf_weight * conf_loss
        return total, pos_loss, cls_loss, conf_loss


@dataclass
class _TrainingArtifacts:
    model: UWBHyperTunedModel
    device: torch.device


@register_model("Proposed")
class ProposedUWBModel(BaseModel):
    supports_gpu = True
    requires_fit = True

    def __init__(
        self,
        input_dim: int = 20,
        num_classes: int = 3,
        batch_size: int = 128,
        epochs: int = 150,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        patience: int = 30,
        min_epochs: int = 40,
        cache_dir: Path | str = package_path("2_models", "proposed", "checkpoints"),
        use_attention: bool = True,
        use_multitask: bool = True,
        use_confidence: bool = True,
        cls_weight: float = 2.0,
        conf_weight: float = 0.5,
        pos_weight: float = 2.5,
        nlos_boost: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(kwargs)
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.patience = patience
        self.min_epochs = min_epochs
        self.cache_dir = Path(cache_dir) / self.__class__.__name__
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts: Optional[_TrainingArtifacts] = None
        self.use_attention = use_attention
        self.use_multitask = use_multitask
        self.use_confidence = use_confidence
        self.cls_weight = cls_weight if use_multitask else 0.0
        self.conf_weight = conf_weight if use_confidence else 0.0
        self.pos_weight = pos_weight
        self.nlos_boost = nlos_boost

    def _prepare_loaders(
        self, train: SplitData, val: Optional[SplitData]
    ) -> tuple[DataLoader, Optional[DataLoader]]:
        train_ds = UWBDataset(train.X, train.y_pos, train.y_class)
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        val_loader = None
        if val is not None and val.X.size:
            val_ds = UWBDataset(val.X, val.y_pos, val.y_class)
            val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)
        return train_loader, val_loader

    def _init_state(self) -> _TrainingArtifacts:
        device = _device()
        model = UWBHyperTunedModel(
            input_dim=self.input_dim,
            num_classes=self.num_classes,
            use_attention=self.use_attention,
            use_confidence=self.use_confidence,
        )
        model.to(device)
        return _TrainingArtifacts(model=model, device=device)

    def fit(self, train: SplitData, val: Optional[SplitData] = None) -> None:  # noqa: D401
        self.artifacts = self._init_state()
        train_loader, val_loader = self._prepare_loaders(train, val)
        state = self.artifacts

        criterion = UWBMultiTaskLoss(
            pos_weight=self.pos_weight,
            cls_weight=self.cls_weight,
            conf_weight=self.conf_weight,
            nlos_boost=self.nlos_boost,
        )
        if self.cls_weight > 0:
            class_counts = np.bincount(
                train.y_class.astype(int), minlength=self.num_classes
            )
            class_counts = np.maximum(class_counts, 1)
            inv_freq = class_counts.max() / class_counts
            weights = torch.tensor(inv_freq, dtype=torch.float32, device=state.device)
            criterion.set_class_weights(weights)
        criterion.to(state.device)

        optimizer = optim.AdamW(
            state.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs, eta_min=1e-6
        )

        best_loss = float("inf")
        patience_counter = 0
        best_path = self.cache_dir / f"proposed_best_c{self.num_classes}.pth"
        train_size = max(1, len(train_loader.dataset))

        for epoch in range(self.epochs):
            state.model.train()
            running_loss = 0.0
            for batch in train_loader:
                features, pos, cls = [item.to(state.device) for item in batch]
                optimizer.zero_grad()
                pos_pred, cls_pred, conf_pred, _ = state.model(features)
                loss, _, _, _ = criterion(pos_pred, pos, cls_pred, cls, conf_pred)
                loss.backward()
                nn.utils.clip_grad_norm_(state.model.parameters(), max_norm=1.0)
                optimizer.step()
                running_loss += loss.item() * features.size(0)

            scheduler.step()
            val_loss = None
            if val_loader:
                state.model.eval()
                total_val_loss = 0.0
                total_samples = 0
                with torch.no_grad():
                    for batch in val_loader:
                        features, pos, cls = [item.to(state.device) for item in batch]
                        pos_pred, cls_pred, conf_pred, _ = state.model(features)
                        loss, _, _, _ = criterion(
                            pos_pred, pos, cls_pred, cls, conf_pred
                        )
                        total_val_loss += loss.item() * features.size(0)
                        total_samples += features.size(0)
                val_loss = total_val_loss / max(1, total_samples)

            current = val_loss if val_loss is not None else running_loss / train_size
            if epoch >= self.min_epochs:
                if current < best_loss:
                    best_loss = current
                    patience_counter = 0
                    torch.save(state.model.state_dict(), best_path)
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        break
            else:
                if current < best_loss:
                    best_loss = current
                    torch.save(state.model.state_dict(), best_path)

        if best_path.exists():
            state.model.load_state_dict(torch.load(best_path, map_location=state.device))

    def predict(self, features: np.ndarray) -> ModelOutput:
        if self.artifacts is None:
            raise RuntimeError("Model has not been trained.")
        state = self.artifacts
        state.model.eval()
        with torch.no_grad():
            tensor = torch.from_numpy(features).to(state.device)
            pos_pred, cls_pred, conf_pred, attention = state.model(tensor)
            position = pos_pred.cpu().numpy()
            classification = cls_pred.argmax(dim=1).cpu().numpy()
            confidence = conf_pred.squeeze().cpu().numpy()
            extras = {"attention": attention.cpu().numpy(), "logits": cls_pred.cpu().numpy()}
        return ModelOutput(
            position=position,
            classification=classification,
            confidence=confidence,
            extras=extras,
        )
