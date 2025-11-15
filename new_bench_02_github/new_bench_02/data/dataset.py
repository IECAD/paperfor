"""Dataset wrappers for PyTorch models."""
from __future__ import annotations

import torch
from torch.utils.data import Dataset


class UWBDataset(Dataset):
    """Simple tensor dataset wrapping numpy arrays."""

    def __init__(self, features, positions, classes):
        self.features = torch.as_tensor(features, dtype=torch.float32)
        self.positions = torch.as_tensor(positions, dtype=torch.float32)
        self.classes = torch.as_tensor(classes, dtype=torch.long)

    def __len__(self) -> int:  # noqa: D401
        return self.features.shape[0]

    def __getitem__(self, index):  # noqa: D401
        return self.features[index], self.positions[index], self.classes[index]
