"""Common model interfaces for the benchmark."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

from ..data.provider import SplitData


@dataclass
class ModelOutput:
    position: np.ndarray
    classification: Optional[np.ndarray] = None
    confidence: Optional[np.ndarray] = None
    extras: Dict[str, Any] = field(default_factory=dict)


class BaseModel(ABC):
    name: str = "Base"
    supports_gpu: bool = False
    requires_fit: bool = True

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}

    @abstractmethod
    def fit(self, train: SplitData, val: Optional[SplitData] = None) -> None:  # noqa: D401
        """Train the model."""

    @abstractmethod
    def predict(self, features: np.ndarray) -> ModelOutput:  # noqa: D401
        """Run inference and return structured outputs."""

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "config": self.config}
