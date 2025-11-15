"""Model factory and registry."""
from __future__ import annotations

from typing import Callable, Dict, Type

from .base import BaseModel

MODEL_REGISTRY: Dict[str, Type[BaseModel]] = {}


def register_model(name: str) -> Callable[[Type[BaseModel]], Type[BaseModel]]:
    def decorator(cls: Type[BaseModel]) -> Type[BaseModel]:
        MODEL_REGISTRY[name.lower()] = cls
        cls.name = name
        return cls

    return decorator


def create_model(name: str, **kwargs) -> BaseModel:
    key = name.lower()
    if key not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Registered: {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[key](**kwargs)
