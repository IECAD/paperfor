"""Proposed model variants for ablation studies."""
from __future__ import annotations

from ..registry import register_model
from .multitask_attention import ProposedUWBModel


@register_model("Proposed_V1")
class ProposedV1Model(ProposedUWBModel):
    """Original hyper-parameter profile."""

    def __init__(self, **kwargs) -> None:
        defaults = {
            "batch_size": 128,
            "epochs": 150,
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "cls_weight": 2.0,
            "conf_weight": 0.5,
            "pos_weight": 2.5,
            "nlos_boost": 0.0,
        }
        for key, value in defaults.items():
            kwargs.setdefault(key, value)
        super().__init__(**kwargs)


@register_model("Proposed_V2")
class ProposedV2Model(ProposedUWBModel):
    """Doc-inspired variant with boosted NLOS weighting and longer training."""

    def __init__(self, **kwargs) -> None:
        defaults = {
            "batch_size": 96,
            "epochs": 200,
            "lr": 8e-4,
            "weight_decay": 2e-4,
            "patience": 45,
            "min_epochs": 60,
            "cls_weight": 2.5,
            "conf_weight": 0.7,
            "pos_weight": 3.0,
            "nlos_boost": 0.5,
        }
        for key, value in defaults.items():
            kwargs.setdefault(key, value)
        super().__init__(**kwargs)


@register_model("Proposed_NoAttention")
class ProposedNoAttentionModel(ProposedUWBModel):
    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("use_attention", False)
        super().__init__(**kwargs)


@register_model("Proposed_NoMultitask")
class ProposedNoMultitaskModel(ProposedUWBModel):
    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("use_multitask", False)
        kwargs.setdefault("cls_weight", 0.0)
        super().__init__(**kwargs)


@register_model("Proposed_NoConfidence")
class ProposedNoConfidenceModel(ProposedUWBModel):
    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("use_confidence", False)
        kwargs.setdefault("conf_weight", 0.0)
        super().__init__(**kwargs)


@register_model("Proposed_NoSNR")
class ProposedNoSNRModel(ProposedUWBModel):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)


@register_model("Proposed_8D")
class ProposedEightDimModel(ProposedUWBModel):
    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("input_dim", 20)
        super().__init__(**kwargs)
