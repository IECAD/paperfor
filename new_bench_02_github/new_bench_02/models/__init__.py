"""Model zoo for the UWB benchmark."""
from __future__ import annotations

from importlib import import_module
from typing import List

from .registry import MODEL_REGISTRY, create_model, register_model

# Import baseline implementations so that they register themselves.
_MODULES: List[str] = [
    ".traditional.ls",
    ".traditional.wls",
    ".traditional.ekf",
    ".traditional.lmkf",
    ".traditional.pf",
    ".ml.svm",
    ".ml.fc_svm",
    ".ml.bo_fdt",
    ".dl.lstm",
    ".dl.cnn_mlp",
    ".dl.fcn_attention",
    ".dl.f_bert",
    ".dl.att_lstm",
    ".hybrid.dnn_ekf",
    ".hybrid.tptp",
    ".proposed.multitask_attention",
    ".proposed.variants",
]

for _module in _MODULES:
    import_module(_module, package=__name__)

__all__ = ["MODEL_REGISTRY", "create_model", "register_model"]
