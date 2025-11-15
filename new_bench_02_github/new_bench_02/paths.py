"""Utilities for resolving package-local paths."""
from __future__ import annotations

from pathlib import Path

# Absolute path to the root of the new_bench_02 package.
PACKAGE_ROOT = Path(__file__).resolve().parent
# Legacy project root retained for backward compatibility checks.
LEGACY_ROOT = Path("uwb_benchmark")


def package_path(*parts: str) -> Path:
    """Return a path inside the package root."""
    return PACKAGE_ROOT.joinpath(*parts)


def legacy_path(*parts: str) -> Path:
    """Return a path inside the legacy uwb_benchmark tree."""
    return LEGACY_ROOT.joinpath(*parts)


def ensure_dir(path: Path) -> Path:
    """Create a directory (and parents) if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)
    return path
