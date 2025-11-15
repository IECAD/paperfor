"""Export reproducible split metadata."""
from __future__ import annotations

import json
from pathlib import Path

from ..data.provider import DataProvider


def main() -> None:
    provider = DataProvider()
    bundle = provider.prepare_once()
    info = {
        "metadata": bundle.metadata,
        "train_size": int(len(bundle.train.X)),
        "val_size": int(len(bundle.val.X) if bundle.val else 0),
        "test_size": int(len(bundle.test.X)),
    }
    output_path = Path("data/splits.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"Wrote split metadata to {output_path}")


if __name__ == "__main__":
    main()
