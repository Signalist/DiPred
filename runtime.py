"""Load small train-only normalization metadata without binary artifacts."""

import json
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent


def load_normalization(device="cpu"):
    values = json.loads((ROOT / "metadata/normalization.json").read_text())
    return {key: torch.tensor(values[key], dtype=torch.float32, device=device).reshape(1, 2, 1, 1, 1)
            for key in ("mean", "std")}
