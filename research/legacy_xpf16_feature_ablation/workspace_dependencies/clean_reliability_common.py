"""Shared paths, data access, and scales for the isolated direct20 UQ study."""

import hashlib
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from joint20_data_head_diffusion import Joint20DataHeadDiffusion
from local_grid_representation import LocalGridRepresentation
from retrain_clean_direct20 import digest, dump


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "runs/clean_direct20_20261107"
OUT = ROOT / "runs/clean_direct20_reliability_20261108"
DATA = ROOT.parent / "OLD/REALPDE/realpdebench-competition/data/train_real"
KINDS = ("ae", "vae", "rae")
MAPS = ("constant", "history_std", "forecast_departure", "forecast_gradient",
        "train_template", "learned_risk", "learned_risk_spread", "diffusion_std8",
        "diffusion_std32", "diffusion_quantile32", "feedback_error")


def configure():
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False


def manifest():
    return json.loads((SOURCE / "manifest.json").read_text())


def starts(split):
    # Training/calibration end at 200; unseen-group evaluation continues to 600.
    return range(0, 161 if split in ("train", "calibration") else 561, 20)


def field_path(role, split, start):
    return OUT / "fields" / f"{role}_{split}_{start}.pt"


def issued_path(kind, split, start):
    return OUT / "issued" / f"{kind}_{split}_{start}.pt"


def load_field(role, split, start, device="cuda"):
    return torch.load(field_path(role, split, start), map_location=device, weights_only=False)


def load_issued(kind, split, start, device="cuda"):
    return torch.load(issued_path(kind, split, start), map_location=device, weights_only=False)


def log_event(**values):
    with (OUT / "events.jsonl").open("a") as stream:
        stream.write(json.dumps(dict(time=time.time(), **values)) + "\n")


def read_range(split, begin, end):
    values = []
    for record in manifest()["files"][split]:
        with h5py.File(DATA / record["file"], "r") as source:
            assert min(len(source[c]) for c in ("u", "v")) >= end
            raw = np.stack([source[c][begin:end, ::2, ::2] for c in ("u", "v")]).astype(np.float32)
        assert np.isfinite(raw).all() and raw.shape == (2, 20, 32, 64)
        values.append(torch.from_numpy(raw))
    return torch.stack(values)


def load_predictor(kind):
    ck = torch.load(SOURCE / kind / "representation.pt", map_location="cuda", weights_only=False)
    state = ck["model"]
    pretrained = {k[len("encoder."):]: v for k, v in state.items() if k.startswith("encoder.")} if kind == "rae" else None
    codec = LocalGridRepresentation(kind, pretrained).cuda().eval().requires_grad_(False)
    codec.load_state_dict(state)
    model = Joint20DataHeadDiffusion().cuda().eval().requires_grad_(False)
    model.load_state_dict(torch.load(SOURCE / kind / "diffusion.pt", map_location="cuda", weights_only=False)["model"])
    norm = torch.load(SOURCE / "normalization.pt", map_location="cuda", weights_only=False)
    return model, codec, ck, norm


def deterministic_maps(history, mean):
    # All operations use the observed history or already issued forecast mean.
    temporal = history.std(dim=2, keepdim=True).expand_as(mean)
    departure = (mean - history[:, :, -1:]).abs()
    dx = F.pad((mean[..., 1:] - mean[..., :-1]).square(), (0, 1, 0, 0))
    dy = F.pad((mean[..., 1:, :] - mean[..., :-1, :]).square(), (0, 0, 0, 1))
    return dict(constant=torch.ones_like(mean), history_std=temporal,
                forecast_departure=departure, forecast_gradient=(dx + dy).sqrt())


class RiskMapNet(nn.Module):
    """A control head for interval scale only; never changes the PDE forecast.

    Both controls have identical architecture and initialization. The last 40
    inputs contain either zero or the log sampling spread of the future field.
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(120, 48, 3, padding=1), nn.SiLU(),
                                 nn.Conv2d(48, 48, 3, padding=1), nn.SiLU(),
                                 nn.Conv2d(48, 40, 3, padding=1))

    def forward(self, x):
        return self.net(x)


def risk_input(history, forecast, spread, norm, use_spread):
    h = (history - norm["mean"]) / norm["std"]
    mu = (forecast - norm["mean"]) / norm["std"]
    extra = torch.log(spread / norm["std"] + 1e-4) if use_spread else torch.zeros_like(mu)
    return torch.cat([v.flatten(1, 2) for v in (h, mu, extra)], dim=1)


def normalize_map(values, training_mean):
    # Only training mean sets the positive floor. Between-window amplitude is kept.
    scale = training_mean.reshape(1, 2, 1, 1, 1).clamp_min(1e-9)
    return (values / scale).clamp_min(.05)


def period(start):
    return "early" if start < 180 else ("middle" if start < 380 else "late")


def macro(rows, key):
    groups = sorted({r["group"] for r in rows})
    return float(np.mean([np.mean([r[key] for r in rows if r["group"] == g]) for g in groups]))

