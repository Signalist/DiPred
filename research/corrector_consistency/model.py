"""Deterministic whole-window corrector and dimensionless consistency terms."""
import torch
from torch import nn

ARMS = {"A": (), "B": ("direction",), "C": ("magnitude",), "D": ("velocity",),
        "E": ("acceleration",), "F": ("match",),
        "G": ("direction", "magnitude", "velocity", "acceleration")}
WEIGHT = 0.1
MASK_THRESHOLD = 0.05
COS_FLOOR = 0.01
ZERO_THRESHOLD = 1e-6

# Each unordered valid 8-neighborhood edge is counted once. For symmetric
# measures this has exactly the same weighting as counting both directions.
EDGES = (
    (..., slice(None), slice(None, -1), slice(None), slice(1, None)),
    (..., slice(None, -1), slice(None), slice(1, None), slice(None)),
    (..., slice(None, -1), slice(None, -1), slice(1, None), slice(1, None)),
    (..., slice(None, -1), slice(1, None), slice(1, None), slice(None, -1)),
)


def neighbors(field):
    for prefix, a_y, a_x, b_y, b_x in EDGES:
        yield field[prefix, a_y, a_x], field[prefix, b_y, b_x]


class PointCorrector(nn.Module):
    """Same widths as the window model with just X and P input slots."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(80, 16, 1), nn.SiLU(), nn.Conv2d(16, 32, 3, padding=1),
            nn.SiLU(), nn.Conv2d(32, 32, 3, padding=1), nn.SiLU(), nn.Conv2d(32, 40, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x, p):
        assert x.shape == p.shape and p.shape[1:3] == (2, 20)
        fields = torch.cat((x.flatten(1, 2), p.flatten(1, 2)), dim=1)
        return self.net(fields.contiguous(memory_format=torch.channels_last)).reshape_as(p)


def temporal(field):
    first = field.diff(dim=2)
    return first, first.diff(dim=2)


def spatial(field, truth):
    """Return per-case measures in normalized residual coordinates."""
    magnitude = torch.linalg.vector_norm(field, dim=1, keepdim=True)
    eligible = torch.linalg.vector_norm(truth, dim=1, keepdim=True) > MASK_THRESHOLD
    unit = field / magnitude.clamp_min(COS_FLOOR)
    cosine = field.new_zeros(len(field))
    mask_count = torch.zeros(len(field), device=field.device, dtype=torch.int64)
    mag_sq = torch.zeros_like(cosine)
    pair_count = 0
    for (a, b), (am, bm), (av, bv) in zip(neighbors(unit), neighbors(magnitude), neighbors(eligible)):
        valid = (av & bv).flatten(1)
        dot = (a * b).sum(1).clamp(-1, 1).flatten(1)
        cosine = cosine + (dot * valid).sum(1)
        mask_count = mask_count + valid.sum(1)
        mag_sq = mag_sq + (am - bm).square().flatten(1).sum(1)
        pair_count += dot.shape[1]
    cosine = cosine / mask_count.clamp_min(1)
    return dict(cosine=cosine, direction=torch.where(mask_count > 0, 1 - cosine, cosine),
                magnitude=mag_sq / pair_count, valid_pairs=mask_count / pair_count,
                zero=(magnitude <= ZERO_THRESHOLD).float().flatten(1).mean(1))


def loss(pred, truth, arm):
    active = ARMS[arm]
    total = (pred - truth).square().mean()
    if "direction" in active or "magnitude" in active:
        stats = spatial(pred, truth)
        for name in ("direction", "magnitude"):
            if name in active:
                total = total + WEIGHT * stats[name].mean()
    if any(name in active for name in ("velocity", "acceleration", "match")):
        d1, d2 = temporal(pred)
        if "velocity" in active:
            total = total + WEIGHT * d1.square().mean()
        if "acceleration" in active:
            total = total + WEIGHT * d2.square().mean()
        if "match" in active:
            total = total + WEIGHT * (d1 - truth.diff(dim=2)).square().mean()
    return total
