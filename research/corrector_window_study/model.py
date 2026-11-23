"""Capacity-matched, member-wise whole-window correction."""
import torch
from torch import nn
from torch.nn import functional as F

ARMS = {
    "A": ("summary", False, False, "crps"),
    "B": ("full", False, False, "crps"),
    "C": ("summary", True, False, "crps"),
    "D": ("full", True, False, "crps"),
    "E": ("full", False, True, "crps"),
    "F": ("full", True, False, "mse"),
}
# Put the member-specific block last so the shared 1x1 projection is evaluated once.
SLOTS = (("recent", 40), ("summary", 6), ("earlier", 40), ("mean", 40),
         ("spread", 40), ("memory", 40), ("available", 1), ("coords", 2), ("member", 40))
MEMBERS = 32


def training_normalization(data):
    history = torch.stack([
        data.observed[c["index"], :, c["start"]-380:c["start"]-360] for c in data.cases
    ]).double()
    mean = history.mean((0, 2, 3, 4)).float().view(1, 2, 1, 1, 1)
    scale = history.std((0, 2, 3, 4), unbiased=False).float().view_as(mean)
    assert (scale > 0).all()
    return mean, scale


def inputs(release, arm, normalization):
    recent, earlier, native, average, spread, memory = release
    mode, use_memory, use_earlier, _ = ARMS[arm]
    loc, scale = normalization
    recent = (recent - loc) / scale
    recent_full = recent.flatten(1, 2)
    summary = torch.cat((recent[:, :, -1], recent.mean(2),
                         recent[:, :, -1] - recent[:, :, -2]), 1)
    h, w = native.shape[-2:]
    y, x = torch.meshgrid(torch.linspace(-1, 1, h, device=native.device),
                          torch.linspace(-1, 1, w, device=native.device), indexing="ij")
    coords = torch.stack((x, y)).expand(len(native), -1, -1, -1)
    available = native.new_full((len(native), 1, h, w), int(use_memory))
    older_full = ((earlier - loc) / scale).flatten(1, 2)
    memory_full = (memory / scale).flatten(1, 2)
    share = torch.cat((
        recent_full if mode == "full" else torch.zeros_like(recent_full),
        summary if mode == "summary" else torch.zeros_like(summary),
        older_full if use_earlier else torch.zeros_like(older_full),
        ((average - loc) / scale).flatten(1, 2), (spread / scale).flatten(1, 2),
        memory_full if use_memory else torch.zeros_like(memory_full),
        available, coords,
    ), dim=1).contiguous(memory_format=torch.channels_last)
    standardized = (native - loc[:, None]) / scale[:, None]
    return share, standardized


class WindowCorrector(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Conv2d(249, 16, 1)
        self.spatial = nn.Sequential(
            nn.SiLU(), nn.Conv2d(16, 32, 3, padding=1),
            nn.SiLU(), nn.Conv2d(32, 32, 3, padding=1),
            nn.SiLU(), nn.Conv2d(32, 40, 1),
        )
        nn.init.zeros_(self.spatial[-1].weight)
        nn.init.zeros_(self.spatial[-1].bias)

    def forward(self, shared, member):
        b, m, _, _, h, w = member.shape
        assert m == MEMBERS and shared.shape == (b, 209, h, w)
        member_field = member.reshape(b*m, 40, h, w).contiguous(memory_format=torch.channels_last)
        weight = self.projection.weight
        shared_projection = F.conv2d(shared, weight[:, :209], self.projection.bias)
        member_projection = F.conv2d(member_field, weight[:, 209:]).reshape(b, m, 16, h, w)
        features = member_projection + shared_projection[:, None]
        features = features.flatten(0, 1).contiguous(memory_format=torch.channels_last)
        return self.spatial(features).reshape_as(member)


def empirical_crps(ensemble, target):
    """Exact empirical CRPS along member axis, with the M² denominator."""
    m = ensemble.shape[1]
    ordered = ensemble.sort(dim=1).values
    coef = (2 * torch.arange(m, device=ensemble.device, dtype=ensemble.dtype) + 1 - m) / m**2
    coef = coef.reshape(1, m, *([1] * (ensemble.ndim - 2)))
    return (ensemble - target[:, None]).abs().mean(1) - (ordered * coef).sum(1)


def explicit_crps(ensemble, target):
    """Reference for the numerical and gradient checks on small samples."""
    return ((ensemble - target[:, None]).abs().mean(1)
            - (ensemble[:, :, None] - ensemble[:, None, :]).abs().mean((1, 2)) / 2)


def objective(predicted, target, loss):
    if loss == "crps":
        return empirical_crps(predicted, target).mean()
    return (predicted - target[:, None]).square().mean()
