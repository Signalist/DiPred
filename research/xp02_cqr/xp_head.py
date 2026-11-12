"""XP quantile head: complete normalized history and forecast fields only."""
import torch
from torch import nn
from torch.nn import functional as F

BASE_INDICES = (0, 1)


class XPHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(4, 24, 3, padding=1),
            nn.GroupNorm(4, 24), nn.SiLU(),
            nn.Conv3d(24, 24, 3, padding=(1, 2, 2), dilation=(1, 2, 2)),
            nn.GroupNorm(4, 24), nn.SiLU(), nn.Conv3d(24, 4, 1),
        )

    def forward(self, base, history):
        if history.shape[1:] != (2, 20, *base.shape[-2:]):
            raise ValueError('Expected full 20-frame history and prediction.')
        value = self.net(torch.cat((base[:, list(BASE_INDICES)], history), dim=1))
        mid = value[:, :2]
        radius = F.softplus(value[:, 2:]) + 1e-4
        return mid - radius, mid + radius
