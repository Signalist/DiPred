"""Ordered XPF quantiles at 0.25% tail and 1% central resolution."""
import torch
from torch import nn
import torch.nn.functional as F


def quantile_levels():
    """129 levels: .25..4.75, 5..95, 95.25..99.75 percent."""
    return [i / 400 for i in range(1, 20)] + [i / 100 for i in range(5, 96)] + [i / 400 for i in range(381, 400)]


class Quantiles129Head(nn.Module):
    """Freeze 19 anchors; learn 72 interior and 38 tail quantiles."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(16, 24, 3, padding=1),
            nn.GroupNorm(4, 24),
            nn.SiLU(),
            nn.Conv3d(24, 24, 3, padding=(1, 2, 2), dilation=(1, 2, 2)),
            nn.GroupNorm(4, 24),
            nn.SiLU(),
            nn.Conv3d(24, 256, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, features, anchors):
        batch, _, t, h, w = features.shape
        if anchors.shape != (batch, 2, 19, t, h, w):
            raise ValueError('Expected anchors [B,2,19,T,H,W].')
        logits = self.net(features).reshape(batch, 2, 128, t, h, w)
        central_logits = logits[:, :, :90].reshape(batch, 2, 18, 5, t, h, w)
        fractions = central_logits.softmax(3).cumsum(3)[:, :, :, :4]
        lower, upper = anchors[:, :, :-1], anchors[:, :, 1:]
        inner = lower.unsqueeze(3) + (upper - lower).unsqueeze(3) * fractions
        central = torch.cat((lower.unsqueeze(3), inner), 3).reshape(batch, 2, 90, t, h, w)
        scale = (anchors[:, :, -1] - anchors[:, :, 0]).clamp_min(1e-4).unsqueeze(2)
        left_gaps = .0125 * scale * F.softplus(logits[:, :, 90:109])
        right_gaps = .0125 * scale * F.softplus(logits[:, :, 109:128])
        left = anchors[:, :, :1] - left_gaps.flip(2).cumsum(2).flip(2)
        right = anchors[:, :, -1:] + right_gaps.cumsum(2)
        return torch.cat((left, central, anchors[:, :, -1:], right), dim=2)
