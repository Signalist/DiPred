"""A bounded learned interval-scale adapter, with a matched no-memory control."""

import math

import torch
from torch import nn

from conditional_online_risk import risk_features


def adapter_inputs(history, point, lower, upper, previous_point):
    width = (0.5 * (upper - lower)).clamp_min(1e-6)
    known = risk_features(history, point, width)[:, :, :5]
    known = known.reshape(len(point), 10, 20, 32, 64)
    available = previous_point is not None
    if available:
        # The current history is the revealed target of the previous forecast.
        previous_error = (history - previous_point).abs()
        remembered = ((previous_error + 1e-4) / width).log().clamp(-5, 5)
    else:
        remembered = torch.zeros_like(point)
    lead = torch.linspace(0.05, 1, 20, device=point.device, dtype=point.dtype)[None, None, :, None, None]
    lead = lead.expand(len(point), 1, 20, 32, 64)
    mask = torch.full_like(lead, float(available))
    return torch.cat([known, remembered, lead, mask], 1)


class FeedbackScaleAdapter(nn.Module):
    def __init__(self, use_memory):
        super().__init__()
        self.use_memory = use_memory
        self.net = nn.Sequential(
            nn.Conv3d(14, 16, 3, padding=1), nn.GroupNorm(4, 16), nn.SiLU(),
            nn.Conv3d(16, 16, 3, padding=(1, 2, 2), dilation=(1, 2, 2)),
            nn.GroupNorm(4, 16), nn.SiLU(), nn.Conv3d(16, 2, 1))
        # Both controls start at exactly the frozen raw interval.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def bounds(self, inputs, lower, upper):
        if not self.use_memory:
            inputs = inputs.clone()
            inputs[:, 10:12] = 0
        delta = math.log(2) * torch.tanh(self.net(inputs)) * inputs[:, 13:14]
        midpoint = 0.5 * (lower + upper)
        radius = 0.5 * (upper - lower) * delta.exp()
        return midpoint - radius, midpoint + radius

    def loss(self, inputs, lower, upper, residual):
        lo, hi = self.bounds(inputs, lower, upper)
        left, right = residual - lo, residual - hi
        return (torch.maximum(0.05 * left, -0.95 * left)
                + torch.maximum(0.95 * right, -0.05 * right)).mean()
