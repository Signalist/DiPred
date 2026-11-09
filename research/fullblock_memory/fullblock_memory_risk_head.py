"""Project all revealed error frames into future-lead uncertainty features."""

import torch
from torch import nn
import torch.nn.functional as F

from structured_quantile_uncertainty import StructuredQuantileHead


MODES = ("none", "last", "full")


class FullBlockMemoryRiskHead(StructuredQuantileHead):
    """Identical allocated parameters for three error-information controls.

The forty previous-error channels span two velocity components and20frames.
A dense40->80 projection maps these to four features for each future lead.
All models retain the existing complete physical-history/denoising features.
"""

    def __init__(self, memory_mode, memory_scale):
        if memory_mode not in MODES:
            raise ValueError("Choose none,last or full.")
        super().__init__("rich")
        self.memory_mode = memory_mode
        self.memory_projection = nn.Conv2d(40, 80, 1)
        nn.init.zeros_(self.memory_projection.weight)
        nn.init.zeros_(self.memory_projection.bias)
        self.register_buffer("memory_scale", memory_scale.clone())

    def encode_memory(self, previous_error):
        if self.memory_mode == "none":
            previous_error = torch.zeros_like(previous_error)
        elif self.memory_mode == "last":
            previous_error = previous_error[:, :, -1:].expand_as(previous_error)
        return torch.log1p(previous_error.abs() / self.memory_scale).clamp(0, 5)

    def bounds(self, base, last, history, previous_error, available):
        b, _, _, height, width = base.shape
        historical = self.history_projection(history.reshape(b, 40, height, width))
        memory = self.encode_memory(previous_error).reshape(b, 40, height, width)
        gate = torch.as_tensor(available, dtype=base.dtype, device=base.device)
        gate = gate.expand(b) if gate.numel() == 1 else gate.reshape(b)
        projected = self.memory_projection(memory) * gate[:, None, None, None]
        historical = (historical + projected).reshape(b, 4, 20, height, width)
        features = self.feature_projection(last).reshape(b, 8, 20, height, width)
        yy = torch.linspace(-1, 1, height, device=base.device, dtype=base.dtype)
        xx = torch.linspace(-1, 1, width, device=base.device, dtype=base.dtype)
        y, x = torch.meshgrid(yy, xx, indexing="ij")
        coordinates = torch.stack([x, y])[None, :, None].expand(b, 2, 20, height, width)
        value = self.net(torch.cat([base, historical, features, coordinates], 1))
        midpoint, radius = value[:, :2], F.softplus(value[:, 2:]) + 1e-4
        return midpoint - radius, midpoint + radius

    def loss(self, base, last, history, previous_error, residual, available):
        lower, upper = self.bounds(base, last, history, previous_error, available)
        left, right = residual - lower, residual - upper
        return (torch.maximum(.05 * left, -.95 * left) + torch.maximum(.95 * right, -.05 * right)).mean()
