"""Forecast-conditioned movement versus diagonal rescaling of risk memory.

This transforms latent features of already revealed absolute forecast errors,
not the PDE point forecast or current unrevealed error. Pixel displacements
are learned separately for all20 future leads in one complete block.
"""

import math

import torch
from torch import nn
import torch.nn.functional as F

from .memory_heads import FullBlockMemoryRiskHead


def warp_risk_features(value, displacement):
    """Backward bilinear sampling with signed displacements in grid pixels."""
    batch, channels, time, height, width = value.shape
    if displacement.shape != (batch, 2, time, height, width):
        raise ValueError("Require one x/y pixel displacement per full future lead.")
    yy = (torch.arange(height, device=value.device, dtype=value.dtype) + .5) * (2 / height) - 1
    xx = (torch.arange(width, device=value.device, dtype=value.dtype) + .5) * (2 / width) - 1
    y, x = torch.meshgrid(yy, xx, indexing="ij")
    base = torch.stack([x, y], dim=-1)
    flow = displacement.permute(0, 2, 3, 4, 1).reshape(batch * time, height, width, 2)
    conversion = value.new_tensor([2 / width, 2 / height])
    grid = base[None] - flow * conversion
    flattened = value.permute(0, 2, 1, 3, 4).reshape(batch * time, channels, height, width)
    result = F.grid_sample(flattened, grid, mode="bilinear", padding_mode="border", align_corners=False)
    return result.reshape(batch, time, channels, height, width).permute(0, 2, 1, 3, 4).contiguous()


class DynamicRiskMemoryHead(FullBlockMemoryRiskHead):
    """Hold all existing risk weights fixed; learn one10430parameter module."""

    def __init__(self, memory_scale, original_state, kind):
        if kind not in ("modulate", "warp"):
            raise ValueError("Choose matched diagonal modulation or spatial warping.")
        super().__init__("full", memory_scale)
        self.load_state_dict(original_state, strict=True)
        self.requires_grad_(False)
        self.kind = kind
        self.modifier = nn.Sequential(
            nn.Conv3d(32, 12, 3, padding=1),
            nn.GroupNorm(3, 12),
            nn.SiLU(),
            nn.Conv3d(12, 2, 1),
        )
        nn.init.zeros_(self.modifier[-1].weight)
        nn.init.zeros_(self.modifier[-1].bias)
        assert sum(parameter.numel() for parameter in self.parameters()) == 58770
        assert sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad) == 10430

    def bounds(self, base, last, history, previous_error, available):
        batch, _, _, height, width = base.shape
        historical = self.history_projection(history.reshape(batch, 40, height, width))
        encoded = self.encode_memory(previous_error).reshape(batch, 40, height, width)
        gate = torch.as_tensor(available, dtype=base.dtype, device=base.device)
        gate = gate.expand(batch) if gate.numel() == 1 else gate.reshape(batch)
        projected = self.memory_projection(encoded) * gate[:, None, None, None]
        memory = projected.reshape(batch, 4, 20, height, width)
        features = self.feature_projection(last).reshape(batch, 8, 20, height, width)
        yy = torch.linspace(-1, 1, height, device=base.device, dtype=base.dtype)
        xx = torch.linspace(-1, 1, width, device=base.device, dtype=base.dtype)
        y, x = torch.meshgrid(yy, xx, indexing="ij")
        coordinates = torch.stack([x, y])[None, :, None].expand(batch, 2, 20, height, width)
        context = torch.cat([base, historical.reshape(batch, 4, 20, height, width),
                             features, coordinates, memory], dim=1)
        raw = self.modifier(context)
        if self.kind == "warp":
            # A fixed four-pixel bound is half the existing8x8 regional metric
            # cell. It is not a physical velocity or a tuned advection step.
            transformed = warp_risk_features(memory, 4 * raw.tanh())
        else:
            # Two feature groups receive bounded gains. There is no spatial
            # movement in this equal-parameter, same-context control.
            gain = (math.log(2) * raw.tanh()).exp().repeat_interleave(2, dim=1)
            transformed = memory * gain
        historical = (historical + transformed.reshape(batch, 80, height, width)).reshape(
            batch, 4, 20, height, width
        )
        value = self.net(torch.cat([base, historical, features, coordinates], dim=1))
        midpoint, radius = value[:, :2], F.softplus(value[:, 2:]) + 1e-4
        return midpoint - radius, midpoint + radius

    @torch.no_grad()
    def modifier_fields(self, base, last, history, previous_error, available):
        """Expose learned fields for diagnostics without reading future truth."""
        captured = []
        hook = self.modifier.register_forward_hook(lambda module, inputs, output: captured.append(output))
        try:
            self.bounds(base, last, history, previous_error, available)
        finally:
            hook.remove()
        return captured[0]
