"""Frozen-forecast risk fields and causal transport of revealed absolute error.

This module never corrects the point forecast. Future prediction tensors are
known model outputs; future truth only enters reveal after a whole20 block.
"""

import torch
from torch import nn
import torch.nn.functional as F


BASE_CHANNELS = 14
FEATURE_CHANNELS = 19


def grid_derivative(x, axis, spacing):
    shape = x.shape
    if axis == -1:
        derivative = x.diff(dim=-1).reshape(-1, 1, shape[-2], shape[-1] - 1)
        derivative = F.pad(derivative, (0, 1, 0, 0), mode="replicate").reshape(shape)
    else:
        derivative = x.diff(dim=-2).reshape(-1, 1, shape[-2] - 1, shape[-1])
        derivative = F.pad(derivative, (0, 0, 0, 1), mode="replicate").reshape(shape)
    return derivative / spacing.reshape(-1, 1, 1, 1, 1)


def base_features(history, prediction, mean, std, geometry):
    h = (history - mean) / std
    p = (prediction - mean) / std
    last = h[:, :, -1:].expand_as(p)
    variability = h.std(dim=2, keepdim=True).expand_as(p)
    departure = p - last
    dx = grid_derivative(p, -1, torch.ones_like(geometry[:, 0]))
    dy = grid_derivative(p, -2, torch.ones_like(geometry[:, 1]))
    gradient = (dx.square() + dy.square() + 1e-8).sqrt()
    change = torch.cat([p[:, :, :1] - h[:, :, -1:], p.diff(dim=2)], dim=2).abs()
    # Coordinate-grid curl proxy; no pressure/viscosity residual is claimed.
    raw_dx = grid_derivative(prediction, -1, geometry[:, 0])
    raw_dy = grid_derivative(prediction, -2, geometry[:, 1])
    curl = raw_dx[:, 1:2] - raw_dy[:, :1]
    lead = torch.arange(1, 21, device=p.device, dtype=p.dtype).reshape(1, 1, 20, 1, 1) / 20
    lead = lead.expand(len(p), 1, 20, *p.shape[-2:])
    features = torch.cat([p, last, variability, departure, gradient, change, curl, lead], dim=1)
    assert features.shape[1] == BASE_CHANNELS
    return features


def advect_memory(memory, history, prediction, geometry, sign=1):
    """Repeated semi-Lagrangian backward sampling of a nonnegative risk field.

    Stored velocity and x/y/t coordinates are assumed mutually compatible.
    This is a candidate error-motion model, not an exact error PDE.
    """
    b, _, _, ny, nx = prediction.shape
    yy = torch.linspace(-1, 1, ny, device=prediction.device, dtype=prediction.dtype)
    xx = torch.linspace(-1, 1, nx, device=prediction.device, dtype=prediction.dtype)
    gy, gx = torch.meshgrid(yy, xx, indexing="ij")
    base = torch.stack([gx, gy], dim=-1)[None].expand(b, -1, -1, -1)
    conversion = torch.stack([2 * geometry[:, 2] / (geometry[:, 0] * (nx - 1)),
                              2 * geometry[:, 2] / (geometry[:, 1] * (ny - 1))], dim=1)
    field = memory
    previous_velocity = history[:, :, -1]
    frames = []
    for lead in range(20):
        velocity = .5 * (previous_velocity + prediction[:, :, lead])
        displacement = velocity.permute(0, 2, 3, 1) * conversion[:, None, None, :] * sign
        field = F.grid_sample(field, base - displacement, mode="bilinear", padding_mode="border", align_corners=True)
        frames.append(field)
        previous_velocity = prediction[:, :, lead]
    return torch.stack(frames, dim=2)


class CausalRiskMemory:
    def __init__(self, std):
        self.std = std
        self.last = None
        self.pending = None
        self.revealed_windows = 0

    def issue(self, history, prediction, geometry):
        if self.pending is not None:
            raise RuntimeError("Reveal the preceding complete20 window before issuing the next.")
        self.pending = prediction.detach().clone()
        available = self.last is not None
        if not available:
            zero = torch.zeros_like(prediction)
            return dict(static=zero, advected=zero, reverse=zero), False
        static = self.last[:, :, None].expand_as(prediction)
        return dict(static=static,
                    advected=advect_memory(self.last, history, prediction, geometry, sign=1),
                    reverse=advect_memory(self.last, history, prediction, geometry, sign=-1)), True

    def reveal(self, truth):
        if self.pending is None or truth.shape != self.pending.shape:
            raise RuntimeError("Reveal requires exactly the previously issued20-frame shape.")
        self.last = ((truth - self.pending).abs() / self.std)[:, :, -1].detach().clone()
        self.pending = None
        self.revealed_windows += 1


def assemble_features(base, memory, available, spread, mode, use_spread):
    zero = torch.zeros_like(spread)
    if mode in ("static", "advected", "reverse"):
        remembered = (memory[mode] + 1e-4).log() if available else zero
        mask = base.new_full((len(base), 1, 20, *base.shape[-2:]), float(available))
    else:
        remembered = zero
        mask = base.new_zeros((len(base), 1, 20, *base.shape[-2:]))
    uncertainty = (spread + 1e-4).log() if use_spread and mode != "base" else zero
    return torch.cat([base, remembered, mask, uncertainty], dim=1)


class SpatiotemporalRiskNet(nn.Module):
    """One small shared3D head outputs positive risk for both future channels."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(FEATURE_CHANNELS, 16, 3, padding=1), nn.GroupNorm(4, 16), nn.SiLU(),
            nn.Conv3d(16, 16, 3, padding=1), nn.GroupNorm(4, 16), nn.SiLU(),
            nn.Conv3d(16, 2, 1),
        )

    def forward(self, features):
        return self.net(features).clamp(-9, 4)


def preflight():
    std = torch.ones(1, 2, 1, 1, 1, device="cuda")
    h = torch.zeros(1, 2, 20, 32, 64, device="cuda")
    p = torch.zeros_like(h)
    geometry = h.new_tensor([[1., -1., 1.]])
    memory = CausalRiskMemory(std)
    _, available = memory.issue(h, p, geometry)
    assert not available
    try:
        memory.issue(h, p, geometry)
        raise AssertionError("Unrevealed window accepted")
    except RuntimeError:
        pass
    memory.reveal(h)
    _, available = memory.issue(h, p, geometry)
    assert available and memory.revealed_windows == 1
    memory.reveal(h)
    impulse = h.new_zeros((1, 2, 32, 64))
    impulse[:, :, 12, 20] = 1
    velocity = torch.zeros_like(h)
    velocity[:, 0] = 1
    moved = advect_memory(impulse, velocity, velocity, geometry)
    reverse = advect_memory(impulse, velocity, velocity, geometry, sign=-1)
    assert moved[0, 0, 0, 12].argmax() == 21
    assert reverse[0, 0, 0, 12].argmax() == 19
    constant = advect_memory(torch.ones_like(impulse), velocity, velocity, geometry)
    assert torch.allclose(constant, torch.ones_like(constant), atol=1e-6)
    features = base_features(h, p, torch.zeros_like(std), std, geometry)
    full = assemble_features(features, dict(static=h), False, h, "base", False)
    torch.manual_seed(42)
    head = SpatiotemporalRiskNet().cuda()
    out = head(full)
    assert out.shape == h.shape and torch.isfinite(out).all()
    out.square().mean().backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in head.parameters())
    return dict(passed=True, parameters=sum(p.numel() for p in head.parameters()),
                causal_complete_window_guard=True, advection_direction_checked=True,
                output_shape=list(out.shape))
