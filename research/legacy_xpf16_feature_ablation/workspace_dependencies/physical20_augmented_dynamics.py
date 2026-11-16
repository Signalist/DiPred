"""Joint20 diffusion with original history and optional learned difference cues.

The only learned extension is40 input channels in the first block. Their
weights start at zero, preserving the original conditioning function at init.
"""

import copy

import torch
from torch import nn
import torch.nn.functional as F

from joint20_physical_meanrisk import Joint20PhysicalMeanRisk


def expand_history_input(layer):
    # Creating zero-added columns must not change the training RNG sequence.
    with torch.random.fork_rng(devices=[]):
        expanded = nn.Conv2d(layer.in_channels + 40, layer.out_channels,
                             layer.kernel_size, stride=layer.stride,
                             padding=layer.padding, bias=layer.bias is not None)
    with torch.no_grad():
        expanded.weight.zero_()
        expanded.weight[:, :layer.in_channels].copy_(layer.weight)
        if layer.bias is not None:
            expanded.bias.copy_(layer.bias)
    return expanded


class AugmentedDynamicsNet(nn.Module):
    def __init__(self, original, scale):
        super().__init__()
        for name in ("time", "first", "down1", "enc1", "down2", "mid", "up1", "up2", "out"):
            setattr(self, name, getattr(original, name))
        self.first = copy.deepcopy(self.first)
        self.first.a = expand_history_input(self.first.a)
        self.first.skip = expand_history_input(self.first.skip)
        self.register_buffer("scale", scale.clone())

    def forward(self, x, t, h):
        b, c, nt, ny, nx = h.shape
        assert (c, nt) == (2, 20) and x.shape == h.shape
        increments = torch.cat([torch.zeros_like(h[:, :, :1]), h.diff(dim=2)], dim=2) / self.scale
        u = t.to(h.dtype) / 99
        embedding = self.time(torch.stack([u, u * u, torch.sin(torch.pi * u), torch.cos(torch.pi * u)], 1))
        inputs = torch.cat([h.reshape(b, 40, ny, nx), x.reshape(b, 40, ny, nx),
                            increments.reshape(b, 40, ny, nx)], dim=1)
        first = self.first(inputs, embedding)
        encoded = self.enc1(self.down1(first), embedding)
        middle = self.mid(self.down2(encoded), embedding)
        upper = self.up1(torch.cat([F.interpolate(middle, size=encoded.shape[-2:], mode="bilinear", align_corners=False), encoded], 1), embedding)
        upper = self.up2(torch.cat([F.interpolate(upper, size=first.shape[-2:], mode="bilinear", align_corners=False), first], 1), embedding)
        return self.out(upper).reshape_as(h)


class AugmentedDynamics20(Joint20PhysicalMeanRisk):
    def __init__(self, scale):
        super().__init__()
        self.net = AugmentedDynamicsNet(self.net, scale)
