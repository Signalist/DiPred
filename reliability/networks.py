"""Checkpoint-compatible dynamics-conditioned native four-step DiPred."""

import torch
from torch import nn
import torch.nn.functional as F

from .backbone import Physical20Net, PhysicalDiffusion20


class DynamicsNet(Physical20Net):
    def __init__(self, scale):
        super().__init__()
        # Add observed temporal increments to history and noisy future. These
        #40channels are conditioning information, not forecast time subblocks.
        self.first.a = nn.Conv2d(120, 48, 3, padding=1)
        self.first.skip = nn.Conv2d(120, 48, 1)
        self.register_buffer("scale", scale.clone())

    def forward(self, x, t, h):
        if h.shape[1:] != (2, 20, 32, 64) or x.shape != h.shape:
            raise ValueError("Expected equal[B,2,20,32,64]tensors.")
        b, _, _, ny, nx = h.shape
        increments = torch.cat([torch.zeros_like(h[:, :, :1]), h.diff(dim=2)], dim=2) / self.scale
        u = t.to(h.dtype) / 99
        embedding = self.time(torch.stack([u, u * u, torch.sin(torch.pi * u), torch.cos(torch.pi * u)], 1))
        inputs = torch.cat([h.reshape(b, 40, ny, nx), x.reshape(b, 40, ny, nx),
                            increments.reshape(b, 40, ny, nx)], dim=1)
        first = self.first(inputs, embedding)
        encoded = self.enc1(self.down1(first), embedding)
        middle = self.mid(self.down2(encoded), embedding)
        upper = self.up1(torch.cat([F.interpolate(middle, size=encoded.shape[-2:],
            mode="bilinear", align_corners=False), encoded], 1), embedding)
        upper = self.up2(torch.cat([F.interpolate(upper, size=first.shape[-2:],
            mode="bilinear", align_corners=False), first], 1), embedding)
        return self.out(upper).reshape_as(h)


class DynamicsDiffusion20(PhysicalDiffusion20):
    def __init__(self, scale):
        super().__init__()
        self.net = DynamicsNet(scale)

    @torch.no_grad()
    def rollout(self, history, seed=500000, sign=0):
        """All four calls refine the same complete twenty-frame state."""
        generator = torch.Generator(device=history.device).manual_seed(seed)
        x = torch.randn(history.shape, device=history.device,
                        dtype=history.dtype, generator=generator) * sign
        for t in reversed(range(4)):
            alpha = self.abar[t]
            labels = torch.full((len(history),), t * 33, device=history.device, dtype=torch.long)
            clean = self.net(x, labels, history)
            noise = (x - alpha.sqrt() * clean) / (1 - alpha).sqrt()
            previous = self.abar[t - 1] if t else x.new_tensor(1.)
            x = previous.sqrt() * clean + (1 - previous).sqrt() * noise
        return x
