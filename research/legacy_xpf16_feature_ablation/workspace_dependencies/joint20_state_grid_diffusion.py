"""One full-state diffusion block for all20 future frames in frozen VAE codes."""
import math

import torch
from torch import nn
import torch.nn.functional as F

from fiveblock_diffusion import ResidualBlock
from local_grid_diffusion import LocalGridNet
from local_grid_mean_risk import MeanRiskDiffusion


class Joint20GridNet(LocalGridNet):
    def __init__(self):
        super().__init__()
        self.first = ResidualBlock(160, 48)
        self.out = nn.Conv2d(48, 80, 3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x, t, h):
        b, c, nt, ny, nx = h.shape
        assert (c, nt) == (4, 20) and x.shape == h.shape
        u = t.to(h.dtype) / 99
        te = self.time(torch.stack([u, u * u, torch.sin(torch.pi * u), torch.cos(torch.pi * u)], 1))
        f = self.first(torch.cat([h.reshape(b, 80, ny, nx), x.reshape(b, 80, ny, nx)], 1), te)
        g = self.enc1(self.down1(f), te)
        z = self.mid(self.down2(g), te)
        z = self.up1(torch.cat([F.interpolate(z, size=g.shape[-2:], mode='bilinear', align_corners=False), g], 1), te)
        z = self.up2(torch.cat([F.interpolate(z, size=f.shape[-2:], mode='bilinear', align_corners=False), f], 1), te)
        return self.out(z).reshape_as(h)


class Joint20MeanRiskDiffusion(MeanRiskDiffusion):
    def __init__(self):
        nn.Module.__init__(self)
        self.net = Joint20GridNet()
        t = torch.arange(5, dtype=torch.float64) / 4
        a = torch.cos((t + .008) / 1.008 * math.pi / 2).square()
        a = a / a[0]
        betas = 1 - a[1:] / a[:-1]
        betas[-1] = 1.
        self.register_buffer('betas', betas.float())
        self.register_buffer('abar', torch.cumprod(1 - self.betas, 0))

    def loss(self, h, y):
        assert h.shape == y.shape and h.shape[1:] == (4, 20, 16, 32)
        t = torch.randint(4, (len(h),), device=h.device)
        a = self.abar[t, None, None, None, None]
        eps = torch.randn_like(y)
        target = a.sqrt() * eps - (1 - a).sqrt() * y
        return (self.net(a.sqrt() * y + (1 - a).sqrt() * eps, t * 33, h) - target).square().mean()

    def sample_grad(self, h, seed, sign=1):
        assert h.shape[1:] == (4, 20, 16, 32)
        generator = torch.Generator(device=h.device).manual_seed(seed)
        x = torch.randn(h.shape, device=h.device, dtype=h.dtype, generator=generator) * sign
        for t in reversed(range(4)):
            a = self.abar[t]
            v = self.net(x, torch.full((len(h),), t * 33, device=h.device, dtype=torch.long), h)
            clean = a.sqrt() * x - (1 - a).sqrt() * v
            eps = (1 - a).sqrt() * x + a.sqrt() * v
            prev = self.abar[t - 1] if t else x.new_tensor(1.)
            x = prev.sqrt() * clean + (1 - prev).sqrt() * eps
        return x

    def rollout_grad(self, h, seed, sign=1):
        return self.sample_grad(h, seed, sign)

    @torch.no_grad()
    def rollout(self, h, seed, sign=1):
        return self.sample_grad(h, seed, sign)
