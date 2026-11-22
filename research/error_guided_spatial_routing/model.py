"""Frozen DiPred wrapper with a differentiable 3x3 trust router.

The native network is never modified in-place.  The wrapper calls its frozen
modules and inserts the router after the native 48-channel first feature.
"""
import math
import torch
from torch import nn
import torch.nn.functional as F


class SpatialRouter(nn.Module):
    def __init__(self, trust: bool):
        super().__init__()
        self.trust = bool(trust)
        # 48 feature channels + three frozen-prediction context channels and
        # two normalized spatial coordinates.  Both arms have identical size.
        self.context = nn.Conv2d(53, 8, 1)
        self.gate = nn.Conv2d(53, 1, 1)
        nn.init.zeros_(self.context.bias)
        nn.init.constant_(self.gate.bias, -1.5)
        self.register_buffer("offsets", torch.tensor(
            [[-1,-1],[-1,0],[-1,1],[0,-1],[0,1],[1,-1],[1,0],[1,1]],
            dtype=torch.long), persistent=False)

    def forward(self, f, aux, risk, enabled=True):
        if not enabled:
            return f
        n, c, h, w = f.shape
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, h, device=f.device, dtype=f.dtype),
            torch.linspace(-1, 1, w, device=f.device, dtype=f.dtype), indexing="ij")
        coords = torch.stack((xx, yy)).expand(n, -1, -1, -1)
        z = torch.cat((f, aux, coords), 1)
        logits = self.context(z)
        strength = 0.25 * torch.sigmoid(self.gate(z))
        # Replicate padding is a legal boundary rule and never connects the
        # left/right or top/bottom edges periodically.
        padded = F.pad(f, (1, 1, 1, 1), mode="replicate")
        neigh = F.unfold(padded, 3).reshape(n, c, 9, h, w)
        center = neigh[:, :, 4:5]
        neigh = torch.cat((neigh[:, :, :4], neigh[:, :, 5:]), 2)
        diff = neigh - center
        rp = F.pad(risk, (1, 1, 1, 1), mode="replicate")
        rnb = F.unfold(rp, 3).reshape(n, 1, 9, h, w)
        rnb = torch.cat((rnb[:, :, :4], rnb[:, :, 5:]), 2).squeeze(1)
        if self.trust:
            weights = torch.softmax(logits - rnb, dim=1)
        else:
            weights = torch.softmax(logits, dim=1)
        routed = (weights[:, None] * diff).sum(2)
        return f + strength * routed


class TrustDiffusion(nn.Module):
    """Differentiable four-step sampler around a frozen native model."""
    def __init__(self, native_model, trust: bool):
        super().__init__()
        self.native = native_model
        for p in self.native.parameters():
            p.requires_grad_(False)
        self.router = SpatialRouter(trust)
        self.trust = bool(trust)

    @property
    def abar(self):
        return self.native.abar

    def forward(self, x, iteration, history, aux, risk, enabled=True):
        net = self.native.net
        if history.shape != x.shape or history.shape[1:] != (2, 20, 32, 64):
            raise ValueError("expected [B,2,20,32,64]")
        b = len(history)
        u = iteration.to(history.dtype) / 99
        emb = net.time(torch.stack((u, u*u, torch.sin(math.pi*u),
                                    torch.cos(math.pi*u)), 1))
        increments = torch.cat((torch.zeros_like(history[:, :, :1]),
                                history.diff(dim=2)), 2) / net.scale
        inputs = torch.cat((history.reshape(b,40,32,64),
                            x.reshape(b,40,32,64),
                            increments.reshape(b,40,32,64)), 1)
        first = net.first(inputs, emb)
        first = self.router(first, aux, risk, enabled=enabled)
        encoded = net.enc1(net.down1(first), emb)
        middle = net.mid(net.down2(encoded), emb)
        upper = net.up1(torch.cat((F.interpolate(middle, size=encoded.shape[-2:],
            mode="bilinear", align_corners=False), encoded), 1), emb)
        upper = net.up2(torch.cat((F.interpolate(upper, size=first.shape[-2:],
            mode="bilinear", align_corners=False), first), 1), emb)
        return net.out(upper).reshape_as(history)

    def sample(self, history, aux, risk, noise, enabled=True):
        x = noise
        for step in reversed(range(4)):
            label = torch.full((len(history),), step*33, device=history.device,
                               dtype=torch.long)
            clean = self.forward(x, label, history, aux, risk, enabled=True) if enabled else \
                self.native.net(x, label, history)
            alpha = self.abar[step]
            residual = (x - alpha.sqrt()*clean) / (1-alpha).sqrt()
            previous = self.abar[step-1] if step else x.new_tensor(1.)
            x = previous.sqrt()*clean + (1-previous).sqrt()*residual
        return x


def crps(paths, truth):
    """Exact empirical CRPS, retaining the complete member-distance term."""
    n = paths.shape[0]
    first = (paths-truth).abs().mean(0)
    ordered = paths.sort(dim=0).values
    weights = (2*torch.arange(n, device=paths.device, dtype=paths.dtype)-n+               + 1).reshape(n, *([1]*(paths.ndim-1)))
    pair = (ordered*weights).sum(0)/(n*n)
    return (first-pair).mean()
