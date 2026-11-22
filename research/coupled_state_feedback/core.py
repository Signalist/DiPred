"""Four-call differentiable sampler; no truth accepted by any generation API."""
import copy
import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class StateUpdate(nn.Module):
    def __init__(self):
        super().__init__()
        self.candidate = nn.Sequential(
            nn.Conv3d(22, 8, 3, padding=1, padding_mode='replicate'), nn.SiLU(),
            nn.Conv3d(8, 8, 3, padding=1, padding_mode='replicate'), nn.SiLU())
        self.gate = nn.Conv3d(22, 8, 1)

    def forward(self, r, p, previous, h, step, coupled):
        dt = torch.cat([torch.zeros_like(p[:, :, :1]), p.diff(dim=2)], 2)
        change = torch.zeros_like(p) if previous is None else p - previous
        if not coupled:
            p, dt, change = torch.zeros_like(p), torch.zeros_like(dt), torch.zeros_like(change)
        context = torch.cat([h[:, :, -1:], h.mean(2, keepdim=True),
                             h[:, :, -1:] - h[:, :, -2:-1]], 1).expand(-1, -1, 20, -1, -1)
        lead = torch.linspace(0, 1, 20, device=h.device, dtype=h.dtype)[None, None, :, None, None]
        lead = lead.expand(len(h), 1, 20, 32, 64)
        label = torch.full_like(lead, step / 3)
        inputs = torch.cat([r, p, dt, change, context, lead, label], 1)
        gate = torch.sigmoid(self.gate(inputs))
        return (1 - gate) * r + gate * self.candidate(inputs)


class Forecast(nn.Module):
    def __init__(self, native, arm):
        super().__init__()
        assert arm in ('N', 'F', 'O', 'C')
        self.arm = arm
        self.base = copy.deepcopy(native).eval().requires_grad_(False)
        if arm != 'N':
            self.base.net.up2.requires_grad_(True)
            self.base.net.out.requires_grad_(True)
        if arm in ('O', 'C'):
            # Forking isolates identical O/C initialization from all other RNG use.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(42)
                self.update = StateUpdate()
                self.feedback = nn.Conv2d(160, 48, 1)
                nn.init.zeros_(self.feedback.weight)
                nn.init.zeros_(self.feedback.bias)
            self.update.to(next(native.parameters()).device)
            self.feedback.to(next(native.parameters()).device)

    def denoise(self, x, labels, h, r):
        if self.arm in ('N', 'F'):
            return self.base.net(x, labels, h)
        net = self.base.net
        b, _, _, ny, nx = h.shape
        increments = torch.cat([torch.zeros_like(h[:, :, :1]), h.diff(dim=2)], 2) / net.scale
        u = labels.to(h.dtype) / 99
        embedding = net.time(torch.stack([u, u*u, torch.sin(torch.pi*u), torch.cos(torch.pi*u)], 1))
        inputs = torch.cat([h.reshape(b, 40, ny, nx), x.reshape(b, 40, ny, nx),
                            increments.reshape(b, 40, ny, nx)], 1)
        first = net.first(inputs, embedding) + self.feedback(r.reshape(b, 160, ny, nx))
        encoded = net.enc1(net.down1(first), embedding)
        middle = net.mid(net.down2(encoded), embedding)
        upper = net.up1(torch.cat([F.interpolate(middle, size=encoded.shape[-2:],
            mode='bilinear', align_corners=False), encoded], 1), embedding)
        upper = net.up2(torch.cat([F.interpolate(upper, size=first.shape[-2:],
            mode='bilinear', align_corners=False), first], 1), embedding)
        return net.out(upper).reshape_as(h)

    def forward(self, h, noise, trace=None, feedback=True, checkpointing=False):
        assert h.shape == noise.shape and h.shape[1:] == (2, 20, 32, 64)
        x = noise.clone()
        r = h.new_zeros((len(h), 8, 20, 32, 64))
        previous = None
        for step in (3, 2, 1, 0):
            r_input = r
            labels = torch.full((len(h),), step*33, device=h.device, dtype=torch.long)
            state = r if feedback else torch.zeros_like(r)
            if checkpointing:
                p = checkpoint(self.denoise, x, labels, h, state, use_reentrant=False)
            else:
                p = self.denoise(x, labels, h, state)
            alpha = self.base.abar[step]
            residual = (x - alpha.sqrt()*p) / (1-alpha).sqrt()
            prev = self.base.abar[step-1] if step else x.new_tensor(1.)
            x = prev.sqrt()*p + (1-prev).sqrt()*residual
            if step and self.arm in ('O', 'C'):
                if checkpointing:
                    r = checkpoint(self.update, r, p, previous, h, step, self.arm == 'C', use_reentrant=False)
                else:
                    r = self.update(r, p, previous, h, step, self.arm == 'C')
            if trace is not None:
                trace.append(dict(step=step, p=p, r=r, r_input=r_input, x=x))
            previous = p
        return x

    def trainable_state(self):
        return {k: p.detach().cpu().clone() for k, p in self.named_parameters() if p.requires_grad}

    def optimizer(self):
        backbone = [p for p in self.base.parameters() if p.requires_grad]
        groups = [dict(params=backbone, lr=1e-5)]
        if self.arm in ('O', 'C'):
            groups.append(dict(params=list(self.update.parameters()) + list(self.feedback.parameters()), lr=1e-4))
        return torch.optim.Adam(groups, weight_decay=0.)


def scores(p, y, unbiased=True):
    """Already in the fixed train-standard-deviation units; full pair gradients."""
    m = len(p)
    assert m > 1 and p.shape[1:] == y.shape
    ordered = p.sort(0).values
    w = (2*torch.arange(m, device=p.device, dtype=p.dtype)-m+1).reshape(m, *([1]*(p.ndim-1)))
    denominator = m*(m-1) if unbiased else m*m
    crps = (p-y).abs().mean() - (ordered*w).sum(0).mean()/denominator
    flat = p.flatten(1)
    first = torch.linalg.vector_norm(flat-y.flatten(), dim=1).mean()/math.sqrt(flat.shape[1])
    pair = sum(torch.linalg.vector_norm(flat[i]-flat[j])/math.sqrt(flat.shape[1])
               for i in range(m) for j in range(i))
    es = first - pair/denominator
    return crps, es


def objective(p, y):
    crps, es = scores(p, y)
    return crps + .1*es
