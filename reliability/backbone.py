"""Readable standalone network and sampler for the archived physical direct20.

Only PyTorch is needed. State-dict keys match the original frozen checkpoints.
Every network call consumes and emits all20 future frames simultaneously.
"""

import math

import torch
from torch import nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.a = nn.Conv2d(cin, cout, 3, padding=1)
        self.b = nn.Conv2d(cout, cout, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, cout)
        self.norm2 = nn.GroupNorm(8, cout)
        self.time = nn.Linear(32, cout)
        self.skip = nn.Identity() if cin == cout else nn.Conv2d(cin, cout, 1)

    def forward(self, x, t):
        y = F.silu(self.norm1(self.a(x)) + self.time(t)[:, :, None, None])
        return self.skip(x) + self.b(F.silu(self.norm2(y)))


class Physical20Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.time = nn.Sequential(nn.Linear(4, 32), nn.SiLU(), nn.Linear(32, 32))
        # Retain original parameter registration order for training/grad clipping.
        self.down1 = nn.Conv2d(48, 96, 4, 2, 1)
        self.enc1 = ResidualBlock(96, 96)
        self.down2 = nn.Conv2d(96, 176, 4, 2, 1)
        self.mid = ResidualBlock(176, 176)
        self.up1 = ResidualBlock(176 + 96, 96)
        self.up2 = ResidualBlock(96 + 48, 48)
        self.first = ResidualBlock(80, 48)
        self.out = nn.Conv2d(48, 40, 3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x, iteration, history):
        if history.shape != x.shape or history.shape[1:] != (2, 20, 32, 64):
            raise ValueError("Expected equal tensors [B,2,20,32,64].")
        b = len(history)
        u = iteration.to(history.dtype) / 99
        embedding = self.time(torch.stack([u, u * u, torch.sin(torch.pi * u), torch.cos(torch.pi * u)], 1))
        # Fold all20time positions into channels, without temporal recursion.
        inputs = torch.cat([history.reshape(b, 40, 32, 64), x.reshape(b, 40, 32, 64)], 1)
        first = self.first(inputs, embedding)
        middle = self.enc1(self.down1(first), embedding)
        deep = self.mid(self.down2(middle), embedding)
        deep = self.up1(torch.cat([F.interpolate(deep, size=middle.shape[-2:], mode="bilinear", align_corners=False), middle], 1), embedding)
        deep = self.up2(torch.cat([F.interpolate(deep, size=first.shape[-2:], mode="bilinear", align_corners=False), first], 1), embedding)
        return self.out(deep).reshape_as(history)


class PhysicalDiffusion20(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = Physical20Net()
        t = torch.arange(5, dtype=torch.float64) / 4
        alpha = torch.cos((t + .008) / 1.008 * math.pi / 2).square()
        alpha = alpha / alpha[0]
        betas = 1 - alpha[1:] / alpha[:-1]
        betas[-1] = 1.
        self.register_buffer("betas", betas.float())
        self.register_buffer("abar", torch.cumprod(1 - self.betas, 0))

    def sample(self, history, seed=None):
        """Differentiable four-step graph; seed=None gives zero initial noise.

        New deterministic controls train this exact zero-noise graph using only
        final-future MSE. The diffusion trains native_loss instead.
        """
        if seed is None:
            x = torch.zeros_like(history)
        else:
            generator = torch.Generator(device=history.device).manual_seed(seed)
            x = torch.randn(history.shape, device=history.device, dtype=history.dtype, generator=generator)
        for step in reversed(range(4)):
            alpha = self.abar[step]
            iteration = torch.full((len(history),), step * 33, device=history.device, dtype=torch.long)
            clean = self.net(x, iteration, history)
            noise = (x - alpha.sqrt() * clean) / (1 - alpha).sqrt()
            previous = self.abar[step - 1] if step else x.new_tensor(1.)
            x = previous.sqrt() * clean + (1 - previous).sqrt() * noise
        return x

    def native_loss(self, history, future):
        step = torch.randint(4, (len(history),), device=history.device)
        alpha = self.abar[step, None, None, None, None]
        noise = torch.randn_like(future)
        noisy_future = alpha.sqrt() * future + (1 - alpha).sqrt() * noise
        predicted = self.net(noisy_future, step * 33, history)
        return ((predicted - future).square() / (1 - alpha)).mean()


def configure():
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
