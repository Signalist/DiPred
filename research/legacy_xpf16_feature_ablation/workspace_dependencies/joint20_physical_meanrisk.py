"""Direct full20 diffusion without state compression; same native-v/mean risk.

The shared spatial U-Net core initializes identically to the latent control.
Input/output layers change to two normalized physical velocity channels.
"""
import torch
from torch import nn

from fiveblock_diffusion import ResidualBlock
from fullblock_diffusion_parameterization import BlockNet
from joint20_data_head_diffusion import Joint20DataHeadDiffusion


class PhysicalIdentity(nn.Module):
    kind = 'physical_identity'

    def encode(self, x, sample=False):
        assert x.shape[1:] == (2, 20, 32, 64)
        return x, x.new_zeros(())

    def decode(self, z):
        assert z.shape[1:] == (2, 20, 32, 64)
        return z


class Physical20Net(BlockNet):
    def __init__(self, latent_net):
        nn.Module.__init__(self)
        for name in ['time', 'down1', 'enc1', 'down2', 'mid', 'up1', 'up2']:
            setattr(self, name, getattr(latent_net, name))
        self.first = ResidualBlock(80, 48)
        self.out = nn.Conv2d(48, 40, 3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)


class Joint20PhysicalMeanRisk(Joint20DataHeadDiffusion):
    def __init__(self):
        super().__init__()
        self.net = Physical20Net(self.net)

    def loss(self, h, y):
        assert h.shape == y.shape and h.shape[1:] == (2, 20, 32, 64)
        t = torch.randint(4, (len(h),), device=h.device)
        a = self.abar[t, None, None, None, None]
        eps = torch.randn_like(y)
        x = a.sqrt() * y + (1-a).sqrt() * eps
        return ((self.net(x, t*33, h) - y).square() / (1-a)).mean()

    def sample_grad(self, h, seed, sign=1):
        assert h.shape[1:] == (2, 20, 32, 64)
        generator = torch.Generator(device=h.device).manual_seed(seed)
        x = torch.randn(h.shape, device=h.device, dtype=h.dtype, generator=generator) * sign
        for t in reversed(range(4)):
            a = self.abar[t]
            clean = self.net(x, torch.full((len(h),), t*33, device=h.device, dtype=torch.long), h)
            eps = (x - a.sqrt() * clean) / (1-a).sqrt()
            previous = self.abar[t-1] if t else x.new_tensor(1.)
            x = previous.sqrt() * clean + (1-previous).sqrt() * eps
        return x
