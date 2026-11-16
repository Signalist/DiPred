"""Full-20 physical diffusion: isolate noise range from output parameterization.

The network always processes the complete future block. A v output includes an
analytic noisy-input skip in the clean estimate; it adds no learned module.
"""

import torch

from joint20_physical_meanrisk import Joint20PhysicalMeanRisk


ARMS = ("v_cosine", "v_low_noise", "clean_low_noise")


class NoiseRefinement20(Joint20PhysicalMeanRisk):
    def __init__(self, arm):
        if arm not in ARMS:
            raise ValueError(arm)
        super().__init__()
        self.arm = arm
        if arm.endswith("low_noise"):
            # Fixed before evaluation: geometrically spaced corruption standard
            # deviations, including an exact pure-noise generation endpoint.
            sigma = torch.logspace(-2, 0, 4)
            self.abar.copy_(1 - sigma.square())
            previous = torch.cat([self.abar.new_ones(1), self.abar[:-1]])
            self.betas.copy_(1 - self.abar / previous)

    def clean(self, x, t, h):
        output = self.net(x, t * 33, h)
        if self.arm.startswith("v_"):
            a = self.abar[t].reshape(-1, 1, 1, 1, 1)
            return a.sqrt() * x - (1 - a).sqrt() * output
        return output

    def loss(self, h, y):
        assert h.shape == y.shape and h.shape[1:] == (2, 20, 32, 64)
        t = torch.randint(4, (len(h),), device=h.device)
        a = self.abar[t, None, None, None, None]
        eps = torch.randn_like(y)
        x = a.sqrt() * y + (1 - a).sqrt() * eps
        output = self.net(x, t * 33, h)
        if self.arm.startswith("v_"):
            target = a.sqrt() * eps - (1 - a).sqrt() * y
            return (output - target).square().mean()
        # This is the clean-head equivalent of the same native v-risk. The
        # factorial control exposes any difficulty from its low-noise weights.
        return ((output - y).square() / (1 - a)).mean()

    def sample_grad(self, h, seed, sign=1):
        assert h.shape[1:] == (2, 20, 32, 64)
        generator = torch.Generator(device=h.device).manual_seed(seed)
        x = torch.randn(h.shape, device=h.device, dtype=h.dtype, generator=generator) * sign
        for level in reversed(range(4)):
            t = torch.full((len(h),), level, device=h.device, dtype=torch.long)
            a = self.abar[level]
            clean = self.clean(x, t, h)
            epsilon = (x - a.sqrt() * clean) / (1 - a).sqrt()
            previous = self.abar[level - 1] if level else x.new_tensor(1.)
            x = previous.sqrt() * clean + (1 - previous).sqrt() * epsilon
        return x
