"""Direct20frame diffusion with a clean-state head and the SAME native-v risk.

The analytic noisy-input term handles the v-to-state conversion, so the learned
head represents the PDE state rather than also reconstructing white noise.
This changes parameterization; it does not change the native risk's noise weights.
"""
import torch

from joint20_state_grid_diffusion import Joint20MeanRiskDiffusion


class Joint20DataHeadDiffusion(Joint20MeanRiskDiffusion):
    def clean(self, x, t, h):
        return self.net(x, t * 33, h)

    def loss(self, h, y):
        assert h.shape == y.shape and h.shape[1:] == (4, 20, 16, 32)
        t = torch.randint(4, (len(h),), device=h.device)
        a = self.abar[t, None, None, None, None]
        eps = torch.randn_like(y)
        x = a.sqrt() * y + (1 - a).sqrt() * eps
        prediction = self.net(x, t * 33, h)
        # v_hat=(sqrt(a)*x-x0_hat)/sqrt(1-a), hence squared v error is
        # squared x0 error divided by1-a. All4noise levels have1-a>0.
        return ((prediction - y).square() / (1 - a)).mean()

    def sample_grad(self, h, seed, sign=1):
        assert h.shape[1:] == (4, 20, 16, 32)
        generator = torch.Generator(device=h.device).manual_seed(seed)
        x = torch.randn(h.shape, device=h.device, dtype=h.dtype, generator=generator) * sign
        for t in reversed(range(4)):
            a = self.abar[t]
            clean = self.net(x, torch.full((len(h),), t * 33, device=h.device, dtype=torch.long), h)
            eps = (x - a.sqrt() * clean) / (1 - a).sqrt()
            prev = self.abar[t - 1] if t else x.new_tensor(1.)
            x = prev.sqrt() * clean + (1 - prev).sqrt() * eps
        return x
