"""The three requested inference methods on one frozen joint20-frame DiPred.

32-noise computes actual32-path spread at issue time. All three methods use
the same zero-noise point center and the same online calibration design.
"""
import torch
from clean_reliability_common import ROOT
from infer_selected16_quantile import Selected16Predictor

RESULT = ROOT / 'runs/requested16_28_noise32_20261110'


class Noise32Spread:
    """No learned uncertainty head; estimate spread using the original sampler."""

    def __init__(self, model, normalization):
        self.model = model
        self.norm = normalization
        self.last_physical_std = None

    @torch.no_grad()
    def bounds(self, base, last, history, previous_error, available):
        # Match the original cache exactly: convert each sample to physical
        # scale before the sample-standard-deviation reduction, correction1.
        paths = torch.stack([self.model.rollout(history, seed) * self.norm['std'] + self.norm['mean']
                             for seed in range(500000, 500032)])
        physical_std = paths.std(dim=0)
        self.last_physical_std = physical_std.clone()
        radius = 1.6448536269514722 * (physical_std / self.norm['std']).clamp_min(1e-6)
        return -radius, radius


class RequestedThreePredictor(Selected16Predictor):
    """Preserve complete-window predict/reveal/reset guards for all methods."""

    def __init__(self, grid_geometry, method='selected16', device='cuda'):
        if method not in ('selected16', 'original28', 'noise32'):
            raise ValueError('Choose selected16,original28 or noise32.')
        arm = {'selected16': 'compact16', 'original28': 'matched_full28', 'noise32': 'published_warp'}[method]
        super().__init__(grid_geometry, arm=arm, strategy='memory_dynamic', device=device)
        if method == 'noise32':
            self.head = Noise32Spread(self.model, self.norm)
        stored = torch.load(RESULT / 'calibration.pt', map_location=self.device, weights_only=False)
        self.initial = stored[method]['separable_integral']
        self.method = method
        self.reset()
