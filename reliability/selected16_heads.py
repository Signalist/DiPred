"""A physically pruned quantile head retaining five selected feature sources.

Core inputs: history variability2, forecast gradient2, forecast time change2,
last-denoising representation8, coordinates2. There is no historical/error
projection or displacement network in the compact model. All20 future leads
are processed jointly. The point predictor is external and unchanged.
"""
import torch
from torch import nn
import torch.nn.functional as F


BASE_INDICES = (4, 5, 8, 9, 10, 11)
CORE_INDICES = BASE_INDICES + tuple(range(18, 28))
SELECTED_GROUPS = ('history_variability', 'forecast_gradient', 'forecast_time_change',
                   'denoising', 'coordinates')


class Selected16QuantileHead(nn.Module):
    """Use real16-channel convolution and remove unused branches/parameters."""

    def __init__(self, original_state):
        super().__init__()
        self.feature_projection = nn.Conv2d(48, 160, 1)
        self.net = nn.Sequential(
            nn.Conv3d(16, 24, 3, padding=1),
            nn.GroupNorm(4, 24),
            nn.SiLU(),
            nn.Conv3d(24, 24, 3, padding=(1, 2, 2), dilation=(1, 2, 2)),
            nn.GroupNorm(4, 24),
            nn.SiLU(),
            nn.Conv3d(24, 4, 1),
        )
        weights = {key: original_state[key] for key in self.state_dict()}
        weights['net.0.weight'] = original_state['net.0.weight'][:, list(CORE_INDICES)].clone()
        self.load_state_dict(weights, strict=True)
        assert sum(p.numel() for p in self.parameters()) == 34004

    def bounds(self, base, last, history, previous_error, available):
        # Keep the shared stream interface; the three omitted source arguments
        # cannot influence this head. The calibrator may still update online.
        batch, _, _, height, width = base.shape
        features = self.feature_projection(last).reshape(batch, 8, 20, height, width)
        yy = torch.linspace(-1, 1, height, device=base.device, dtype=base.dtype)
        xx = torch.linspace(-1, 1, width, device=base.device, dtype=base.dtype)
        y, x = torch.meshgrid(yy, xx, indexing='ij')
        coordinates = torch.stack([x, y])[None, :, None].expand(batch, 2, 20, height, width)
        value = self.net(torch.cat([base[:, list(BASE_INDICES)], features, coordinates], dim=1))
        midpoint = value[:, :2]
        radius = F.softplus(value[:, 2:]) + 1e-4
        return midpoint - radius, midpoint + radius

    def loss(self, base, last, history, previous_error, residual, available):
        lower, upper = self.bounds(base, last, history, previous_error, available)
        left, right = residual - lower, residual - upper
        return (torch.maximum(.05 * left, -.95 * left)
                + torch.maximum(.95 * right, -.05 * right)).mean()
