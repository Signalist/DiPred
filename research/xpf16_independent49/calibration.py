"""Causal two-tail adaptive calibration for arbitrary XPF16 interval levels.

Each head and each trajectory/channel owns separate two-tail location offsets.
Initialization uses calibration-trajectory residuals only. Updated locations
use only the previously issued, completely revealed 20-frame block. The
small midpoint projection keeps any online pair ordered; it is reported.
This is an empirical controller, not a formal coverage guarantee.
"""
import torch


class TwoTailCalibrator:
    def __init__(self, alpha, offsets, batch, gain=.5):
        if not 0 < alpha < .5:
            raise ValueError('alpha must be between 0 and 0.5')
        self.alpha = float(alpha)
        self.gain = float(gain)
        self.offsets = offsets.reshape(1, 2, 2, 1, 1, 1).expand(batch, -1, -1, -1, -1, -1).clone()
        self.pending = None
        self.projected_points = 0
        self.total_points = 0
        self.revealed_windows = 0

    def issue(self, lower, upper):
        if self.pending is not None:
            raise RuntimeError('Reveal the preceding full window before another forecast')
        if lower.shape != upper.shape or lower.ndim != 5 or lower.shape[2] != 20:
            raise ValueError('Expected [B,2,20,H,W] intervals')
        l = lower + self.offsets[:, 0]
        u = upper + self.offsets[:, 1]
        crossing = l > u
        midpoint = .5 * (l + u)
        l = torch.where(crossing, midpoint, l)
        u = torch.where(crossing, midpoint, u)
        self.projected_points += int(crossing.sum())
        self.total_points += crossing.numel()
        self.pending = (lower.detach(), upper.detach())
        return l, u

    def reveal(self, truth):
        if self.pending is None or truth.shape != self.pending[0].shape:
            raise RuntimeError('Require complete 20-frame revealed residual')
        lo, hi = self.pending
        # Quantiles are per trajectory and channel, not pooled across trajectories.
        left = torch.quantile((truth-lo)[..., ::4, ::4].flatten(2), self.alpha, dim=2)
        right = torch.quantile((truth-hi)[..., ::4, ::4].flatten(2), 1-self.alpha, dim=2)
        estimate = torch.stack((left, right), 1)[..., None, None, None]
        self.offsets = (1-self.gain)*self.offsets + self.gain*estimate
        self.pending = None
        self.revealed_windows += 1
