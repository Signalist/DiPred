"""A separate interval-scale state; no forecast-network weights are updated."""

import torch


class OnlineIntervalScale:
    def __init__(self, calibrated_factors, physical_std):
        self.initial = torch.as_tensor(calibrated_factors).reshape(1, 2, 1, 1, 1)
        self.physical_std = physical_std
        self.scale = None
        self.pending = None

    def issue(self, mean, sample_std):
        if self.pending is not None:
            raise RuntimeError("Reveal the previous complete20frame truth before issuing again.")
        if self.scale is None:
            self.scale = self.initial.to(mean).expand(len(mean), 2, 1, 1, 1).clone()
        sd = sample_std.clamp_min(self.physical_std.to(mean) * 1e-4)
        radius = self.scale * sd
        self.pending = (mean.detach().clone(), sd.detach().clone())
        return mean - radius, mean + radius

    def reveal(self, complete_truth):
        if self.pending is None:
            raise RuntimeError("No issued forecast is awaiting feedback.")
        mean, sd = self.pending
        if complete_truth.shape != mean.shape or mean.shape[2] != 20:
            raise ValueError("Feedback must match the complete issued20frame block.")
        ratio = (complete_truth.to(mean) - mean).abs() / sd
        q = torch.quantile(ratio.flatten(2), .9, dim=2).reshape(len(mean), 2, 1, 1, 1)
        self.scale = (self.scale * q.clamp(1e-6, 1e3)).sqrt()
        self.pending = None
