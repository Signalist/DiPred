"""Short revealed-window memory for conditional tail-risk calibration."""

from collections import deque
import math

import torch

from conditional_online_risk import ConditionalRiskCalibrator
from causal_shape_calibration import shape_from_scores
from experiment_reliability_structured_e3 import grouped_quantile
from quantile_online_risk import controller as reference_controller


MODES = ("quantile_dynamic", "memory_dynamic", "memory_fastcal", "memory_scale_fast")


def fit_memory_quantile(memory, start):
    """Average blockwise pinball gradients, profiling an intercept per block.

    Each stored target uses that block's pre-feedback separable shape. No
    future labels are available, and different blocks keep separate intercepts
    so that global magnitude changes do not directly dictate risk slopes.
    """
    blocks = []
    for features, target in memory:
        x = features.flatten(3).reshape(-1, features.shape[2], features[0, 0, 0].numel())
        y = target.flatten(2).reshape(-1, target[0, 0].numel())
        blocks.append((x, y))
    beta = start.reshape(-1, start.shape[-1]).clone()
    for iteration in range(12):
        gradient = torch.zeros_like(beta)
        for x, y in blocks:
            prediction = torch.bmm(beta[:, None], x).squeeze(1)
            intercept = torch.quantile(y - prediction, 0.9, dim=-1, keepdim=True)
            below = (y < prediction + intercept).to(x.dtype)
            gradient += torch.bmm(x, (below - 0.9)[:, :, None]).squeeze(-1) / x.shape[-1]
        gradient = gradient / len(blocks) + 0.025 * beta
        beta = (beta - 0.5 / math.sqrt(iteration + 1) * gradient).clamp(-1, 1)
    return beta.reshape_as(start)


class MemoryQuantileCalibrator(ConditionalRiskCalibrator):
    def __init__(self, mode, initial, batch):
        super().__init__("ridge_dynamic", initial, batch)
        self.memory_mode = mode
        self.memory = deque(maxlen=3)

    def reveal(self, residual):
        if self.pending is None or residual.shape != self.pending[0].shape:
            raise RuntimeError("Memory accepts only the complete issued 20-frame block.")
        mid, width, lower, upper, features = self.pending
        error = (residual - mid).abs()
        target = ((error + 1e-4) / width / self.shape).log().clamp(-8, 8)
        self.memory.append((features.clone(), target.clone()))
        estimate = fit_memory_quantile(self.memory, self.beta)
        self.beta = 0.5 * self.beta + 0.5 * estimate
        adjusted = error / width / self.conditional_shape(features)
        fitted_shape = shape_from_scores(adjusted, "separable_integral")
        self.shape = (self.shape * fitted_shape).sqrt()
        self.shape = self.shape / self.shape.log().mean((2, 3, 4), keepdim=True).exp()
        fitted_amplitude = grouped_quantile(adjusted / self.shape, "global").clamp(1e-6, 1e3)
        if self.memory_mode == "memory_scale_fast" and self.revealed_windows > 0:
            self.amplitude = (self.amplitude * fitted_amplitude).sqrt()
        else:
            self.amplitude = fitted_amplitude
        coverage = ((residual >= lower) & (residual <= upper)).float().mean((2, 3, 4), keepdim=True)
        gain = 0.5 if self.memory_mode == "memory_dynamic" else 2.0
        self.bias = (self.bias + gain * (0.9 - coverage)).clamp(-math.log(2), math.log(2))
        self.pending = None
        self.revealed_windows += 1


def controller(mode, initial, batch):
    if mode.startswith("memory_"):
        return MemoryQuantileCalibrator(mode, initial, batch)
    return reference_controller(mode, initial, batch)
