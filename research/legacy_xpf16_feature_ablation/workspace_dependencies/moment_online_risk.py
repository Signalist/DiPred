"""Pool past-window sufficient statistics for causal risk regression."""

import math

import torch

from causal_shape_calibration import shape_from_scores
from conditional_online_risk import ConditionalRiskCalibrator, FEATURE_NAMES
from experiment_reliability_structured_e3 import grouped_quantile
from quantile_online_risk import controller as reference_controller


MODES = ("ridge_dynamic", "quantile_dynamic", "moment_dynamic", "moment_fastcal")


class MomentRiskCalibrator(ConditionalRiskCalibrator):
    """A fixed five-window effective memory, without retaining field targets.

    Compared with per-window ridge, only the Gram/cross moments are pooled.
    The ridge penalty, beta EMA, feature bounds and shape rule stay unchanged.
    All moment updates happen after complete-window feedback.
    """

    def __init__(self, mode, initial, batch):
        super().__init__("ridge_dynamic", initial, batch)
        self.moment_mode = mode
        self.gram = None
        self.cross = None

    def reveal(self, residual):
        if self.pending is None or residual.shape != self.pending[0].shape:
            raise RuntimeError("Moment updates require the complete issued 20-frame window.")
        mid, width, lower, upper, features = self.pending
        error = (residual - mid).abs()
        target = ((error + 1e-4) / width / self.shape).log().clamp(-8, 8).flatten(2)
        target = target - target.mean(-1, keepdim=True)
        count = features[0, 0, 0].numel()
        x = features.flatten(3).reshape(-1, len(FEATURE_NAMES), count)
        y = target.reshape(-1, count, 1)
        gram = torch.bmm(x, x.transpose(1, 2)) / count
        cross = torch.bmm(x, y) / count
        if self.gram is None:
            self.gram, self.cross = gram, cross
        else:
            self.gram = 0.8 * self.gram + 0.2 * gram
            self.cross = 0.8 * self.cross + 0.2 * cross
        identity = torch.eye(len(FEATURE_NAMES), device=x.device, dtype=x.dtype)[None]
        estimate = torch.linalg.solve(self.gram + 0.25 * identity, self.cross)
        estimate = estimate.squeeze(-1).reshape_as(self.beta).clamp(-1, 1)
        self.beta = 0.5 * self.beta + 0.5 * estimate

        # Preserve the previous controller's scale and coverage rules exactly.
        adjusted = error / width / self.conditional_shape(features)
        fitted_shape = shape_from_scores(adjusted, "separable_integral")
        self.shape = (self.shape * fitted_shape).sqrt()
        self.shape = self.shape / self.shape.log().mean((2, 3, 4), keepdim=True).exp()
        self.amplitude = grouped_quantile(adjusted / self.shape, "global").clamp(1e-6, 1e3)
        coverage = ((residual >= lower) & (residual <= upper)).float().mean((2, 3, 4), keepdim=True)
        gain = 2.0 if self.moment_mode == "moment_fastcal" else 0.5
        self.bias = (self.bias + gain * (0.9 - coverage)).clamp(-math.log(2), math.log(2))
        self.pending = None
        self.revealed_windows += 1


def controller(mode, initial, batch):
    if mode.startswith("moment_"):
        return MomentRiskCalibrator(mode, initial, batch)
    return reference_controller(mode, initial, batch)
