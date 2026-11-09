"""Frozen known-context scale forecasting; updates only after complete feedback."""

import math
import torch
import torch.nn.functional as F

from .calibration import SmoothedScaleCalibrator, MemoryQuantileCalibrator

MODES = ("base", "hold", "drift", "context")


def original_controller(strategy, initial, batch):
    if strategy == "scale_fastcal":
        return SmoothedScaleCalibrator(strategy, initial, batch)
    return MemoryQuantileCalibrator(strategy, initial, batch)


def context_summary(history, point, lower, upper):
    """Absolute known-field summaries, before per-window feature zscoring."""
    midpoint = 0.5 * (lower + upper)
    width = (0.5 * (upper - lower)).clamp_min(1e-6)

    def log_rms(value):
        return value.square().mean(tuple(range(2, value.ndim))).clamp_min(1e-12).sqrt().log()

    variability = history.std(dim=2, keepdim=True)
    departure = point - history[:, :, -1:]
    change = torch.cat([point[:, :, :1] - history[:, :, -1:], point.diff(dim=2)], 2)
    dx = F.pad(point.diff(dim=-1), (0, 1, 0, 0))
    dy = F.pad(point.diff(dim=-2), (0, 0, 0, 1))
    gradient = (dx.square() + dy.square()).sqrt()
    values = (width.log().mean((2, 3, 4)), log_rms(variability), log_rms(departure),
              log_rms(change), log_rms(gradient), point.mean((2, 3, 4)),
              history[:, :, -1].mean((2, 3)), (midpoint / width).mean((2, 3, 4)))
    return torch.stack(values, -1)


class ContextScaleForecaster:
    def __init__(self, mode, initial, batch, scale_models):
        if mode not in MODES:
            raise ValueError("Unknown global scale-forecasting control.")
        self.mode = mode
        self.preset = initial["_preset"]
        strategy = "scale_fastcal" if self.preset == "balanced" else "memory_dynamic"
        self.parent = original_controller(strategy, initial, batch)
        stored = scale_models[self.preset]
        prototype = self.parent.amplitude
        self.coefficients = prototype.new_tensor([row["beta"] for row in stored])
        self.mean = prototype.new_tensor([row["context_delta_mean"] for row in stored])
        self.std = prototype.new_tensor([row["context_delta_std"] for row in stored])
        self.previous_context = None
        self.previous_raw_tail = None
        self.raw_tail_ema = None
        self.pending_context = None
        self.enabled = True
        self.last_log_factor = None

    def __getattr__(self, name):
        return getattr(self.parent, name)

    @torch.no_grad()
    def issue(self, lower, upper, features):
        if isinstance(features, tuple):
            risk, context = features
        else:
            # The unmodified common audit can replay the base controller with
            # the original risk tensor alone; a context policy requires context.
            risk, context = features, None
        if self.mode != "base" and context is None:
            raise ValueError("A scale forecast requires known current context.")
        if context is not None and (context.shape != (len(lower), 2, 8)
                                    or not torch.isfinite(context).all()):
            raise ValueError("Require finite[B,2,8]known context.")
        lo, hi = self.parent.issue(lower, upper, risk)
        self.pending_context = None if context is None else context.clone()
        log_factor = lower.new_zeros(len(lower), 2)
        # Two feedback windows avoid the cold-to-warm adapter-gate transition;
        # the meta-regression was fitted on warm-to-warm context changes.
        active = self.enabled and self.revealed_windows >= 2 and self.mode != "base"
        if active:
            reference = self.raw_tail_ema if self.preset == "balanced" else self.previous_raw_tail
            innovation = torch.zeros_like(self.previous_raw_tail)
            if self.mode in ("drift", "context"):
                innovation = self.coefficients[:, 0][None].expand_as(innovation)
                if self.mode == "context":
                    change = context - self.previous_context
                    standardized = ((change - self.mean[None]) / self.std[None]).clamp(-5, 5)
                    innovation = innovation + (standardized * self.coefficients[None, :, 1:]).sum(-1)
                innovation = innovation.clamp(-math.log(2), math.log(2))
            log_factor = (self.previous_raw_tail + innovation - reference).clamp(-math.log(2), math.log(2))
            if torch.count_nonzero(log_factor):
                mid, width, _, _, risk = self.parent.pending
                # Recompute the original radius in the original operation
                # order; do not infer it by subtracting rounded endpoints.
                radius = width * self.parent.conditional_shape(risk) * self.parent.shape
                radius = radius * self.parent.amplitude * self.parent.bias.exp()
                factor = log_factor[:, :, None, None, None].exp()
                lo, hi = mid - radius * factor, mid + radius * factor
                self.parent.pending = (mid, width, lo.clone(), hi.clone(), risk)
        self.last_log_factor = log_factor.clone()
        return lo, hi

    @torch.no_grad()
    def reveal(self, residual):
        if self.parent.pending is None or residual.shape != self.parent.pending[0].shape:
            raise RuntimeError("Scale-state update requires the complete issued20-frame target.")
        midpoint, width, _, _, _ = self.parent.pending
        ratio = ((residual - midpoint).abs() / width).flatten(2)
        raw_tail = torch.quantile(ratio, 0.9, dim=-1).clamp_min(1e-6).log()
        self.parent.reveal(residual)
        self.previous_raw_tail = raw_tail.clone()
        self.raw_tail_ema = (raw_tail.clone() if self.raw_tail_ema is None
                             else 0.5 * (self.raw_tail_ema + raw_tail))
        self.previous_context = self.pending_context
        self.pending_context = None
