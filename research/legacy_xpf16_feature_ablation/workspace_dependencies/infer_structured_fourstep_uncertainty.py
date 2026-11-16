"""Native four-call DiPred plus one structured predictive-error head.

No saved forecast, future target, or 32-path ensemble is used for prediction.
The optional stateful wrapper requires full-window feedback before each issue.
"""

from pathlib import Path

import torch

from direct_error_uncertainty import forecast_features
from physical20_augmented_dynamics import AugmentedDynamics20
from reliability_risk_transport import base_features
from structured_quantile_uncertainty import StructuredQuantileHead
from conditional_online_risk import risk_features
from memory_quantile_online_risk import controller as memory_controller
from smoothed_scale_online_risk import controller as scale_controller


ROOT = Path(__file__).resolve().parent


class StructuredFourStepPredictor:
    def __init__(self, grid_geometry, device="cuda"):
        self.device = torch.device(device)
        self.geometry = torch.as_tensor(grid_geometry, dtype=torch.float32, device=self.device)
        if self.geometry.ndim == 1:
            self.geometry = self.geometry[None]
        if (self.geometry.ndim != 2 or self.geometry.shape[1] != 3
                or not torch.isfinite(self.geometry).all()
                or (self.geometry[:, :2] == 0).any() or (self.geometry[:, 2] <= 0).any()):
            raise ValueError("Require finite signed dx,dy and positive dt, shape [B,3] or [3].")
        predictor = ROOT / "runs/physical20_augmented_dynamics_20261109"
        risk = ROOT / "runs/structured_quantile_uncertainty_20261109"
        self.norm = torch.load(ROOT / "runs/clean_direct20_20261107/normalization.pt",
                               map_location=self.device, weights_only=False)
        scale = torch.load(predictor / "dynamics_scale.pt", map_location=self.device, weights_only=False)["scale"]
        self.model = AugmentedDynamics20(scale).to(self.device).eval().requires_grad_(False)
        self.model.load_state_dict(torch.load(predictor / "augmented_diffusion/model.pt",
                                              map_location=self.device, weights_only=False)["model"])
        self.head = StructuredQuantileHead("rich").to(self.device).eval().requires_grad_(False)
        self.head.load_state_dict(torch.load(risk / "rich.pt", map_location=self.device, weights_only=False)["model"])
        self.stats = torch.load(risk / "statistics.pt", map_location=self.device, weights_only=False)["statistics"]

    @torch.no_grad()
    def forecast(self, observed):
        observed = observed.to(device=self.device, dtype=torch.float32)
        if observed.ndim != 5 or observed.shape[1:] != (2, 20, 32, 64) or not torch.isfinite(observed).all():
            raise ValueError("History must be finite [B,2,20,32,64].")
        if len(self.geometry) not in (1, len(observed)):
            raise ValueError("Grid geometry must match the trajectory batch.")
        h = (observed - self.norm["mean"]) / self.norm["std"]
        point, features = forecast_features(self.model, h)
        physical = point * self.norm["std"] + self.norm["mean"]
        base = base_features(observed, physical, self.norm["mean"], self.norm["std"],
                             self.geometry.expand(len(observed), 3))
        # Match the feature precision on which this frozen risk head was fitted.
        last = features[:, -48:].half().float()
        base = ((base - self.stats["base_mean"]) / self.stats["base_std"]).clamp(-10, 10)
        last = ((last - self.stats["last_mean"]) / self.stats["last_std"]).clamp(-10, 10)
        lower, upper = self.head.bounds(base, last, h, torch.ones_like(point[:1]))
        return point, lower, upper


class StructuredReliabilityPredictor(StructuredFourStepPredictor):
    """One instance maintains a fixed ordered batch of trajectory identities."""

    def __init__(self, grid_geometry, strategy="memory_dynamic", device="cuda"):
        if strategy not in ("memory_dynamic", "memory_scale_fast", "quantile_dynamic", "scale_fastcal"):
            raise ValueError("Choose a validated four-step risk-calibration strategy.")
        super().__init__(grid_geometry, device)
        self.strategy = strategy
        self.initial = torch.load(ROOT / "runs/causal_shape_calibration_20261109/calibration.pt",
                                  map_location=self.device, weights_only=False)["rich"]["separable_integral"]
        self.reset()

    def reset(self):
        self.calibrator = None
        self.pending = None

    @torch.no_grad()
    def predict(self, observed):
        if self.pending is not None:
            raise RuntimeError("Reveal the complete previously issued 20-frame window first.")
        if self.calibrator is not None and len(observed) != len(self.calibrator.amplitude):
            raise RuntimeError("Reset before changing batch membership.")
        p, lower, upper = self.forecast(observed)
        h = (observed.to(self.device) - self.norm["mean"]) / self.norm["std"]
        features = risk_features(h, p, 0.5 * (upper - lower))
        if self.calibrator is None:
            factory = scale_controller if self.strategy == "scale_fastcal" else memory_controller
            self.calibrator = factory(self.strategy, self.initial, len(p))
        lo, hi = self.calibrator.issue(lower, upper, features)
        self.pending = p.clone()
        physical = p * self.norm["std"] + self.norm["mean"]
        return dict(prediction=physical, lower=physical + lo * self.norm["std"],
                    upper=physical + hi * self.norm["std"],
                    risk_half_width=0.5 * (hi - lo) * self.norm["std"],
                    previous_feedback_windows=self.calibrator.revealed_windows)

    @torch.no_grad()
    def reveal(self, truth):
        if self.pending is None:
            raise RuntimeError("No forecast is awaiting feedback.")
        truth = truth.to(device=self.device, dtype=torch.float32)
        if truth.shape != self.pending.shape or not torch.isfinite(truth).all():
            raise ValueError("Feedback must contain exactly all 20 finite future frames.")
        residual = (truth - self.norm["mean"]) / self.norm["std"] - self.pending
        self.calibrator.reveal(residual)
        self.pending = None
