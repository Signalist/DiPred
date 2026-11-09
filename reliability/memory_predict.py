"""Frozen four-call DiPred with directly learned revealed-error uncertainty.

One instance follows a fixed ordered batch of contiguous 20-frame windows.
Only reveal() can introduce observed forecast errors into its state.
"""

import argparse
from pathlib import Path

import torch

from .backbone import configure
from .networks import DynamicsDiffusion20
from .memory_heads import FullBlockMemoryRiskHead, TemporalMeanMemoryRiskHead
from .features import base_features, forecast_features, risk_features
from .calibration import SmoothedScaleCalibrator, MemoryQuantileCalibrator


SHARED_CHECKPOINT = Path(__file__).resolve().parent / "checkpoints/current.pt"
MEMORY_MODES = ("none", "last", "mean", "full")
STRATEGIES = ("memory_dynamic", "scale_fastcal")


class MemoryReliabilityPredictor:
    """A fixed point forecast plus asymmetric, causally updated risk intervals."""

    def __init__(self, grid_geometry, memory_mode="full", strategy="memory_dynamic",
                 device="cuda", checkpoint=SHARED_CHECKPOINT, memory_checkpoint=None):
        if memory_mode not in MEMORY_MODES or strategy not in STRATEGIES:
            raise ValueError(f"Choose memory in {MEMORY_MODES} and strategy in {STRATEGIES}.")
        configure()
        self.device = torch.device(device)
        self.memory_mode = memory_mode
        self.strategy = strategy
        self.geometry = torch.as_tensor(grid_geometry, device=self.device, dtype=torch.float32)
        if self.geometry.ndim == 1:
            self.geometry = self.geometry[None]
        if (self.geometry.ndim != 2 or self.geometry.shape[1] != 3
                or not torch.isfinite(self.geometry).all()
                or (self.geometry[:, :2] == 0).any() or (self.geometry[:, 2] <= 0).any()):
            raise ValueError("Require signed dx,dy and positive dt, shape [B,3] or [3].")

        shared = torch.load(checkpoint, map_location=self.device, weights_only=False)
        path = Path(memory_checkpoint) if memory_checkpoint else Path(checkpoint).with_name("memory.pt")
        memory = torch.load(path, map_location=self.device, weights_only=False)
        if shared["format_version"] != 1 or memory["format_version"] != 1:
            raise ValueError("Unsupported inference checkpoint format.")
        self.norm = shared["normalization"]
        self.stats = shared["statistics"]
        self.model = DynamicsDiffusion20(shared["dynamics_scale"]).to(self.device)
        self.model.load_state_dict(shared["predictor"], strict=True)
        self.model.eval().requires_grad_(False)
        if memory_mode == "mean":
            head = TemporalMeanMemoryRiskHead(memory["memory_scale"])
        else:
            head = FullBlockMemoryRiskHead(memory_mode, memory["memory_scale"])
        self.head = head.to(self.device).eval().requires_grad_(False)
        self.head.load_state_dict(memory["heads"][memory_mode], strict=True)
        self.initial = memory["calibration"][memory_mode]
        self.reset()

    def reset(self):
        """Forget feedback when changing trajectory identity or batch order."""
        self.calibrator = None
        self.pending = None
        self.previous_error = None
        self.previous_truth = None
        self.last_call_labels = None

    @torch.no_grad()
    def predict(self, observed):
        if self.pending is not None:
            raise RuntimeError("Reveal all 20 frames of the previously issued target first.")
        observed = torch.as_tensor(observed, device=self.device, dtype=torch.float32)
        if observed.ndim != 5 or observed.shape[1:] != (2, 20, 32, 64) or not torch.isfinite(observed).all():
            raise ValueError("History must be finite original-scale [B,2,20,32,64].")
        if len(self.geometry) not in (1, len(observed)):
            raise ValueError("Geometry must match the ordered trajectory batch.")
        if self.previous_truth is not None and not torch.equal(observed, self.previous_truth):
            raise RuntimeError("History must equal the preceding revealed block; reset for a new stream.")

        history = (observed - self.norm["mean"]) / self.norm["std"]
        calls = []
        hook = self.model.net.register_forward_pre_hook(lambda module, args: calls.append(int(args[1][0])))
        try:
            point, features = forecast_features(self.model, history)
        finally:
            hook.remove()
        if calls != [99, 66, 33, 0]:
            raise RuntimeError("Native four-call inference invariant violated.")
        self.last_call_labels = calls
        physical = point * self.norm["std"] + self.norm["mean"]
        base = base_features(observed, physical, self.norm["mean"], self.norm["std"],
                             self.geometry.expand(len(point), 3))
        # Training cached the last 48 denoising features in float16. Preserve
        # that rounding before the frozen train-only feature normalization.
        last = features[:, -48:].half().float()
        base = ((base - self.stats["base_mean"]) / self.stats["base_std"]).clamp(-10, 10)
        last = ((last - self.stats["last_mean"]) / self.stats["last_std"]).clamp(-10, 10)
        available = self.previous_error is not None
        memory = self.previous_error if available else torch.zeros_like(history)
        lower, upper = self.head.bounds(base, last, history, memory, available)
        if self.calibrator is None:
            factory = SmoothedScaleCalibrator if self.strategy == "scale_fastcal" else MemoryQuantileCalibrator
            self.calibrator = factory(self.strategy, self.initial, len(point))
        lo, hi = self.calibrator.issue(lower, upper, risk_features(history, point, 0.5 * (upper - lower)))
        # Store the actual issued point prediction; never regenerate it after
        # seeing truth. The point forecast itself receives no correction.
        self.pending = point.clone()
        return dict(
            prediction=physical,
            lower=physical + lo * self.norm["std"],
            upper=physical + hi * self.norm["std"],
            risk_half_width=0.5 * (hi - lo) * self.norm["std"],
            memory_available=available,
            previous_feedback_windows=self.calibrator.revealed_windows,
        )

    @torch.no_grad()
    def reveal(self, truth):
        if self.pending is None:
            raise RuntimeError("No complete forecast is awaiting feedback.")
        truth = torch.as_tensor(truth, device=self.device, dtype=torch.float32)
        if truth.shape != self.pending.shape or not torch.isfinite(truth).all():
            raise ValueError("Feedback must contain exactly all 20 finite future frames.")
        residual = (truth - self.norm["mean"]) / self.norm["std"] - self.pending
        self.calibrator.reveal(residual)
        self.previous_error = residual.clone()
        self.previous_truth = truth.clone()
        self.pending = None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Original-scale tensor [B,2,20,32,64].")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--geometry", type=float, nargs=3, required=True, metavar=("DX", "DY", "DT"))
    parser.add_argument("--memory", choices=MEMORY_MODES, default="full")
    parser.add_argument("--strategy", choices=STRATEGIES, default="memory_dynamic")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint", type=Path, default=SHARED_CHECKPOINT)
    parser.add_argument("--memory-checkpoint", type=Path)
    args = parser.parse_args()
    predictor = MemoryReliabilityPredictor(args.geometry, args.memory, args.strategy, args.device,
                                           args.checkpoint, args.memory_checkpoint)
    result = predictor.predict(torch.load(args.input, map_location="cpu", weights_only=False))
    result = {key: value.cpu() if torch.is_tensor(value) else value for key, value in result.items()}
    result.update(memory_mode=args.memory, strategy=args.strategy, native_call_labels=predictor.last_call_labels)
    torch.save(result, args.output)


if __name__ == "__main__":
    main()
