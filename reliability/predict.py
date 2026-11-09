"""Standalone frozen4-call forecast plus direct predictive-error intervals.

Use one instance for a fixed ordered trajectory batch. Feedback is accepted
only after predicting the complete next20frames; reset for new trajectories.
"""

import argparse
from pathlib import Path

import torch

from .backbone import configure
from .networks import DynamicsDiffusion20
from .heads import StructuredQuantileHead, TailScaleRiskAdapter, CueTailAdapter
from .features import base_features, forecast_features, adapter_inputs, cue_inputs, risk_features
from .calibration import SmoothedScaleCalibrator, MemoryQuantileCalibrator
from .context import ContextScaleForecaster, context_summary, MODES


CHECKPOINT = Path(__file__).resolve().parent / "checkpoints/current.pt"
PRESETS = ("low_regression", "best_mean_score", "context_mean_score", "context_balanced")


class ReliabilityPredictor:
    def __init__(self, grid_geometry, preset="low_regression", device="cuda", checkpoint=CHECKPOINT,
                 context_policy="context", context_checkpoint=None):
        if preset not in PRESETS:
            raise ValueError(f"Choose one of {PRESETS}.")
        if context_policy not in MODES:
            raise ValueError(f"Choose one of {MODES} for context_policy.")
        if not preset.startswith("context_") and context_policy != "context":
            raise ValueError("Policy ablations apply to context presets only.")
        configure()
        self.device = torch.device(device)
        self.preset = preset
        self.context_policy = context_policy
        self.uses_context = preset.startswith("context_")
        self.geometry = torch.as_tensor(grid_geometry, device=self.device, dtype=torch.float32)
        if self.geometry.ndim == 1:
            self.geometry = self.geometry[None]
        if (self.geometry.ndim != 2 or self.geometry.shape[1] != 3
                or not torch.isfinite(self.geometry).all()
                or (self.geometry[:, :2] == 0).any() or (self.geometry[:, 2] <= 0).any()):
            raise ValueError("Require signed dx,dy and positive dt, shape[B,3]or[3].")
        state = torch.load(checkpoint, map_location=self.device, weights_only=False)
        if state["format_version"] != 1:
            raise ValueError("Unsupported inference checkpoint format.")
        self.norm = state["normalization"]
        self.stats = state["statistics"]
        self.model = DynamicsDiffusion20(state["dynamics_scale"]).to(self.device).eval().requires_grad_(False)
        self.model.load_state_dict(state["predictor"], strict=True)
        if self.uses_context:
            path = (Path(context_checkpoint) if context_checkpoint is not None
                    else Path(checkpoint).with_name("context.pt"))
            extra = torch.load(path, map_location=self.device, weights_only=False)
            if extra["format_version"] != 1:
                raise ValueError("Unsupported context checkpoint format.")
            # Only the risk head/adapters change. Keep the shared predictor,
            # train-only normalization, and frozen feature statistics intact.
            state = dict(state, risk_head=extra["risk_head"],
                         adapters=extra["adapters"], calibration=extra["calibration"])
            self.scale_models = extra["scale_models"]
        self.head = StructuredQuantileHead("rich").to(self.device).eval().requires_grad_(False)
        self.head.load_state_dict(state["risk_head"], strict=True)
        adapter = (TailScaleRiskAdapter(True) if preset in ("low_regression", "context_balanced")
                   else CueTailAdapter())
        self.adapter = adapter.to(self.device).eval().requires_grad_(False)
        self.adapter.load_state_dict(state["adapters"][preset], strict=True)
        self.initial = state["calibration"][preset]
        self.reset()

    def reset(self):
        self.calibrator = None
        self.pending = None
        self.last_call_labels = None

    @torch.no_grad()
    def predict(self, observed):
        if self.pending is not None:
            raise RuntimeError("Reveal the complete previously issued20-frame window first.")
        observed = torch.as_tensor(observed, device=self.device, dtype=torch.float32)
        if observed.ndim != 5 or observed.shape[1:] != (2, 20, 32, 64) or not torch.isfinite(observed).all():
            raise ValueError("History must be finite[B,2,20,32,64].")
        if len(self.geometry) not in (1, len(observed)):
            raise ValueError("Geometry must match the ordered trajectory batch.")
        if self.calibrator is not None and len(self.calibrator.amplitude) != len(observed):
            raise RuntimeError("Reset before changing trajectory batch membership.")
        h = (observed - self.norm["mean"]) / self.norm["std"]
        labels = []
        hook = self.model.net.register_forward_pre_hook(lambda module, args: labels.append(int(args[1][0])))
        try:
            point, features = forecast_features(self.model, h)
        finally:
            hook.remove()
        if labels != [99, 66, 33, 0]:
            raise RuntimeError("Native four-call inference invariant violated.")
        self.last_call_labels = labels
        physical = point * self.norm["std"] + self.norm["mean"]
        base = base_features(observed, physical, self.norm["mean"], self.norm["std"],
                             self.geometry.expand(len(point), 3))
        last = features[:, -48:].half().float()
        base = ((base - self.stats["base_mean"]) / self.stats["base_std"]).clamp(-10, 10)
        last = ((last - self.stats["last_mean"]) / self.stats["last_std"]).clamp(-10, 10)
        lower, upper = self.head.bounds(base, last, h, torch.ones_like(point[:1]))
        available = self.calibrator is not None and self.calibrator.revealed_windows > 0
        if self.preset == "low_regression":
            inputs = adapter_inputs(h, point, lower, upper, None)
            inputs[:, 13:14] = float(available)
        else:
            inputs = cue_inputs(h, point, lower, upper, None, "history_change", available)
        lower, upper = self.adapter.bounds(inputs, lower, upper)
        known = risk_features(h, point, 0.5 * (upper - lower))
        if self.uses_context:
            known = (known, context_summary(h, point, lower, upper))
        if self.calibrator is None:
            if self.uses_context:
                name = "balanced" if self.preset == "context_balanced" else "mean_score"
                self.calibrator = ContextScaleForecaster(self.context_policy,
                    dict(self.initial, _preset=name), len(point), self.scale_models)
            elif self.preset == "low_regression":
                self.calibrator = SmoothedScaleCalibrator("scale_fastcal", self.initial, len(point))
            else:
                self.calibrator = MemoryQuantileCalibrator("memory_dynamic", self.initial, len(point))
        lo, hi = self.calibrator.issue(lower, upper, known)
        self.pending = point.clone()
        result = dict(prediction=physical, lower=physical + lo * self.norm["std"],
                    upper=physical + hi * self.norm["std"],
                    risk_half_width=0.5 * (hi - lo) * self.norm["std"],
                    previous_feedback_windows=self.calibrator.revealed_windows)
        if self.uses_context:
            result["global_log_scale_correction"] = self.calibrator.last_log_factor.clone()
        return result

    @torch.no_grad()
    def reveal(self, truth):
        if self.pending is None:
            raise RuntimeError("No complete forecast is awaiting feedback.")
        truth = torch.as_tensor(truth, device=self.device, dtype=torch.float32)
        if truth.shape != self.pending.shape or not torch.isfinite(truth).all():
            raise ValueError("Feedback must contain exactly all20finite future frames.")
        residual = (truth - self.norm["mean"]) / self.norm["std"] - self.pending
        self.calibrator.reveal(residual)
        self.pending = None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Original-scale tensor[B,2,20,32,64].")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--geometry", type=float, nargs=3, required=True, metavar=("DX", "DY", "DT"))
    parser.add_argument("--preset", choices=PRESETS, default="low_regression")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--context-policy", choices=MODES, default="context")
    parser.add_argument("--context-checkpoint", type=Path)
    args = parser.parse_args()
    predictor = ReliabilityPredictor(args.geometry, args.preset, args.device, args.checkpoint,
                                     args.context_policy, args.context_checkpoint)
    observed = torch.load(args.input, map_location="cpu", weights_only=False)
    result = predictor.predict(observed)
    saved = {key: value.cpu() if torch.is_tensor(value) else value for key, value in result.items()}
    saved["preset"] = args.preset
    saved["native_call_labels"] = predictor.last_call_labels
    torch.save(saved, args.output)


if __name__ == "__main__":
    main()
