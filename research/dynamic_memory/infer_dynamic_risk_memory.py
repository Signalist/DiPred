"""Native four-call PDE forecasts with learned risk-memory feature operators."""

import torch

from infer_fullblock_memory_risk import FullBlockMemoryReliabilityPredictor
from fullblock_memory_risk_head import FullBlockMemoryRiskHead
from dynamic_risk_memory import DynamicRiskMemoryHead
from experiment_dynamic_risk_memory import OUT as TRAIN, BEST, NAMES


class DynamicRiskMemoryPredictor(FullBlockMemoryReliabilityPredictor):
    """Use the original predict/reveal/reset guards and original-scale outputs."""

    def __init__(self, grid_geometry, objective="warp", strategy="memory_dynamic", device="cuda"):
        if objective not in NAMES:
            raise ValueError(f"Choose one of {NAMES}.")
        super().__init__(grid_geometry, memory_mode="full", strategy=strategy, device=device)
        scale = self.head.memory_scale.detach().clone()
        original = torch.load(BEST / "density_quantile.pt", map_location=self.device, weights_only=False)["model"]
        if objective == "frozen_best":
            head = FullBlockMemoryRiskHead("full", scale)
            head.load_state_dict(original)
        else:
            head = DynamicRiskMemoryHead(scale, original, objective)
            checkpoint = torch.load(TRAIN / (objective + ".pt"), map_location=self.device, weights_only=False)
            head.load_state_dict(checkpoint["model"])
        self.head = head.to(self.device).eval().requires_grad_(False)
        self.objective = objective
        self.initial = torch.load(TRAIN / "calibration.pt", map_location=self.device,
                                  weights_only=False)[objective]["separable_integral"]
        self.reset()
