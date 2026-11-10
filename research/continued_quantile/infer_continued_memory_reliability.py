"""Original four-call forecasts with a continued full-memory uncertainty head."""

import torch

from infer_fullblock_memory_risk import FullBlockMemoryReliabilityPredictor
from experiment_point_error_scale_alignment import OUT as TRAIN, NAMES


class ContinuedMemoryReliabilityPredictor(FullBlockMemoryReliabilityPredictor):
    def __init__(self, grid_geometry, objective="quantile", strategy="memory_dynamic", device="cuda"):
        if objective not in NAMES:
            raise ValueError(f"Choose one of the frozen objectives {NAMES}.")
        super().__init__(grid_geometry, memory_mode="full", strategy=strategy, device=device)
        self.objective = objective
        if objective != "identity":
            state = torch.load(TRAIN/(objective+".pt"),map_location=self.device,weights_only=False)
            self.head.load_state_dict(state["model"],strict=True)
        self.initial = torch.load(TRAIN/"calibration.pt",map_location=self.device,weights_only=False)[objective]["separable_integral"]
        self.reset()
