"""Frozen four-call point forecasts with the matched error-density risk heads."""

import torch

from infer_fullblock_memory_risk import FullBlockMemoryReliabilityPredictor
from experiment_distribution_error_risk import OUT as TRAIN,NAMES,PREVIOUS


class DistributionRiskPredictor(FullBlockMemoryReliabilityPredictor):
    """Retain predict/reveal/reset and original-scale complete20-frame outputs."""

    def __init__(self,grid_geometry,objective="density_quantile",strategy="memory_dynamic",device="cuda"):
        if objective not in NAMES:
            raise ValueError(f"Choose one of the frozen objectives {NAMES}.")
        super().__init__(grid_geometry,memory_mode="full",strategy=strategy,device=device)
        path = PREVIOUS/"quantile.pt" if objective=="density_reference" else TRAIN/(objective+".pt")
        self.head.load_state_dict(torch.load(path,map_location=self.device,weights_only=False)["model"],strict=True)
        self.objective = objective
        self.initial = torch.load(TRAIN/"calibration.pt",map_location=self.device,weights_only=False)[objective]["separable_integral"]
        self.reset()
