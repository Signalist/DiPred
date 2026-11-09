"""Native four-call forecast with causal revealed-error uncertainty memory."""

import torch

from clean_reliability_common import configure
from infer_structured_fourstep_uncertainty import StructuredFourStepPredictor
from experiment_fullblock_memory_risk import OUT as TRAIN
from temporal_mean_memory_control import OUT as MEAN, TemporalMeanMemoryRiskHead
from fullblock_memory_risk_head import FullBlockMemoryRiskHead
from direct_error_uncertainty import forecast_features
from reliability_risk_transport import base_features
from experiment_temporal_error_uncertainty import normalized
from conditional_online_risk import risk_features
from experiment_continue_rich_uncertainty import controller


class FullBlockMemoryReliabilityPredictor(StructuredFourStepPredictor):
    """One instance follows a fixed ordered batch of contiguous20-frame blocks."""

    def __init__(self, grid_geometry, memory_mode="full", strategy="memory_dynamic", device="cuda"):
        if memory_mode not in ("none","last","mean","full"):
            raise ValueError("Choose none,last,mean or full memory.")
        if strategy not in ("memory_dynamic","scale_fastcal"):
            raise ValueError("Choose an evaluated online calibration strategy.")
        configure()
        super().__init__(grid_geometry,device)
        self.memory_mode=memory_mode
        self.strategy=strategy
        scale=torch.load(TRAIN/"statistics.pt",map_location=self.device,weights_only=False)["memory_scale"]
        head=TemporalMeanMemoryRiskHead(scale) if memory_mode=="mean" else FullBlockMemoryRiskHead(memory_mode,scale)
        folder=MEAN if memory_mode=="mean" else TRAIN
        self.head=head.to(self.device).eval().requires_grad_(False)
        self.head.load_state_dict(torch.load(folder/(memory_mode+".pt"),map_location=self.device,weights_only=False)["model"])
        self.initial=torch.load(folder/"calibration.pt",map_location=self.device,weights_only=False)[memory_mode]["separable_integral"]
        self.reset()

    def reset(self):
        self.calibrator=None
        self.pending=None
        self.previous_error=None
        self.previous_truth=None
        self.last_call_labels=None

    @torch.no_grad()
    def predict(self, observed):
        if self.pending is not None:
            raise RuntimeError("Reveal all20frames of the previously issued target first.")
        observed=torch.as_tensor(observed,device=self.device,dtype=torch.float32)
        if observed.ndim!=5 or observed.shape[1:]!=(2,20,32,64) or not torch.isfinite(observed).all():
            raise ValueError("History must be finite original-scale[B,2,20,32,64].")
        if len(self.geometry) not in (1,len(observed)):
            raise ValueError("Geometry must match the ordered trajectory batch.")
        if self.previous_truth is not None and not torch.equal(observed,self.previous_truth):
            raise RuntimeError("History must equal the preceding revealed block; reset for a new stream or batch order.")
        h=(observed-self.norm["mean"])/self.norm["std"]
        calls=[]
        hook=self.model.net.register_forward_pre_hook(lambda module,args:calls.append(int(args[1][0])))
        try:
            point,features=forecast_features(self.model,h)
        finally:
            hook.remove()
        assert calls==[99,66,33,0]
        self.last_call_labels=calls
        physical=point*self.norm["std"]+self.norm["mean"]
        base=base_features(observed,physical,self.norm["mean"],self.norm["std"],self.geometry.expand(len(point),3))
        base,last=normalized(base,features[:,-48:].half().float(),self.stats)
        available=self.previous_error is not None
        memory=torch.zeros_like(h) if not available else self.previous_error
        lower,upper=self.head.bounds(base,last,h,memory,available)
        if self.calibrator is None:
            self.calibrator=controller(self.strategy,self.initial,len(point))
        lo,hi=self.calibrator.issue(lower,upper,risk_features(h,point,.5*(upper-lower)))
        self.pending=point.clone()
        return dict(prediction=physical,lower=physical+lo*self.norm["std"],upper=physical+hi*self.norm["std"],
            risk_half_width=.5*(hi-lo)*self.norm["std"],memory_available=available,
            previous_feedback_windows=self.calibrator.revealed_windows)

    @torch.no_grad()
    def reveal(self, truth):
        if self.pending is None:
            raise RuntimeError("No complete forecast is awaiting feedback.")
        truth=torch.as_tensor(truth,device=self.device,dtype=torch.float32)
        if truth.shape!=self.pending.shape or not torch.isfinite(truth).all():
            raise ValueError("Feedback must contain exactly all20finite future frames.")
        residual=(truth-self.norm["mean"])/self.norm["std"]-self.pending
        self.calibrator.reveal(residual)
        self.previous_error=residual.clone()
        self.previous_truth=truth.clone()
        self.pending=None
