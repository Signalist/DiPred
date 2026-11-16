"""Causal patch retrieval of revealed forecast-risk patterns.

Retrieval changes interval dispersion only. It neither transports signed state
errors into the point forecast nor assumes errors follow passive advection.
"""

from collections import deque
import math

import torch
import torch.nn.functional as F

from smoothed_scale_online_risk import SmoothedScaleCalibrator,controller as baseline_controller
from experiment_reliability_structured_e3 import grouped_quantile


MODES=("scale_fastcal","memory_global","memory_position","memory_content","memory_misaligned")


def patch_keys(features):
    """Eight known physical/lead features, excluding spatial coordinates."""
    b,c,_,t,h,w=features.shape
    selected=features[:,:,:8].reshape(b*c,8,t,h,w)
    pooled=F.avg_pool3d(selected,(1,8,8),(1,8,8))
    return pooled.permute(0,2,3,4,1).reshape(b,c,640,8)


def patch_log_scores(ratio):
    b,c,t,h,w=ratio.shape
    assert (t,h,w)==(20,32,64)
    samples=ratio.reshape(b,c,20,4,8,8,8).permute(0,1,2,3,5,4,6).reshape(b,c,640,64)
    score=torch.quantile(samples,0.9,dim=-1).clamp(1e-3,1e3).log()
    return score-score.mean(-1,keepdim=True)


def retrieve_log_scores(query,memory,mode):
    """One trajectory/channel never retrieves another one's observed labels."""
    b,c,n,d=query.shape
    if mode=="memory_global":
        return query.new_zeros(b,c,n)
    values=torch.stack([item[1] for item in memory],dim=2)
    if mode=="memory_position":
        return values.mean(2)
    keys=torch.cat([item[0] for item in memory],dim=2)
    if mode=="memory_misaligned":
        # Preserve time lead and label marginal, deliberately break spatial
        # feature/score correspondence using a fixed reversed patch ordering.
        values=values.reshape(b,c,len(memory),20,32).flip(-1).reshape(b,c,-1)
    else:
        values=values.reshape(b,c,-1)
    q=query.reshape(b*c,n,d)
    k=keys.reshape(b*c,-1,d)
    distance=(q.square().sum(-1,keepdim=True)+k.square().sum(-1)[:,None]-2*torch.bmm(q,k.transpose(1,2))).clamp_min(0)/d
    nearest,indices=distance.topk(min(16,k.shape[1]),dim=-1,largest=False)
    flat_values=values.reshape(b*c,-1)
    retrieved=flat_values.gather(1,indices.reshape(b*c,-1)).reshape_as(nearest)
    prediction=(torch.softmax(-nearest,dim=-1)*retrieved).sum(-1)
    return prediction.reshape(b,c,n)


def redistribute_factor(log_scores):
    """Geometric-mean-one positive factor, bounded between0.5and2."""
    centered=0.5*(log_scores-log_scores.mean(-1,keepdim=True))
    scale=(centered.abs().amax(-1,keepdim=True)/math.log(2)).clamp_min(1)
    centered=centered/scale
    coarse=centered.reshape(*centered.shape[:2],20,4,8)
    return coarse.repeat_interleave(8,-2).repeat_interleave(8,-1).exp()


class ContentRiskMemory(SmoothedScaleCalibrator):
    def __init__(self,mode,initial,batch):
        super().__init__("scale_fastcal",initial,batch)
        self.memory_mode=mode
        self.memory=deque(maxlen=3)
        self.memory_pending=None

    def issue(self,lower,upper,features):
        lo,hi=super().issue(lower,upper,features)
        mid,width,_,_,stored=self.pending
        keys=patch_keys(stored)
        base_radius=width*self.conditional_shape(stored)*self.shape*self.amplitude*self.bias.exp()
        factor=torch.ones_like(width)
        if self.memory and self.memory_mode!="memory_global":
            predicted=retrieve_log_scores(keys,self.memory,self.memory_mode)
            factor=redistribute_factor(predicted)
            lo,hi=mid-base_radius*factor,mid+base_radius*factor
            self.pending=(mid,width,lo.clone(),hi.clone(),stored)
        self.memory_pending=(keys.clone(),base_radius.clone(),factor.clone())
        return lo,hi

    def reveal(self,residual):
        if self.pending is None or residual.shape!=self.pending[0].shape:
            raise RuntimeError("Risk retrieval accepts only the complete issued20-frame window.")
        mid,width,_,_,features=self.pending
        keys,base_radius,issued_factor=self.memory_pending
        scores=patch_log_scores((residual-mid).abs()/base_radius.clamp_min(1e-6))
        previous_amplitude=self.amplitude.clone()
        completed=self.revealed_windows
        super().reveal(residual)
        # Calibrate against the memory factor actually issued, never a factor
        # retrospectively recomputed with the just-revealed target in memory.
        adjusted=(residual-mid).abs()/width/self.conditional_shape(features)
        fitted=grouped_quantile(adjusted/self.shape/issued_factor,"global").clamp(1e-6,1e3)
        self.amplitude=(previous_amplitude*fitted).sqrt() if completed else fitted
        self.memory.append((keys,scores.clone()))
        self.memory_pending=None


def controller(mode,initial,batch):
    if mode=="scale_fastcal":
        return baseline_controller(mode,initial,batch)
    return ContentRiskMemory(mode,initial,batch)
