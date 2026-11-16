"""Online conditional calibration targeting a log-error90%quantile."""

import math

import torch

from conditional_online_risk import ConditionalRiskCalibrator
from causal_shape_calibration import shape_from_scores
from experiment_reliability_structured_e3 import grouped_quantile


MODES=("ridge_off","ridge_dynamic","quantile_dynamic","quantile_fastcal")


def fit_quantile(features,target,start):
    """Twelve deterministic full-block projected subgradient steps.

    The intercept is reprofiled as a90%quantile; only slope coefficients receive
    the0.025ridge penalty. Labels are already revealed when called.
    """
    x=features.flatten(3).reshape(-1,features.shape[2],features[0,0,0].numel())
    y=target.flatten(2).reshape(-1,target[0,0].numel())
    beta=start.reshape(-1,start.shape[-1]).clone()
    for iteration in range(12):
        prediction=torch.bmm(beta[:,None],x).squeeze(1)
        intercept=torch.quantile(y-prediction,.9,dim=-1,keepdim=True)
        below=(y<prediction+intercept).to(x.dtype)
        gradient=torch.bmm(x,(below-.9)[:,:,None]).squeeze(-1)/x.shape[-1]+.025*beta
        rate=.5/math.sqrt(iteration+1)
        beta=(beta-rate*gradient).clamp(-1,1)
    return beta.reshape_as(start)


class QuantileRiskCalibrator(ConditionalRiskCalibrator):
    def __init__(self,mode,initial,batch):
        super().__init__("ridge_dynamic",initial,batch)
        self.quantile_mode=mode

    def reveal(self,residual):
        if self.pending is None or residual.shape!=self.pending[0].shape:
            raise RuntimeError("Feedback requires the complete issued20-frame block.")
        mid,width,lo,hi,features=self.pending
        error=(residual-mid).abs()
        target=((error+1e-4)/width/self.shape).log().clamp(-8,8)
        estimate=fit_quantile(features,target,self.beta)
        self.beta=.5*self.beta+.5*estimate
        adjusted=error/width/self.conditional_shape(features)
        fitted_shape=shape_from_scores(adjusted,"separable_integral")
        self.shape=(self.shape*fitted_shape).sqrt()
        self.shape=self.shape/self.shape.log().mean((2,3,4),keepdim=True).exp()
        self.amplitude=grouped_quantile(adjusted/self.shape,"global").clamp(1e-6,1e3)
        coverage=((residual>=lo)&(residual<=hi)).float().mean((2,3,4),keepdim=True)
        gain=2. if self.quantile_mode=="quantile_fastcal" else .5
        self.bias=(self.bias+gain*(.9-coverage)).clamp(-math.log(2),math.log(2))
        self.pending=None
        self.revealed_windows+=1


def controller(mode,initial,batch):
    if mode in ("ridge_off","ridge_dynamic"):
        return ConditionalRiskCalibrator(mode,initial,batch)
    return QuantileRiskCalibrator(mode,initial,batch)
