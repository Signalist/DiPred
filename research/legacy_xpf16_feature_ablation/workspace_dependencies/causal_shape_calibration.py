"""Separate interval-shape memory from its marginal coverage amplitude.

All fitting in reveal uses an already issued COMPLETE forecast window. None of
these empirical updates establishes finite-sample or conditional guarantees.
"""

import math

import torch

from experiment_reliability_structured_e3 import grouped_quantile


MODES=("global_smooth","global_lagged","lead_lagged",
       "separable_lagged","separable_integral")


def shape_from_scores(ratio,mode,pooled=False):
    global_q=grouped_quantile(ratio,"global",pooled=pooled).clamp(1e-6,1e3)
    if mode.startswith("global"):
        return torch.ones_like(global_q)
    lead=grouped_quantile(ratio,"lead",pooled=pooled).clamp(1e-6,1e3)/global_q
    if mode.startswith("separable"):
        patch=grouped_quantile(ratio,"patch",pooled=pooled).clamp(1e-6,1e3)/global_q
        lead=lead*patch.sqrt()
    return lead/lead.log().mean((2,3,4),keepdim=True).exp()


def initial_state(ratio,mode):
    shape=shape_from_scores(ratio,mode,pooled=True)
    amplitude=grouped_quantile(ratio/shape,"global",pooled=True).clamp(1e-6,1e3)
    return dict(shape=shape,amplitude=amplitude)


class CausalShapeCalibrator:
    def __init__(self,mode,initial,batch):
        self.mode=mode
        self.shape=initial["shape"].expand(batch,*initial["shape"].shape[1:]).clone()
        self.amplitude=initial["amplitude"].expand(batch,2,1,1,1).clone()
        self.bias=torch.zeros_like(self.amplitude)
        self.pending=None
        self.revealed_windows=0

    def issue(self,lower,upper):
        if self.pending is not None:
            raise RuntimeError("Reveal the complete preceding20-frame window first.")
        assert lower.shape==upper.shape and lower.shape[1:]==(2,20,32,64)
        mid,width=.5*(lower+upper),(.5*(upper-lower)).clamp_min(1e-6)
        radius=width*self.shape*self.amplitude*self.bias.exp()
        lo,hi=mid-radius,mid+radius
        self.pending=(mid.clone(),width.clone(),lo.clone(),hi.clone())
        return lo,hi

    def reveal(self,residual):
        if self.pending is None or residual.shape!=self.pending[0].shape:
            raise RuntimeError("Feedback requires the entire previously issued20-frame window.")
        mid,width,lo,hi=self.pending
        ratio=(residual-mid).abs()/width
        fit_shape=shape_from_scores(ratio,self.mode)
        self.shape=(self.shape*fit_shape).sqrt()
        self.shape=self.shape/self.shape.log().mean((2,3,4),keepdim=True).exp()
        # Estimate amplitude AFTER updating shape, instead of geometrically
        # averaging their product. This avoids mixing amplitude and shape drift.
        amplitude=grouped_quantile(ratio/self.shape,"global").clamp(1e-6,1e3)
        self.amplitude=(self.amplitude*amplitude).sqrt() if self.mode=="global_smooth" else amplitude
        if self.mode=="separable_integral":
            coverage=((residual>=lo)&(residual<=hi)).float().mean((2,3,4),keepdim=True)
            self.bias=(self.bias+.5*(.9-coverage)).clamp(-math.log(2),math.log(2))
        self.pending=None
        self.revealed_windows+=1
