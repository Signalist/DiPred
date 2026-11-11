"""Causal low-dimensional risk adaptation from complete revealed windows.

This updates calibration coefficients, not the DiPred or risk-network weights.
It never corrects the point forecast. Empirical adaptation has no CP guarantee.
"""

import math

import torch
import torch.nn.functional as F

from causal_shape_calibration import shape_from_scores
from experiment_reliability_structured_e3 import grouped_quantile


FEATURE_NAMES=("log_width","log_gradient","log_departure","log_history_variability",
               "log_forecast_change","forecast_magnitude","lead","lead_squared","x","y","xy")
MODES=("ridge_off","ridge_static","ridge_dynamic")


def risk_features(history,prediction,raw_width):
    """Inputs use normalized physical fields, all available at forecast issue."""
    b,c,t,h,w=prediction.shape
    dx=F.pad(prediction.diff(dim=-1),(0,1,0,0))
    dy=F.pad(prediction.diff(dim=-2),(0,0,0,1))
    gradient=(dx.square()+dy.square()+1e-8).sqrt()
    departure=(prediction-history[:,:,-1:]).abs()
    variability=history.std(dim=2,keepdim=True).expand_as(prediction)
    change=torch.cat([prediction[:,:,:1]-history[:,:,-1:],prediction.diff(dim=2)],2).abs()
    lead=torch.arange(1,t+1,device=prediction.device,dtype=prediction.dtype)[None,None,:,None,None]/t
    lead=lead.expand_as(prediction)
    yy=torch.linspace(-1,1,h,device=prediction.device,dtype=prediction.dtype)
    xx=torch.linspace(-1,1,w,device=prediction.device,dtype=prediction.dtype)
    y,x=torch.meshgrid(yy,xx,indexing="ij")
    x=x[None,None,None].expand_as(prediction)
    y=y[None,None,None].expand_as(prediction)
    values=[value.clamp_min(1e-4).log() for value in (raw_width,gradient,departure,variability,change)]
    values.extend([prediction.abs(),lead,lead.square(),x,y,x*y])
    value=torch.stack(values,dim=2)
    mean=value.mean((3,4,5),keepdim=True)
    std=value.std((3,4,5),keepdim=True).clamp_min(.05)
    value=((value-mean)/std).clamp(-5,5)
    return value-value.mean((3,4,5),keepdim=True)


class ConditionalRiskCalibrator:
    def __init__(self,mode,initial,batch):
        self.mode=mode
        self.shape=initial["shape"].expand(batch,*initial["shape"].shape[1:]).clone()
        self.amplitude=initial["amplitude"].expand(batch,2,1,1,1).clone()
        self.bias=torch.zeros_like(self.amplitude)
        self.beta=self.amplitude.new_zeros(batch,2,len(FEATURE_NAMES))
        self.pending=None
        self.revealed_windows=0

    def conditional_shape(self,features):
        log_shape=(features*self.beta[:,:,:,None,None,None]).sum(2)
        return log_shape.clamp(-math.log(2),math.log(2)).exp()

    def issue(self,lower,upper,features):
        if self.pending is not None:
            raise RuntimeError("Reveal the whole preceding20-frame window first.")
        if lower.shape!=upper.shape or lower.shape[1:]!=(2,20,32,64):
            raise ValueError("Expected complete[B,2,20,32,64] error bounds.")
        if self.mode=="ridge_static":
            features=features.clone()
            features[:,:,1:6]=0
        mid,width=.5*(lower+upper),(.5*(upper-lower)).clamp_min(1e-6)
        radius=width*self.conditional_shape(features)*self.shape*self.amplitude*self.bias.exp()
        lo,hi=mid-radius,mid+radius
        self.pending=(mid.clone(),width.clone(),lo.clone(),hi.clone(),features.clone())
        return lo,hi

    def reveal(self,residual):
        if self.pending is None or residual.shape!=self.pending[0].shape:
            raise RuntimeError("Feedback requires the entire previously issued20-frame block.")
        mid,width,lo,hi,features=self.pending
        error=(residual-mid).abs()
        ratio=error/width
        if self.mode!="ridge_off":
            # Condition on the current known field; fit only AFTER truth reveal.
            target=((error+1e-4)/width/self.shape).log().clamp(-8,8).flatten(2)
            target=target-target.mean(-1,keepdim=True)
            x=features.flatten(3).reshape(-1,len(FEATURE_NAMES),features[0,0,0].numel())
            y=target.reshape(-1,target.shape[-1],1)
            count=x.shape[-1]
            gram=torch.bmm(x,x.transpose(1,2))/count
            cross=torch.bmm(x,y)/count
            identity=torch.eye(len(FEATURE_NAMES),device=x.device,dtype=x.dtype)[None]
            estimate=torch.linalg.solve(gram+.25*identity,cross).squeeze(-1).reshape_as(self.beta)
            self.beta=(.5*self.beta+.5*estimate.clamp(-1,1))
        conditional=self.conditional_shape(features)
        adjusted=ratio/conditional
        fit_shape=shape_from_scores(adjusted,"separable_integral")
        self.shape=(self.shape*fit_shape).sqrt()
        self.shape=self.shape/self.shape.log().mean((2,3,4),keepdim=True).exp()
        self.amplitude=grouped_quantile(adjusted/self.shape,"global").clamp(1e-6,1e3)
        coverage=((residual>=lo)&(residual<=hi)).float().mean((2,3,4),keepdim=True)
        self.bias=(self.bias+.5*(.9-coverage)).clamp(-math.log(2),math.log(2))
        self.pending=None
        self.revealed_windows+=1
