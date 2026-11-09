"""Structured quantiles and matched independent-tail adapters."""

import math
import torch
from torch import nn
import torch.nn.functional as F


class StructuredQuantileHead(nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode
        self.history_projection = nn.Conv2d(40, 80, 1)
        self.feature_projection = nn.Conv2d(48, 160, 1)
        self.net = nn.Sequential(
            nn.Conv3d(28, 24, 3, padding=1), nn.GroupNorm(4, 24), nn.SiLU(),
            nn.Conv3d(24, 24, 3, padding=(1,2,2), dilation=(1,2,2)),
            nn.GroupNorm(4, 24), nn.SiLU(), nn.Conv3d(24, 4, 1))

    def bounds(self, base, last, history, scale):
        b, _, _, h, w = base.shape
        historical = self.history_projection(history.reshape(b,40,h,w)).reshape(b,4,20,h,w)
        features = self.feature_projection(last).reshape(b,8,20,h,w)
        if self.mode.endswith("base"):
            features = features * 0
        yy = torch.linspace(-1,1,h,device=base.device,dtype=base.dtype)
        xx = torch.linspace(-1,1,w,device=base.device,dtype=base.dtype)
        y,x = torch.meshgrid(yy,xx,indexing="ij")
        coordinates = torch.stack([x,y])[None,:,None].expand(b,2,20,h,w)
        value = self.net(torch.cat([base,historical,features,coordinates],1))
        mid, width = value[:,:2], F.softplus(value[:,2:])+1e-4
        return (mid-width)*scale, (mid+width)*scale

    def loss(self, base, last, history, residual, scale):
        lower,upper = self.bounds(base,last,history,scale)
        # A train-only fixed position/lead scale preserves each conditional
        # quantile target, while balancing optimization across spatial regions.
        e_lower,e_upper = (residual-lower)/scale,(residual-upper)/scale
        return (torch.maximum(.05*e_lower,-.95*e_lower)
                + torch.maximum(.95*e_upper,-.05*e_upper)).mean()


class FeedbackScaleAdapter(nn.Module):
    def __init__(self, use_memory):
        super().__init__()
        self.use_memory = use_memory
        self.net = nn.Sequential(
            nn.Conv3d(14, 16, 3, padding=1), nn.GroupNorm(4, 16), nn.SiLU(),
            nn.Conv3d(16, 16, 3, padding=(1, 2, 2), dilation=(1, 2, 2)),
            nn.GroupNorm(4, 16), nn.SiLU(), nn.Conv3d(16, 2, 1))
        # Both controls start at exactly the frozen raw interval.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def bounds(self, inputs, lower, upper):
        if not self.use_memory:
            inputs = inputs.clone()
            inputs[:, 10:12] = 0
        delta = math.log(2) * torch.tanh(self.net(inputs)) * inputs[:, 13:14]
        midpoint = 0.5 * (lower + upper)
        radius = 0.5 * (upper - lower) * delta.exp()
        return midpoint - radius, midpoint + radius

    def loss(self, inputs, lower, upper, residual):
        lo, hi = self.bounds(inputs, lower, upper)
        left, right = residual - lo, residual - hi
        return (torch.maximum(0.05 * left, -0.95 * left)
                + torch.maximum(0.95 * right, -0.05 * right)).mean()


class TailScaleRiskAdapter(FeedbackScaleAdapter):
    def __init__(self,asymmetric):
        super().__init__(use_memory=False)
        self.net[-1]=nn.Conv3d(16,4,1)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.asymmetric=asymmetric

    def bounds(self,inputs,lower,upper):
        value=inputs.clone()
        value[:,10:12]=0
        raw=self.net(value)
        left,right=raw[:,:2],raw[:,2:]
        if not self.asymmetric:
            left=right=0.5*(left+right)
        gate=value[:,13:14]
        left=(math.log(2)*left.tanh()*gate).exp()
        right=(math.log(2)*right.tanh()*gate).exp()
        anchor=0.5*(lower+upper)
        width=0.5*(upper-lower)
        return anchor-width*left,anchor+width*right


class CueTailAdapter(TailScaleRiskAdapter):
    def __init__(self):
        super().__init__(asymmetric=True)

    def bounds(self,inputs,lower,upper):
        # Cue construction controls the two reserved slots; do not erase them.
        value=inputs.clone()
        raw=self.net(value)
        left,right=raw[:,:2],raw[:,2:]
        gate=value[:,13:14]
        left=(math.log(2)*left.tanh()*gate).exp()
        right=(math.log(2)*right.tanh()*gate).exp()
        anchor=0.5*(lower+upper)
        width=0.5*(upper-lower)
        return anchor-width*left,anchor+width*right
