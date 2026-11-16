"""Matched symmetric and independently scaled tails around a frozen anchor."""

import math

import torch
from torch import nn

from learned_feedback_risk import FeedbackScaleAdapter,adapter_inputs


def tail_inputs(history,point,lower,upper,available):
    value=adapter_inputs(history,point,lower,upper,None)
    value[:,13:14]=float(available)
    return value


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
