"""Current four-step inference cues with independent lower and upper scales."""

import math

from tail_scale_risk_adapter import TailScaleRiskAdapter


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
