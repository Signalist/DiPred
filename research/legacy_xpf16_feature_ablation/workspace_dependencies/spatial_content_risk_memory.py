"""Remove lead-level components from retrieved spatial-risk corrections."""

import math

from content_matched_risk_memory import ContentRiskMemory,controller as original_controller


MODES=("scale_fastcal","memory_content","memory_position_spatial","memory_content_spatial","memory_misaligned_spatial")


class SpatialContentRiskMemory(ContentRiskMemory):
    def __init__(self,mode,initial,batch):
        super().__init__(mode[:-8],initial,batch)

    def issue(self,lower,upper,features):
        lo,hi=super().issue(lower,upper,features)
        if not self.memory:
            return lo,hi
        mid,width,_,_,stored=self.pending
        keys,base_radius,full_factor=self.memory_pending
        # Same neighbors and scores; remove only each lead's uniform spatial
        # log-scale. This preserves within-frame relative corrections.
        log_factor=full_factor.log()
        log_factor=log_factor-log_factor.mean((3,4),keepdim=True)
        scale=(log_factor.abs().amax((2,3,4),keepdim=True)/math.log(2)).clamp_min(1)
        factor=(log_factor/scale).exp()
        lo,hi=mid-base_radius*factor,mid+base_radius*factor
        self.pending=(mid,width,lo.clone(),hi.clone(),stored)
        self.memory_pending=(keys,base_radius,factor.clone())
        return lo,hi


def controller(mode,initial,batch):
    if mode.endswith("_spatial"):
        return SpatialContentRiskMemory(mode,initial,batch)
    return original_controller(mode,initial,batch)
