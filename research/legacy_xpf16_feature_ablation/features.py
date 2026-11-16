"""Information-source masking for the unchanged 16-channel legacy head.

P remains the common forecast anchor for every ablation. F is masked *after*
its learned projection, so the projection bias cannot leak into absent slots.
"""
import torch
from torch import nn
import torch.nn.functional as F
from selected16_quantile_head import BASE_INDICES

ARMS=('X','P','F','XP','XF','PF','XPF')

def mask_base(base,arm):
    if arm not in ARMS:
        raise ValueError(arm)
    result=base.clone()
    if 'X' not in arm:
        result[:,4:6]=0
    if 'P' not in arm:
        result[:,8:12]=0
    elif 'X' not in arm:
        # P[0]-X[-1] is mixed information; later temporal differences use P only.
        result[:,10:12,0]=0
    return result

class AblatedHead(nn.Module):
    """Wrap the exact old module; no new parameter, no new adapter."""
    def __init__(self,head,arm):
        super().__init__()
        self.head=head
        self.arm=arm
    def forward(self,base,last):
        base=mask_base(base,self.arm)
        batch,_,_,height,width=base.shape
        projected=self.head.feature_projection(last).reshape(batch,8,20,height,width)
        if 'F' not in self.arm:
            projected=torch.zeros_like(projected)
        yy=torch.linspace(-1,1,height,device=base.device,dtype=base.dtype)
        xx=torch.linspace(-1,1,width,device=base.device,dtype=base.dtype)
        y,x=torch.meshgrid(yy,xx,indexing='ij')
        coords=torch.stack((x,y))[None,:,None].expand(batch,2,20,height,width)
        value=self.head.net(torch.cat((base[:,list(BASE_INDICES)],projected,coords),dim=1))
        mid=value[:,:2]
        radius=F.softplus(value[:,2:])+1e-4
        return mid-radius,mid+radius
