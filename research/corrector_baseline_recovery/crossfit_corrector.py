"""Physical residual corrector using only observed history and issued forecast."""
import torch
from torch import nn

class BlockResidualCorrector(nn.Module):
    def __init__(self):
        super().__init__()
        self.history=nn.Sequential(nn.Conv2d(40,16,3,padding=1),nn.SiLU(),nn.Conv2d(16,16,3,padding=1))
        self.residual=nn.Sequential(nn.Conv3d(23,16,3,padding=1),nn.SiLU(),nn.Conv3d(16,16,3,padding=1),nn.SiLU(),nn.Conv3d(16,2,3,padding=1))
        nn.init.zeros_(self.residual[-1].weight);nn.init.zeros_(self.residual[-1].bias)
    def forward(self,h,p):
        if h.shape!=p.shape or h.shape[1:3]!=(2,20):raise ValueError('Expected equal B,2,20,H,W blocks')
        b,c,t,ny,nx=h.shape
        context=self.history(h.reshape(b,40,ny,nx))[:,:,None].expand(-1,-1,20,-1,-1)
        last=h[:,:,-1:].expand_as(p)
        trend=(h[:,:,-1:]-h[:,:,-2:-1]).expand_as(p)
        lead=torch.linspace(.05,1,20,device=h.device,dtype=h.dtype).reshape(1,1,20,1,1).expand(b,1,20,ny,nx)
        return p+self.residual(torch.cat([context,p,last,trend,lead],1))


def fold_indices(groups,heldout):
    heldout=set(heldout)
    train=[i for i,g in enumerate(groups) if g not in heldout]
    test=[i for i,g in enumerate(groups) if g in heldout]
    if not train or not test:raise ValueError('Empty fold')
    assert not set(groups[i] for i in train)&set(groups[i] for i in test)
    return train,test
