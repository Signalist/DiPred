"""Same conditional quantile network, with fixed X/P/F input slots."""
import torch
from torch import nn
import torch.nn.functional as F


class FairQuantileHead(nn.Module):
    def __init__(self,hidden=32):
        super().__init__()
        # The same projection exists in every arm; absent inputs are zeroed
        # after projection so its bias cannot masquerade as an observed F.
        self.f_projection=nn.Conv2d(48,160,1)
        self.encoder=nn.Sequential(nn.Conv3d(12,hidden,3,padding=1),nn.GroupNorm(4,hidden),nn.SiLU(),
            nn.Conv3d(hidden,hidden,3,padding=(1,2,2),dilation=(1,2,2)),nn.GroupNorm(4,hidden),nn.SiLU(),
            nn.Conv3d(hidden,hidden,3,padding=1),nn.GroupNorm(4,hidden),nn.SiLU())
        # Six increasing bases give a flexible, noncrossing shared quantile
        # function; all three experimental arms use the identical decoder.
        self.decoder=nn.Sequential(nn.Conv3d(hidden,hidden,1),nn.SiLU(),nn.Conv3d(hidden,14,1))
        self.register_buffer('knots',torch.tensor([.2,.4,.6,.8]))

    def encode(self,x,p,f,mask):
        if x.ndim!=5 or x.shape[1:3]!=(2,20) or p.shape!=x.shape or f.shape!=(len(x),48,*x.shape[-2:]):
            raise ValueError('Require complete X, P, and real F with aligned dimensions')
        projected=self.f_projection(f).reshape(len(x),8,20,*x.shape[-2:])
        # mask is [B,2]; X is always available, P and F are fixed zero slots if absent.
        if mask.shape!=(len(x),2):raise ValueError('Expected per-sample P/F availability')
        slots=torch.cat((x,p*mask[:,0,None,None,None,None],projected*mask[:,1,None,None,None,None]),dim=1)
        return self.encoder(slots)

    def decode(self,features,x,tau):
        if tau.shape!=(len(x),) or not bool(((tau>0)&(tau<1)).all()):
            raise ValueError('tau must be in (0,1), one value per trajectory')
        out=self.decoder(features)
        center=out[:,:2]
        weights=F.softplus(out[:,2:].reshape(len(x),2,6,20,*x.shape[-2:]))/6
        bases=self.quantile_bases(tau)
        residual=center+(weights*bases[:,None,:,None,None,None]).sum(dim=2)
        return x[:,:,-1:].expand_as(residual)+residual

    def quantile_bases(self,tau):
        t=tau[:,None]
        hinge=(F.softplus(8*(t-self.knots[None]))-F.softplus(8*(.5-self.knots[None])))/8
        return torch.cat((t-.5,torch.logit(t)/4,hinge),dim=1)

    def quantiles(self,x,p,f,mask,levels):
        """All levels from one encoder/decoder pass; returns [Q,B,2,20,H,W]."""
        out=self.decoder(self.encode(x,p,f,mask))
        center=out[:,:2]+x[:,:,-1:]
        weights=F.softplus(out[:,2:].reshape(len(x),2,6,20,*x.shape[-2:]))/6
        bases=self.quantile_bases(levels.to(device=x.device,dtype=x.dtype))
        return center.unsqueeze(0)+torch.einsum('bckthw,qk->qbcthw',weights,bases)

    def forward(self,x,p,f,mask,tau):
        return self.decode(self.encode(x,p,f,mask),x,tau)


def quantile_huber(q,y,tau,kappa=.1):
    error=y-q;absolute=error.abs()
    huber=torch.where(absolute<=kappa,.5*error.square(),kappa*(absolute-.5*kappa))
    weight=(tau[:,None,None,None,None]-(error.detach()<0).to(error.dtype)).abs()
    return (weight*huber/kappa).mean()
