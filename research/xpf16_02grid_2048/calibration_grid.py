"""Vectorized but independent causal two-tail online states for a level grid."""
import torch


def per_level_quantiles(values, levels):
    """Compute each head's distinct quantile along final dim, linear interpolation."""
    if values.shape[0]!=len(levels):
        raise ValueError('One probability level per head is required.')
    ordered=values.sort(dim=-1).values
    n=ordered.shape[-1]
    index=levels.to(values.device,dtype=values.dtype)*(n-1)
    lower=index.floor().long();upper=(lower+1).clamp(max=n-1)
    shape=(len(levels),)+(1,)*(values.ndim-2)+(1,)
    low=lower.reshape(shape).expand(*values.shape[:-1],1)
    high=upper.reshape(shape).expand(*values.shape[:-1],1)
    lo=torch.gather(ordered,-1,low).squeeze(-1)
    hi=torch.gather(ordered,-1,high).squeeze(-1)
    frac=(index-lower).reshape((len(levels),)+(1,)*(values.ndim-2))
    return lo+(hi-lo)*frac


class GridCalibrator:
    """Separate offsets per head/trajectory/channel, no cross-head fitting."""
    def __init__(self,alpha,initial,batch,gain=.5):
        if (alpha<=0).any() or (alpha>=.5).any():
            raise ValueError('Interval tail probabilities must lie in (0,.5).')
        self.alpha=alpha
        self.gain=gain
        self.offsets=initial[:,None,:,:,None,None,None].expand(-1,batch,-1,-1,-1,-1,-1).clone()
        self.pending=None
        self.projected=torch.zeros(len(alpha),device=alpha.device,dtype=torch.long)
        self.total=0
        self.updates=0

    def issue(self,lower,upper):
        if self.pending is not None or lower.shape!=upper.shape or lower.ndim!=6 or lower.shape[3]!=20:
            raise RuntimeError('Require a new complete 20-frame interval block.')
        lo=lower+self.offsets[:,:,0]
        hi=upper+self.offsets[:,:,1]
        bad=lo>hi
        self.projected+=bad.flatten(1).sum(1)
        self.total+=bad[0].numel()
        mid=.5*(lo+hi)
        self.pending=(lower.detach(),upper.detach())
        return torch.where(bad,mid,lo),torch.where(bad,mid,hi)

    def reveal(self,residual):
        if self.pending is None or residual.shape!=self.pending[0].shape[1:]:
            raise RuntimeError('Feedback must be exactly the revealed full window.')
        lo,hi=self.pending
        left=(residual.unsqueeze(0)-lo)[...,::4,::4].flatten(3)
        right=(residual.unsqueeze(0)-hi)[...,::4,::4].flatten(3)
        estimate=torch.stack((per_level_quantiles(left,self.alpha),per_level_quantiles(right,1-self.alpha)),2)
        estimate=estimate[...,None,None,None]
        self.offsets=(1-self.gain)*self.offsets+self.gain*estimate
        self.pending=None
        self.updates+=1


class MedianCalibrator:
    """Independent per-trajectory/channel shift for a separately predicted median."""
    def __init__(self,initial,batch,gain=.5):
        self.offsets=initial[None,:,None,None,None].expand(batch,-1,-1,-1,-1).clone()
        self.gain=gain
        self.pending=None
        self.updates=0

    def issue(self,median):
        if self.pending is not None or median.ndim!=5 or median.shape[2]!=20:
            raise RuntimeError('Require a new complete 20-frame median block.')
        self.pending=median.detach()
        return median+self.offsets

    def reveal(self,residual):
        if self.pending is None or residual.shape!=self.pending.shape:
            raise RuntimeError('Feedback must be exactly the revealed full window.')
        values=(residual-self.pending)[...,::4,::4].flatten(2)
        estimate=torch.quantile(values,.5,dim=2)[...,None,None,None]
        self.offsets=(1-self.gain)*self.offsets+self.gain*estimate
        self.pending=None
        self.updates+=1
