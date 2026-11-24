"""Three-layer residual CNNs and fixed spacetime input organizations."""
import torch
from torch import nn

from research.corrector_consistency.model import spatial, temporal

CHANNELS = dict(S0=4,S1=4,S2=7,S3=6,S4=4)
LOSSES = ("direction","magnitude","velocity","acceleration","match")


class Corrector(nn.Module):
    def __init__(self,mode,width=8):
        super().__init__()
        self.net = nn.Sequential(nn.Conv3d(CHANNELS[mode],width,3,padding=1),nn.SiLU(),
                                 nn.Conv3d(width,width,3,padding=1),nn.SiLU(),
                                 nn.Conv3d(width,2,3,padding=1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self,inputs):
        return self.net(inputs.contiguous(memory_format=torch.channels_last_3d))


def encode(x,p,mode):
    last = x[:,:,-1:].expand_as(x)
    if mode=="S0":
        slots = (x,p)
    elif mode=="S1":
        slots = (last,p)
    elif mode=="S2":
        trend = (x[:,:,-1:]-x[:,:,-2:-1]).expand_as(x)
        lead = (torch.arange(1,21,device=x.device,dtype=x.dtype)/20).view(1,1,20,1,1)
        slots = (last,trend,p,lead.expand_as(p[:,:1]))
    elif mode=="S3":
        slots = (x,last,p)
    else:
        assert mode=="S4"
        slots = (x.flip(2),p)
    return torch.cat(slots,dim=1)


def network(config,device):
    return Corrector(config["input"],config["width"]).to(device=device,memory_format=torch.channels_last_3d)


def inputs(split,norm,mode):
    x,p = ((t-norm["mean"])/norm["std"] for t in (split.x,split.p))
    if mode!="S2":
        return encode(x,p,mode)
    # Compute the explicit physical increment before normalizing the trend.
    z = encode(x,p,mode)
    z[:,2:4] = ((split.x[:,:,-1:]-split.x[:,:,-2:-1])/norm["std"]).expand_as(x)
    return z


def components(r,y,names):
    result = {"mse":(r-y).square().mean()}
    if set(names)&{"direction","magnitude"}:
        estimates = spatial(r,y)
        result.update({k:estimates[k].mean() for k in names if k in estimates})
    if set(names)&{"velocity","acceleration","match"}:
        first,second = temporal(r)
        values = {"velocity":first.square().mean(),"acceleration":second.square().mean(),
                  "match":(first-y.diff(dim=2)).square().mean()}
        result.update({k:values[k] for k in names if k in values})
    return result
