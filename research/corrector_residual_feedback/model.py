"""Shared input layout and the three-layer FP32 feedback Corrector."""
import torch
from torch import nn

ARMS = dict(A=(0,0,0),B=(1,0,0),C=(0,1,0),D=(0,0,1),
            E=(1,1,0),F=(1,0,1),G=(0,1,1),H=(1,1,1))
SLOTS = ("true","predicted","remaining")


def model(channels,device):
    return nn.Sequential(nn.Conv3d(channels,32,3,padding=1),nn.SiLU(),
                         nn.Conv3d(32,32,3,padding=1),nn.SiLU(),
                         nn.Conv3d(32,2,3,padding=1)).to(device)


def base(x,p,norm):
    x = (x-norm["mean"])/norm["std"]
    p = (p-norm["mean"])/norm["std"]
    return torch.cat((x,x[:,:,-1:].expand_as(x),p),dim=1)


def encode(current,feedback,mask,scale,capacity=False):
    enabled = []
    for i,on in enumerate(mask):
        slot = current[:,2*i:2*i+2] if capacity else (feedback[i]/scale).float()
        enabled.append(slot if on else torch.zeros_like(slot))
    return torch.cat((current,*enabled),dim=1)


def seeded(channels,initial,device):
    torch.manual_seed(42)
    nn = model(channels,device)
    state = nn.state_dict()
    state["0.weight"][:,:6].copy_(initial["0.weight"].to(device))
    for name,tensor in initial.items():
        if name!="0.weight":
            state[name].copy_(tensor.to(device))
    assert not torch.count_nonzero(state["4.weight"]) and not torch.count_nonzero(state["4.bias"])
    return nn


@torch.inference_mode()
def predict(model,z,scale):
    model.eval()
    return torch.cat([model(part)*scale for part in z.split(8)])
