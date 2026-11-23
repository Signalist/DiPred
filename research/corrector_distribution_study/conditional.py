"""Deterministic state-conditioned positive scaling with matched basis sizes."""

import torch

from .calibration import static_basis

METHODS = {"recent": (0,), "state": (0,1), "state_memory": (0,2)}


@torch.no_grad()
def release_features(release, scale):
    recent, _, _, _, spread, memory = release
    def magnitude(field):
        return (field/scale).square().mean((2,3,4)).sqrt()
    return torch.stack((magnitude(recent.diff(dim=2)), magnitude(spread), magnitude(memory)),dim=-1)


def features(data, scale):
    blocks = []
    for start in range(0,len(data.cases),8):
        idx = list(range(start,min(start+8,len(data.cases))))
        blocks.append(release_features(data.release(idx),scale))
    return torch.cat(blocks)


def knots(features):
    lo,hi = torch.quantile(features, features.new_tensor([.1,.9]), dim=0).unbind()
    assert (hi>lo).all()
    return dict(low=lo.tolist(),high=hi.tolist())


def basis(name, features, limits):
    features = features.double()
    lo = features.new_tensor(limits["low"])
    hi = features.new_tensor(limits["high"])
    normalized = ((features-lo)/(hi-lo)).clamp(0,1)
    f = normalized[:,:,list(METHODS[name])]
    a = torch.stack((1-f[:,:,0], f[:,:,0]),-1)
    if f.shape[-1] == 2:
        b = torch.stack((1-f[:,:,1], f[:,:,1]),-1)
        a = (a[...,None]*b[:,:,None,:]).flatten(2)
    lead = static_basis("S2",len(f),f.device,dtype=f.dtype)
    return (lead[...,None] * a[:,:,None,None,:]).flatten(3)
