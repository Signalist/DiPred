"""Shared scenario data, frozen DiPred and four-step inference."""
import json,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'runs/five_pde_reliability_20261114'
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT.parents[1]))
from reliability.networks import DynamicsDiffusion20
from reliability.backbone import configure
from five_pde_reliability.prepare_data import basename


def records(name):return json.loads((OUT/f'inventory_{name}.json').read_text())


def metadata(name):return json.loads((OUT/f'normalization_{name}.json').read_text())


class Scenario:
    def __init__(self,name,device='cuda'):
        self.name=name;self.info=metadata(name);self.channels=len(self.info['channels'])
        mean=self.info['mean']+[0.]*(2-self.channels);std=self.info['std']+[1.]*(2-self.channels)
        increments=self.info['increment_rms_normalized']+[1.]*(2-self.channels)
        self.mean=torch.tensor(mean,device=device,dtype=torch.float32)[None,:,None,None,None]
        self.std=torch.tensor(std,device=device,dtype=torch.float32)[None,:,None,None,None]
        self.scale=torch.tensor(increments,device=device,dtype=torch.float32)[None,:,None,None,None].clamp_min(.01)
        self.mask=torch.tensor([1.]*self.channels+[0.]*(2-self.channels),device=device)[None,:,None,None,None]
        self.cache={}

    def block(self,record,start):
        key=record['id']
        if key not in self.cache:self.cache[key]=np.load(OUT/'fields'/self.name/basename(record),mmap_mode='r')
        a=self.cache[key][start:start+20]
        if len(a)!=20:raise ValueError('A full 20-frame window is required')
        return a

    def batch(self,rs,starts,future=False):
        array=np.stack([self.block(r,int(s)+(20 if future else 0)) for r,s in zip(rs,starts)])
        value=torch.from_numpy(array).to(self.mean.device).permute(0,2,1,3,4).contiguous()
        if self.channels==1:value=torch.cat((value,torch.zeros_like(value)),1)
        return ((value-self.mean)/self.std)*self.mask

    def network(self):return DynamicsDiffusion20(self.scale).to(self.mean.device)


def save_atomic(value,path):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp')
    torch.save(value,tmp);tmp.replace(path)


@torch.no_grad()
def predict(model,x,mask,with_features=False):
    """Four native steps; optional features are from the final actual up2 call."""
    features=[]
    hook=model.net.up2.register_forward_hook(lambda module,args,out:features.append(out.detach())) if with_features else None
    try:
        value=torch.zeros_like(x)
        for t in reversed(range(4)):
            alpha=model.abar[t];labels=torch.full((len(x),),t*33,device=x.device,dtype=torch.long)
            clean=model.net(value,labels,x)*mask
            noise=(value-alpha.sqrt()*clean)/(1-alpha).sqrt()
            previous=model.abar[t-1] if t else value.new_tensor(1.)
            value=(previous.sqrt()*clean+(1-previous).sqrt()*noise)*mask
    finally:
        if hook:hook.remove()
    if with_features:
        assert len(features)==4
        return value,features[-1]
    return value

@torch.no_grad()
def sample_predict(model, x, mask, count, first_seed=500000, chunk=4):
    """Draw actual four-step trajectories with stable per-path random seeds.

    Each path uses its own generator, so 8 and 32 paths are prefixes of the
    same 256-path ensemble. No future observations are used.
    """
    result = []
    for start in range(0, count, chunk):
        size = min(chunk, count - start)
        draws = []
        for index in range(start, start + size):
            generator = torch.Generator(device=x.device).manual_seed(first_seed + index)
            draws.append(torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator))
        value = torch.stack(draws).flatten(0, 1) * mask
        history = x.unsqueeze(0).expand(size, *x.shape).reshape(size * len(x), *x.shape[1:])
        for t in reversed(range(4)):
            alpha = model.abar[t]
            labels = torch.full((len(value),), t * 33, device=x.device, dtype=torch.long)
            clean = model.net(value, labels, history) * mask
            noise = (value - alpha.sqrt() * clean) / (1 - alpha).sqrt()
            previous = model.abar[t - 1] if t else value.new_tensor(1.)
            value = (previous.sqrt() * clean + (1 - previous).sqrt() * noise) * mask
        result.append(value.reshape(size, *x.shape))
    return torch.cat(result, dim=0)
