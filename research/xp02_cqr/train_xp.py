"""Independently train 249 symmetric XP heads with fixed seed and budget."""
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
from functorch import combine_state_for_ensemble, vmap
from xp_head import XPHead

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from clean_reliability_common import configure, digest, dump
from experiment_reliability_transport_e12 import normalization
from temporal_mean_memory_control import training_data

OUT=ROOT/'runs/xp02_cqr_20261112'
STEPS=2000


def save(value, path):
    tmp=path.with_suffix('.tmp')
    torch.save(value,tmp)
    os.replace(tmp,path)


def train_group(ids, base, history, residual, initial):
    folder=OUT/'progress';folder.mkdir(parents=True,exist_ok=True)
    progress=folder/('_'.join(map(str,ids))+'.pt')
    models=[]
    for _ in ids:
        model=XPHead().cuda().train()
        model.load_state_dict(initial)
        models.append(model)
    functional,params,buffers=combine_state_for_ensemble(models)
    params=tuple(p.requires_grad_() for p in params)
    optimizer=torch.optim.AdamW(params,lr=3e-4,weight_decay=1e-5)
    sampler=torch.Generator(device='cuda').manual_seed(42)
    begin_step=0;elapsed=0.;logs=[[] for _ in ids]
    if progress.exists():
        previous=torch.load(progress,map_location='cuda',weights_only=False)
        assert previous['ids']==ids
        for p,v in zip(params,previous['parameters']):p.data.copy_(v)
        optimizer.load_state_dict(previous['optimizer'])
        sampler.set_state(previous['sampler'].cpu())
        begin_step=previous['step'];elapsed=previous['elapsed'];logs=previous['logs']
    alpha=torch.tensor(ids,device='cuda',dtype=base.dtype).reshape(-1,1,1,1,1,1)/500
    start=time.monotonic()
    for step in range(begin_step+1,STEPS+1):
        indices=torch.randint(len(base),(4,),device='cuda',generator=sampler)
        torch.rand(4,device='cuda',generator=sampler)
        optimizer.param_groups[0]['lr']=1e-6+(3e-4-1e-6)*.5*(1+math.cos(math.pi*(step-1)/STEPS))
        optimizer.zero_grad(set_to_none=True)
        lower,upper=vmap(lambda p,b:functional(p,b,base[indices],history[indices]),in_dims=(0,0))(params,buffers)
        left=residual[indices].unsqueeze(0)-lower;right=residual[indices].unsqueeze(0)-upper
        loss=.5*(torch.maximum(alpha*left,(alpha-1)*left)+torch.maximum((1-alpha)*right,-alpha*right))
        losses=loss.flatten(1).mean(1)
        if not torch.isfinite(losses).all():raise RuntimeError('Nonfinite loss')
        losses.sum().backward()
        norm=torch.zeros(len(ids),device='cuda')
        for p in params:norm+=p.grad.flatten(1).square().sum(1)
        factor=(1/(norm.sqrt()+1e-6)).clamp(max=1)
        if not torch.isfinite(factor).all():raise RuntimeError('Nonfinite gradient')
        for p in params:p.grad.mul_(factor.reshape(len(ids),*([1]*(p.ndim-1))))
        optimizer.step()
        if step==1 or step%100==0:
            for i in range(len(ids)):logs[i].append(dict(step=step,loss=float(losses[i])))
        if step%250==0:
            save(dict(ids=ids,step=step,elapsed=elapsed+time.monotonic()-start,parameters=[p.detach() for p in params],
                      optimizer=optimizer.state_dict(),sampler=sampler.get_state(),logs=logs),progress)
    elapsed+=time.monotonic()-start
    names=[name for name,_ in models[0].named_parameters()]
    for i,k in enumerate(ids):
        folder=OUT/'models'/f'XP-P{k:03d}';folder.mkdir(parents=True,exist_ok=True)
        path=folder/'model.pt'
        state={name:p[i].detach().cpu().clone() for name,p in zip(names,params)}
        save(dict(k=k,alpha=k/500,model=state,seed=42,steps=STEPS,elapsed=elapsed,loss=logs[i],
                  parameters=sum(v.numel() for v in state.values()),train_pairs=len(base)),path)
        print('TRAINED_XP',k,'group_seconds',elapsed,flush=True)
    progress.unlink(missing_ok=True)


def main():
    configure();OUT.mkdir(exist_ok=True,parents=True)
    torch.manual_seed(42)
    template=XPHead().cuda()
    initial=template.state_dict()
    data=training_data(normalization())
    base,history,residual=data[0],data[2],data[4]
    assert len(base)==819
    ids=[k for k in range(1,250) if not (OUT/'models'/f'XP-P{k:03d}'/'model.pt').exists()]
    for i in range(0,len(ids),4):train_group(ids[i:i+4],base,history,residual,initial)
    paths=[OUT/'models'/f'XP-P{k:03d}'/'model.pt' for k in range(1,250)]
    assert all(p.exists() for p in paths)
    dump(OUT/'training_complete.json',dict(passed=True,heads=249,seed=42,steps=STEPS,
         parameters=sum(p.numel() for p in template.parameters()),features='20-frame X and P only; no F or coordinates',
         weights={p.parent.name:digest(p) for p in paths}))
    print('XP_TRAINING_COMPLETE',flush=True)

if __name__=='__main__':main()
