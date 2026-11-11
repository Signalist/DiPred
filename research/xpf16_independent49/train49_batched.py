"""Batch independent XPF16 parameter sets on GPU with resumable optimizer.

This computes 4 independent losses and gradients in parallel with functorch;
no parameters, gradient clipping, or Adam moments are shared between heads.
The minibatch sequence, initialization, loss and steps match train49.py.
"""
import math
import os
import time
from pathlib import Path

import torch
from torch import nn
from functorch import combine_state_for_ensemble, vmap
from clean_reliability_common import configure, digest, dump
from experiment_selected16_quantile import new_head
from experiment_reliability_transport_e12 import normalization
from temporal_mean_memory_control import training_data
from train49 import OUT, STEPS, SEED, configs, pinball


class Wrapper(nn.Module):
    def __init__(self, initial):
        super().__init__()
        self.head=new_head('compact16',None)
        self.head.load_state_dict(initial,strict=True)
    def forward(self,base,last):
        return self.head.bounds(base,last,None,None,False)


def stack(models):
    fmodel, params, buffers=combine_state_for_ensemble(models)
    return fmodel,tuple(p.requires_grad_() for p in params),buffers


def atomicsave(value, path):
    temporary=path.with_suffix('.tmp')
    torch.save(value,temporary)
    os.replace(temporary,path)


def train_group(cfgs,data,initial):
    n=len(cfgs)
    tag='_'.join(f"{c['k']:02d}" for c in cfgs)
    progress=OUT/'batch_progress'
    progress.mkdir(exist_ok=True)
    statepath=progress/(tag+'.pt')
    models=[Wrapper(initial).cuda().train() for _ in cfgs]
    fmodel,params,buffers=stack(models)
    def forward(ps,bs,base,last):
        return fmodel(ps,bs,base,last)
    opt=torch.optim.AdamW(params,lr=3e-4,weight_decay=1e-5)
    sampler=torch.Generator(device='cuda').manual_seed(SEED)
    logs=[[] for _ in cfgs]
    start=0;elapsed=0.
    if statepath.exists():
        checkpoint=torch.load(statepath,map_location='cuda',weights_only=False)
        assert checkpoint['configs']==cfgs
        for p,s in zip(params,checkpoint['params']):
            p.data.copy_(s)
        opt.load_state_dict(checkpoint['optimizer'])
        sampler.set_state(checkpoint['sampler'].cpu())
        logs=checkpoint['logs'];start=checkpoint['step'];elapsed=checkpoint['elapsed']
    alpha=torch.tensor([c['lower'] for c in cfgs],device='cuda').reshape(n,1,1,1,1,1)
    base,last,history,memory,residual=data
    begin=time.monotonic()
    for step in range(start+1,STEPS+1):
        ix=torch.randint(819,(4,),device='cuda',generator=sampler)
        torch.rand(4,device='cuda',generator=sampler) # original availability mask RNG
        opt.param_groups[0]['lr']=1e-6+(3e-4-1e-6)*.5*(1+math.cos(math.pi*(step-1)/STEPS))
        opt.zero_grad(set_to_none=True)
        lo,hi=vmap(forward,in_dims=(0,0,None,None))(params,buffers,base[ix],last[ix])
        assert torch.isfinite(lo).all() and torch.isfinite(hi).all() and (lo<hi).all()
        e1=residual[ix].unsqueeze(0)-lo;e2=residual[ix].unsqueeze(0)-hi
        losses=.5*(pinball(e1,alpha)+pinball(e2,1-alpha)).flatten(1).mean(1)
        assert torch.isfinite(losses).all()
        losses.sum().backward()
        # Clip each independent model by its own global norm, never a group norm.
        norm=torch.zeros(n,device='cuda')
        for p in params:
            norm+=p.grad.flatten(1).square().sum(1)
        factor=(1/(norm.sqrt()+1e-6)).clamp(max=1)
        assert torch.isfinite(factor).all()
        for p in params:
            p.grad.mul_(factor.reshape(n,*([1]*(p.ndim-1))))
        opt.step()
        if step==1 or step%100==0 or step==STEPS:
            for i in range(n):
                logs[i].append(dict(step=step,loss=float(losses[i]),seconds=elapsed+time.monotonic()-begin))
        if step%250==0:
            atomicsave(dict(configs=cfgs,step=step,elapsed=elapsed+time.monotonic()-begin,
                            params=[p.detach() for p in params],optimizer=opt.state_dict(),
                            sampler=sampler.get_state(),logs=logs),statepath)
    duration=elapsed+time.monotonic()-begin
    names=[name for name,_ in models[0].head.named_parameters()]
    for i,cfg in enumerate(cfgs):
        folder=OUT/'models'/cfg['id'];folder.mkdir(parents=True,exist_ok=True)
        final=folder/'model.pt'
        assert not final.exists()
        state={name:p[i].detach().cpu().clone() for name,p in zip(names,params)}
        atomicsave(dict(config=cfg,model=state,elapsed=duration,loss=logs[i],
                        parameters=sum(v.numel() for v in state.values()),training_pairs=819,
                        execution='parallel independent parameter group',group_size=n),final)
        dump(folder/'training.json',dict(config=cfg,sha256=digest(final),loss=logs[i],elapsed=duration,
             parameters=sum(v.numel() for v in state.values()), execution='batched independent parameter group',group_size=n))
        print('TRAINED_BATCHED',cfg['id'],'group_seconds',duration,flush=True)
    statepath.unlink(missing_ok=True)


def main():
    configure()
    initial=torch.load(OUT.parent/'selected16_quantile_20261110/compact16_initial.pt',map_location='cuda',weights_only=False)['model']
    data=training_data(normalization())
    cfgs=configs()
    pending=[c for c in cfgs if not (OUT/'models'/c['id']/'model.pt').exists()]
    for start in range(0,len(pending),4):
        group=pending[start:start+4]
        train_group(group,data,initial)
    assert all((OUT/'models'/c['id']/'model.pt').exists() for c in cfgs)
    dump(OUT/'training_complete.json',dict(passed=True,models=49,
         all_checkpoints={c['id']:digest(OUT/'models'/c['id']/'model.pt') for c in cfgs}))
    print('ALL_49_TRAINED',flush=True)

if __name__=='__main__':main()
