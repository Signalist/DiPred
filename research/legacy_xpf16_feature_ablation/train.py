"""Train 49 independent legacy dual-quantile heads per missing feature set."""
import json,math,os,sys,time
from pathlib import Path
import torch
from functorch import combine_state_for_ensemble,vmap
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT/'xpf16_fullpercent_study'))
from clean_reliability_common import configure,digest
from experiment_selected16_quantile import new_head
from experiment_reliability_transport_e12 import normalization
from temporal_mean_memory_control import training_data
from features import ARMS,AblatedHead
from train49 import configs,pinball
OUT=ROOT/'runs/legacy_xpf16_feature_ablation_20261116'
STEPS=2000

def atomic(value,path):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');torch.save(value,tmp);os.replace(tmp,path)

def group(arm,cfgs,data,initial):
    tag='_'.join(f"{c['k']:02d}" for c in cfgs);checkpoint=OUT/'progress'/arm/(tag+'.pt')
    models=[]
    for cfg in cfgs:
        h=new_head('compact16',None);h.load_state_dict(initial,strict=True)
        models.append(AblatedHead(h,arm).cuda().train())
    fun,params,buffers=combine_state_for_ensemble(models);params=tuple(p.requires_grad_() for p in params)
    opt=torch.optim.AdamW(params,lr=3e-4,weight_decay=1e-5)
    sampler=torch.Generator(device='cuda').manual_seed(42);step0=0;elapsed=0.;logs=[[] for _ in cfgs]
    if checkpoint.exists():
        old=torch.load(checkpoint,map_location='cuda',weights_only=False)
        assert old['arm']==arm and old['configs']==cfgs
        for p,q in zip(params,old['params']):p.data.copy_(q)
        opt.load_state_dict(old['optimizer']);sampler.set_state(old['sampler'].cpu())
        step0=old['step'];elapsed=old['elapsed'];logs=old['logs']
    base,last,_,_,residual=data;n=len(cfgs)
    alpha=torch.tensor([c['lower'] for c in cfgs],device='cuda').reshape(n,1,1,1,1,1)
    def call(p,b,base,last):return fun(p,b,base,last)
    begin=time.monotonic()
    for step in range(step0+1,STEPS+1):
        ix=torch.randint(819,(4,),device='cuda',generator=sampler)
        torch.rand(4,device='cuda',generator=sampler)
        opt.param_groups[0]['lr']=1e-6+(3e-4-1e-6)*.5*(1+math.cos(math.pi*(step-1)/STEPS))
        opt.zero_grad(set_to_none=True)
        lo,hi=vmap(call,in_dims=(0,0,None,None))(params,buffers,base[ix],last[ix])
        error=residual[ix].unsqueeze(0)
        losses=.5*(pinball(error-lo,alpha)+pinball(error-hi,1-alpha)).flatten(1).mean(1)
        if not torch.isfinite(losses).all() or not (lo<hi).all():raise FloatingPointError('invalid bounds/loss')
        losses.sum().backward();norm=torch.zeros(n,device='cuda')
        for p in params:
            if p.grad is not None:norm+=p.grad.flatten(1).square().sum(1)
        factor=(1/(norm.sqrt()+1e-6)).clamp(max=1)
        for p in params:
            if p.grad is not None:p.grad.mul_(factor.reshape(n,*([1]*(p.ndim-1))))
        opt.step()
        if step==1 or step%100==0:
            for i in range(n):logs[i].append(dict(step=step,loss=float(losses[i]),seconds=elapsed+time.monotonic()-begin))
        if step%250==0:
            atomic(dict(arm=arm,configs=cfgs,step=step,params=[p.detach() for p in params],optimizer=opt.state_dict(),sampler=sampler.get_state(),logs=logs,elapsed=elapsed+time.monotonic()-begin),checkpoint)
    names=[name for name,_ in models[0].head.named_parameters()];duration=elapsed+time.monotonic()-begin
    for i,cfg in enumerate(cfgs):
        state={name:p[i].detach().cpu().clone() for name,p in zip(names,params)}
        assert sum(p.numel() for p in state.values())==34004
        path=OUT/'models'/arm/(f"Q{cfg['k']:02d}.pt")
        atomic(dict(model=state,arm=arm,config=cfg,steps=STEPS,parameters=34004,training_pairs=819,loss=logs[i],group_seconds=duration,group_size=n,initial_sha256=digest(ROOT/'runs/selected16_quantile_20261110/compact16_initial.pt')),path)
    checkpoint.unlink(missing_ok=True)
    print('GROUP_COMPLETE',arm,tag,round(duration,2),flush=True)

def main():
    configure();OUT.mkdir(parents=True,exist_ok=True)
    data=training_data(normalization());assert len(data[0])==819
    initial=torch.load(ROOT/'runs/selected16_quantile_20261110/compact16_initial.pt',map_location='cuda',weights_only=False)['model']
    for arm in ARMS[:-1]:
        # Complete any interrupted old-sized group before regrouping pending heads.
        for checkpoint in sorted((OUT/'progress'/arm).glob('*.pt')):
            saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
            group(arm,saved['configs'],data,initial)
        pending=[c for c in configs() if not (OUT/'models'/arm/f"Q{c['k']:02d}.pt").exists()]
        for i in range(0,len(pending),16):group(arm,pending[i:i+16],data,initial)
    paths={str(p.relative_to(OUT)):digest(p) for p in sorted((OUT/'models').rglob('*.pt'))}
    assert len(paths)==6*49
    (OUT/'training_complete.json').write_text(json.dumps(dict(complete=True,weights=paths),indent=2)+'\n')
    print('TRAINING_COMPLETE',len(paths),flush=True)
if __name__=='__main__':main()
