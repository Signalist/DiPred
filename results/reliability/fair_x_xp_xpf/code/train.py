"""Three independently optimized fair heads, matched initialization and batches."""
import hashlib,json,math,os,sys,time
from pathlib import Path
import torch
from functorch import combine_state_for_ensemble,vmap
from model import FairQuantileHead,quantile_huber
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from clean_reliability_common import configure,digest,manifest
from experiment_reliability_transport_e12 import FIT_RECORDS,FIT_STARTS,normalization
from experiment_extended_train_error_adapters import LATE_RECORDS,LATE_STARTS
from temporal_mean_memory_control import training_data
from export_aligned_test_distribution import from_hdf
OUT=ROOT/'runs/fair_x_xp_xpf_20261113'
SEEDS=(42,123,2026)
ARMS=('X','XP','XPF')
MASKS=((0.,0.),(1.,0.),(1.,1.))
STEPS=2000


def atomic(value,path):
    tmp=path.with_suffix('.tmp');torch.save(value,tmp);os.replace(tmp,path)


def data(norm):
    # Cached P/F were issued by the frozen DiPred before future target access.
    base,f,x,memory,residual=training_data(norm)
    del memory,residual
    y=[]
    for starts,records in ((FIT_STARTS,FIT_RECORDS),(LATE_STARTS,LATE_RECORDS)):
        for start in starts:y.append(((from_hdf(records,start+20)-norm['mean'])/norm['std']).cpu())
    y=torch.cat(y).cuda();p=base[:,:2].contiguous();del base
    assert len(x)==len(y)==819 and f.shape[1]==48
    return x,p,f,y


def train_seed(seed,x,p,f,y):
    folder=OUT/f'seed_{seed}';folder.mkdir(parents=True,exist_ok=True)
    if all((folder/f'{name}.pt').exists() for name in ARMS):return
    torch.manual_seed(seed)
    template=FairQuantileHead().cuda()
    initial={k:v.detach().clone() for k,v in template.state_dict().items()}
    models=[]
    for _ in ARMS:
        m=FairQuantileHead().cuda();m.load_state_dict(initial);models.append(m)
    names=[n for n,_ in template.named_parameters()]
    functional,params,buffers=combine_state_for_ensemble(models)
    params=tuple(v.requires_grad_() for v in params)
    optimizer=torch.optim.AdamW(params,lr=3e-4,weight_decay=1e-5)
    rng=torch.Generator(device='cuda').manual_seed(seed)
    first=0;elapsed=0.;logs=[[] for _ in ARMS]
    progress=folder/'progress.pt'
    if progress.exists():
        state=torch.load(progress,map_location='cuda',weights_only=False)
        for value,old in zip(params,state['params']):value.data.copy_(old)
        optimizer.load_state_dict(state['optimizer']);rng.set_state(state['rng'].cpu())
        first=state['step'];elapsed=state['elapsed'];logs=state['logs']
    begin=time.monotonic()
    masks=torch.tensor(MASKS,device='cuda')[:,None,:].expand(3,4,2)
    def forward(weights,buf,mask,xb,pb,fb,tau):
        return functional(weights,buf,xb,pb,fb,mask,tau)
    for step in range(first+1,STEPS+1):
        ids=torch.randint(len(x),(4,),device='cuda',generator=rng)
        tau=torch.rand(4,device='cuda',generator=rng).clamp(.002,.998)
        optimizer.param_groups[0]['lr']=1e-6+(3e-4-1e-6)*.5*(1+math.cos(math.pi*(step-1)/STEPS))
        optimizer.zero_grad(set_to_none=True)
        predicted=vmap(forward,in_dims=(0,0,0,None,None,None,None))(params,buffers,masks,x[ids],p[ids],f[ids],tau)
        error=y[ids].unsqueeze(0)-predicted
        absolute=error.abs();k=.1
        huber=torch.where(absolute<=k,.5*error.square(),k*(absolute-.5*k))
        weight=(tau[None,:,None,None,None,None]-(error.detach()<0).to(error.dtype)).abs()
        losses=(weight*huber/k).flatten(1).mean(1)
        if not torch.isfinite(losses).all():raise RuntimeError('nonfinite loss')
        losses.sum().backward()
        norm=torch.zeros(3,device='cuda')
        for q in params:norm+=q.grad.flatten(1).square().sum(1)
        factor=(1/(norm.sqrt()+1e-6)).clamp(max=1)
        for q in params:q.grad.mul_(factor.reshape(3,*([1]*(q.ndim-1))))
        optimizer.step()
        if step==1 or step%100==0:
            for i in range(3):logs[i].append(dict(step=step,loss=float(losses[i])))
            print('FAIR_TRAIN',seed,step,[round(float(v),6) for v in losses],flush=True)
        if step%250==0:
            atomic(dict(step=step,params=[v.detach() for v in params],optimizer=optimizer.state_dict(),rng=rng.get_state(),elapsed=elapsed+time.monotonic()-begin,logs=logs),progress)
    elapsed+=time.monotonic()-begin
    for i,name in enumerate(ARMS):
        state={k:v[i].detach().cpu() for k,v in zip(names,params)}
        state.update({k:initial[k].detach().cpu() for k,_ in template.named_buffers()})
        atomic(dict(model=state,seed=seed,arm=name,mask=MASKS[i],steps=STEPS,training_examples=819,
                    loss=logs[i],group_wall_seconds=elapsed,parameters=sum(q.numel() for q in template.parameters()),
                    init_sha256=hashlib.sha256(b''.join(initial[k].detach().cpu().numpy().tobytes() for k in sorted(initial))).hexdigest()),folder/f'{name}.pt')
    progress.unlink(missing_ok=True)
    print('FAIR_SEED_COMPLETE',seed,'seconds',elapsed,flush=True)


def main():
    configure();OUT.mkdir(parents=True,exist_ok=True)
    groups=manifest()['files']
    for a,b in (('train','calibration'),('train','development'),('train','test'),('calibration','test'),('development','test'),('development','calibration')):
        assert {r['file'] for r in groups[a]}.isdisjoint({r['file'] for r in groups[b]})
        assert {r['group'] for r in groups[a]}.isdisjoint({r['group'] for r in groups[b]})
    x,p,f,y=data(normalization())
    for seed in SEEDS:train_seed(seed,x,p,f,y)
    paths=[OUT/f'seed_{seed}'/f'{arm}.pt' for seed in SEEDS for arm in ARMS]
    (OUT/'training_complete.json').write_text(json.dumps(dict(passed=True,seeds=SEEDS,arms=ARMS,steps=STEPS,
        weights={str(v.relative_to(OUT)):digest(v) for v in paths}),indent=2)+'\n')
    print('FAIR_TRAINING_COMPLETE',flush=True)

if __name__=='__main__':main()
