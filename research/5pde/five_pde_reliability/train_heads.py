"""Independent matched X / XP / XPF quantile heads for each scenario and seed."""
import argparse,hashlib,json,math,time,sys
import numpy as np
import torch
from functorch import combine_state_for_ensemble,vmap
from common import ROOT,OUT,Scenario,configure,save_atomic
sys.path.insert(0,str(ROOT/'fair_x_xp_xpf'))
from model import FairQuantileHead
ARMS=('X','XP','XPF');MASKS=((0.,0.),(1.,0.),(1.,1.))


def train_seed(name,seed,steps):
    scenario=Scenario(name);folder=OUT/'risk_data'/name;meta=json.loads((folder/'complete.json').read_text())
    data={k:np.load(folder/(k+'.npy'),mmap_mode='r') for k in ('x','p','y','f')}
    fmean=torch.tensor(meta['f_mean'],device='cuda')[None,:,None,None]
    fstd=torch.tensor(meta['f_std'],device='cuda')[None,:,None,None]
    output=OUT/'heads'/name/f'seed_{seed}';output.mkdir(parents=True,exist_ok=True)
    if all((output/(arm+'.pt')).exists() for arm in ARMS):return
    torch.manual_seed(seed);template=FairQuantileHead().cuda()
    initial={k:v.detach().clone() for k,v in template.state_dict().items()}
    models=[]
    for _ in ARMS:
        model=FairQuantileHead().cuda();model.load_state_dict(initial);models.append(model)
    names=[n for n,_ in template.named_parameters()]
    functional,params,buffers=combine_state_for_ensemble(models);params=tuple(p.requires_grad_() for p in params)
    opt=torch.optim.AdamW(params,lr=3e-4,weight_decay=1e-5)
    rng=np.random.default_rng(seed);generator=torch.Generator(device='cuda').manual_seed(seed)
    masks=torch.tensor(MASKS,device='cuda')[:,None,:].expand(3,4,2)
    log=[[] for _ in ARMS];first=0;elapsed=0.;progress=output/'progress.pt'
    if progress.exists():
        old=torch.load(progress,map_location='cuda',weights_only=False)
        for q,v in zip(params,old['params']):q.data.copy_(v)
        opt.load_state_dict(old['optimizer']);rng.bit_generator.state=old['rng'];generator.set_state(old['generator'].cpu())
        first=old['step'];elapsed=old['elapsed'];log=old['log']
    begin=time.monotonic()
    def apply(w,b,mask,x,p,f,tau):return functional(w,b,x,p,f,mask,tau)
    for step in range(first+1,steps+1):
        ix=rng.integers(meta['windows'],size=4)
        batch={k:torch.from_numpy(np.array(v[ix],copy=True)).cuda() for k,v in data.items()}
        batch['f']=((batch['f']-fmean)/fstd).clamp(-10,10)
        tau=torch.rand(4,device='cuda',generator=generator).clamp(.002,.998)
        opt.zero_grad(set_to_none=True)
        result=vmap(apply,in_dims=(0,0,0,None,None,None,None))(params,buffers,masks,batch['x'],batch['p'],batch['f'],tau)
        e=batch['y'].unsqueeze(0)-result;ab=e.abs();k=.1
        huber=torch.where(ab<=k,.5*e.square(),k*(ab-.5*k))
        weight=(tau[None,:,None,None,None,None]-(e.detach()<0).to(e.dtype)).abs()
        # Exclude the dummy second channel for the single-channel combustion field.
        losses=(weight*huber/k*scenario.mask.unsqueeze(0)).sum((1,2,3,4,5))/(4*scenario.channels*20*32*64)
        if not torch.isfinite(losses).all():raise RuntimeError('Nonfinite head training loss')
        losses.sum().backward();norm=torch.zeros(3,device='cuda')
        for p in params:norm+=p.grad.flatten(1).square().sum(1)
        factor=(1/(norm.sqrt()+1e-6)).clamp(max=1)
        for p in params:p.grad.mul_(factor.reshape(3,*([1]*(p.ndim-1))))
        opt.param_groups[0]['lr']=1e-6+(3e-4-1e-6)*.5*(1+math.cos(math.pi*(step-1)/steps));opt.step()
        if step==1 or step%100==0:
            for i in range(3):log[i].append(dict(step=step,loss=float(losses[i])))
            print('HEAD',name,seed,step,[round(float(v),6) for v in losses],flush=True)
        if step%250==0:
            save_atomic(dict(params=[p.detach() for p in params],optimizer=opt.state_dict(),step=step,
                rng=rng.bit_generator.state,generator=generator.get_state(),log=log,
                elapsed=elapsed+time.monotonic()-begin),progress)
    elapsed+=time.monotonic()-begin
    for i,arm in enumerate(ARMS):
        state={k:v[i].detach().cpu() for k,v in zip(names,params)}
        state.update({k:initial[k].detach().cpu() for k,_ in template.named_buffers()})
        save_atomic(dict(model=state,scenario=name,arm=arm,seed=seed,steps=steps,
            f_mean=meta['f_mean'],f_std=meta['f_std'],backbone_sha256=meta['backbone_sha256'],
            head_parameters=sum(v.numel() for v in template.parameters()),group_wall_seconds=elapsed,loss=log[i],
            init_sha256=hashlib.sha256(b''.join(initial[k].detach().cpu().numpy().tobytes() for k in sorted(initial))).hexdigest()),output/(arm+'.pt'))
    progress.unlink(missing_ok=True);print('HEADS_COMPLETE',name,seed,elapsed,flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('scenario');parser.add_argument('--steps',type=int,default=2000);args=parser.parse_args()
    configure()
    for seed in (42,123,2026):train_seed(args.scenario,seed,args.steps)

if __name__=='__main__':main()
