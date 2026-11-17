"""Matched center-fixed risk readout; all heads predict the same P error."""
from common import *
import math,time,csv,argparse
from torch import nn
from torch.nn import functional as TF
from functorch import combine_state_for_ensemble,vmap
ARMS=('X','P','F','XPF','Fdet','Pprojection','Frandom','Fshuffle')
class RiskHead(nn.Module):
    def __init__(self,width=16,symmetric=False):
        super().__init__();self.symmetric=symmetric
        self.proj=nn.Conv2d(48,160,1)
        self.enc=nn.Sequential(nn.Conv3d(12,width,3,padding=1),nn.GroupNorm(4,width),nn.SiLU(),
            nn.Conv3d(width,width,3,padding=(1,2,2),dilation=(1,2,2)),nn.GroupNorm(4,width),nn.SiLU(),
            nn.Conv3d(width,width,3,padding=1),nn.GroupNorm(4,width),nn.SiLU(),nn.Conv3d(width,12,1))
    def weights(self,x,p,f,mask):
        z=self.proj(f).reshape(len(x),8,20,32,64)
        v=self.enc(torch.cat((x*mask[0],p*mask[1],z*mask[2]),1))
        return TF.softplus(v.reshape(len(x),2,6,20,32,64))/6
    def quantile_bases(self,t):
        return .5*(bases(t)-bases(1-t)) if self.symmetric else bases(t)
    def forward(self,x,p,f,mask,tau):
        w=self.weights(x,p,f,mask)
        return p+(w*self.quantile_bases(tau)[:,None,:,None,None,None]).sum(2)
    def quantiles(self,x,p,f,mask,levels):
        return p[None]+torch.einsum('bckthw,qk->qbcthw',self.weights(x,p,f,mask),self.quantile_bases(levels))
def features(data,arm,ids,donors=None):
    f=data['f'][ids]
    if arm=='Fdet':f=data['fd'][ids]
    if arm=='Frandom':f=data['fr'][ids]
    if arm=='Fshuffle':f=donors[ids]
    if arm=='Pprojection':
        gen=torch.Generator(device='cuda').manual_seed(42)
        mat=torch.randn(48,40,device='cuda',generator=gen)/(40**.5)
        f=torch.einsum('oc,bchw->bohw',mat,data['p'][ids].flatten(1,2))
    mask=torch.tensor((float(arm in ('X','XPF')),float(arm in ('P','XPF')),float(arm not in ('X','P'))),device='cuda')
    return data['x'][ids],data['p'][ids],f,mask

def train(arms=ARMS,width=16,steps=2000,tag='main',fraction=1.,symmetric=False):
    setup();data=cache('train');folder=OUT/'risk'/tag;folder.mkdir(parents=True,exist_ok=True)
    if all((folder/(a+'.pt')).exists() for a in arms):return
    torch.manual_seed(42);template=RiskHead(width,symmetric).cuda();initial={k:v.detach().clone() for k,v in template.state_dict().items()}
    models=[]
    for _ in arms:
        m=RiskHead(width,symmetric).cuda();m.load_state_dict(initial);models.append(m)
    fn,params,buffers=combine_state_for_ensemble(models);params=tuple(v.requires_grad_() for v in params)
    names=[k for k,_ in template.named_parameters()];opt=torch.optim.AdamW(params,lr=3e-4,weight_decay=1e-5)
    rng=torch.Generator(device='cuda').manual_seed(42);perm=torch.randperm(len(data['x']),device='cuda',generator=rng)
    donors=data['f'][perm];rng.manual_seed(42)
    files=sorted(set(r['file'] for r in data['meta']));allowed=set(np.random.default_rng(42).permutation(files)[:max(1,round(len(files)*fraction))])
    available=torch.tensor([i for i,r in enumerate(data['meta']) if r['file'] in allowed],device='cuda')
    logs=[];begin=time.monotonic();schedule=hashlib.sha256();first=0
    progress=folder/'progress.pt'
    if progress.exists():
        prev=load(progress,'cuda');first=prev['step'];logs=prev['logs'];rng.set_state(prev['rng'].cpu())
        for p,q in zip(params,prev['params']):p.data.copy_(q)
        opt.load_state_dict(prev['optimizer'])
    for step in range(first+1,steps+1):
        ids=available[torch.randint(len(available),(4,),device='cuda',generator=rng)];tau=torch.rand(4,device='cuda',generator=rng).clamp(.002,.998)
        batches=[features(data,a,ids,donors) for a in arms];x,p,f,mask=[torch.stack([b[j] for b in batches]) for j in range(4)]
        opt.zero_grad(set_to_none=True)
        q=vmap(fn,in_dims=(0,0,0,0,0,0,None))(params,buffers,x,p,f,mask,tau)
        e=data['y'][ids][None]-q;t=tau[None,:,None,None,None,None]
        losses=torch.maximum(t*e,(t-1)*e).flatten(1).mean(1)
        assert torch.isfinite(losses).all();losses.sum().backward()
        size=sum(v.grad.flatten(1).square().sum(1) for v in params).sqrt();fac=(1/(size+1e-6)).clamp(max=1)
        for v in params:v.grad.mul_(fac.reshape(len(arms),*([1]*(v.ndim-1))))
        opt.param_groups[0]['lr']=1e-6+(3e-4-1e-6)*.5*(1+math.cos(math.pi*(step-1)/steps));opt.step()
        schedule.update(ids.cpu().numpy().tobytes());schedule.update(tau.cpu().numpy().tobytes())
        if step==1 or step%100==0:
            row=dict(step=step,loss=losses.detach().cpu().tolist(),elapsed=time.monotonic()-begin);logs.append(row);print('RISK',tag,row,flush=True)
        if step%250==0:save(dict(params=[p.detach() for p in params],optimizer=opt.state_dict(),rng=rng.get_state(),step=step,logs=logs),progress)
    init_hash=hashlib.sha256(b''.join(initial[k].cpu().numpy().tobytes() for k in sorted(initial))).hexdigest()
    for i,a in enumerate(arms):
        state={k:v[i].detach().cpu() for k,v in zip(names,params)}
        save(dict(model=state,width=width,symmetric=symmetric,seed=42,steps=steps,arm=a,parameters=sum(v.numel() for v in template.parameters()),
            initial_sha256=init_hash,schedule_sha256=schedule.hexdigest(),training_files=sorted(allowed),logs=logs),folder/(a+'.pt'))
    progress.unlink(missing_ok=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--arms',nargs='+',default=list(ARMS));ap.add_argument('--steps',type=int,default=2000);ap.add_argument('--tag',default='main');ap.add_argument('--width',type=int,default=16);ap.add_argument('--fraction',type=float,default=1.)
    a=ap.parse_args();train(a.arms,a.width,a.steps,a.tag,a.fraction)
