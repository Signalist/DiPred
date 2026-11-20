"""Persistence, matched symmetric/directional models, and component diagnostics."""
import sys

import torch
from torch import nn
import torch.nn.functional as F

from common import *
sys.path.insert(0,str(REPO))
from research.residual_dynamics.run import ResidualPredictor


class SymmetricMixer(nn.Module):
    """D4-invariant positive smoothing; center/axis/diagonal share logits."""
    def __init__(self):
        super().__init__()
        self.logits=nn.Parameter(torch.tensor([[4.,0.,0.],[4.,0.,0.]]))
        self.gate=nn.Parameter(torch.full((2,),-3.))

    def forward(self,x):
        idx=torch.tensor([[2,1,2],[1,0,1],[2,1,2]],device=x.device)
        k=self.logits[:,idx].flatten(1).softmax(1).reshape(2,1,3,3)
        smooth=F.conv2d(F.pad(x,(1,1,1,1),mode='replicate'),k,groups=2)
        return self.gate.sigmoid()[None,:,None,None]*(smooth-x)


def model_new(arm,seed):
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    m=ResidualPredictor('B' if arm=='A4_symmetric' else arm)
    if arm=='A4_symmetric':
        m.mix=SymmetricMixer()
    return m.to(DEVICE)


def sample_points(e,n=262144,seed=1701):
    rng=torch.Generator(device=e.device).manual_seed(seed)
    sizes=[len(e),e.shape[1],e.shape[2],19,e.shape[-2],e.shape[-1]]
    return [torch.randint(s,(n,),device=e.device,generator=rng) for s in sizes]


def fit_linear(e,kind):
    """No intercept, no held-out data; analytical LS rather than neural budget."""
    prev,target=e[:,:,:,:-1],e[:,:,:,1:]
    if kind in ('A2_scaled','A2_lead'):
        dims=(0,1,2,5,6) if kind=='A2_lead' else (0,1,2,3,5,6)
        coef=(prev*target).double().sum(dims)/prev.double().square().sum(dims).clamp_min(1e-30)
        return dict(kind=kind,coef=coef.cpu())
    tr,wi,me,ti,yy,xx=sample_points(e)
    offsets=[(0,0)] if kind=='A3_linear' else [(y,x) for y in (-1,0,1) for x in (-1,0,1)]
    x=torch.stack([e[tr,wi,me,ti,:,(yy+y).clamp(0,31),(xx+x).clamp(0,63)]
                   for y,x in offsets],-1).reshape(len(tr),-1).double()
    if kind=='symmetric_linear3x3':
        v=x.reshape(-1,2,9)
        x=torch.stack([v[:,:,4],v[:,:,[1,3,5,7]].mean(-1),v[:,:,[0,2,6,8]].mean(-1)],-1).flatten(1)
    y=e[tr,wi,me,ti+1,:,yy,xx].double()
    gram=x.T@x
    ridge=float(gram.diag().mean())*1e-8
    coef=torch.linalg.solve(gram+torch.eye(len(gram),device=e.device)*ridge,x.T@y)
    return dict(kind=kind,coef=coef.cpu(),ridge=ridge,points=len(tr),sample_seed=1701)


def simple_predict(x,leads,ck):
    kind=ck['kind']
    if kind=='A0_zero': return torch.zeros_like(x)
    if kind=='A1_persistence': return x
    if kind=='A2_scaled': return x*ck['coef'].to(x)[None,:,None,None]
    if kind=='A2_lead': return x*ck['coef'].to(x)[leads-2,:,None,None]
    if kind=='A3_linear': return torch.einsum('bchw,cd->bdhw',x,ck['coef'].to(x))
    patches=F.unfold(F.pad(x,(1,1,1,1),mode='replicate'),3)
    if kind=='symmetric_linear3x3':
        v=patches.reshape(len(x),2,9,-1)
        patches=torch.stack([v[:,:,4],v[:,:,[1,3,5,7]].mean(2),v[:,:,[0,2,6,8]].mean(2)],2).flatten(1,2)
    return torch.einsum('bdp,dc->bcp',patches,ck['coef'].to(x)).reshape_as(x)


def network_predict(x,leads,model,scale):
    return model(x/scale,leads)*scale


@torch.no_grad()
def errors(e,predict,shuffle_seed=None,mirror=False):
    results=[]
    gen=torch.Generator(device=e.device).manual_seed(shuffle_seed or 0)
    for trajectory in e:
        errors=[]
        for window in trajectory:
            x=window[:,:-1].clone()
            if shuffle_seed is not None:
                for t in range(19): x[:,t]=x[torch.randperm(len(x),generator=gen,device=x.device),t]
            x=x.reshape(-1,2,32,64)
            leads=torch.arange(2,21,device=x.device).repeat(len(window))
            predictions=[]
            for begin in range(0,len(x),64):
                v=x[begin:begin+64]
                if mirror: v=v.flip(-1)
                pred=predict(v,leads[begin:begin+64])
                if mirror: pred=pred.flip(-1)
                predictions.append(pred)
            delta=torch.cat(predictions).reshape(len(window),19,2,32,64)-window[:,1:]
            errors.append(delta.double().square().mean((0,3,4)).cpu().numpy())
        results.append(np.mean(errors,0))
    return np.array(results)  # trajectory,lead,channel


def learn():
    e,dev=residual('train'),residual('development')
    scale=float(e.square().mean().sqrt())
    analytic={kind:fit_linear(e,kind) for kind in ('A2_scaled','A2_lead','A3_linear','linear3x3')}
    torch.save(analytic,EXTRA/'analytic.pt')
    dump(OUT/'analytic_parameters.json',{k:{n:v.tolist() if torch.is_tensor(v) else v for n,v in ck.items()}
                                       for k,ck in analytic.items()})
    logs=[]; parameters={}
    for seed in SEEDS:
        for arm in ('B','C','A4_symmetric'):
            dest=EXTRA/f'mechanism_{arm}_{seed}.pt'
            if dest.exists(): continue
            m=model_new(arm,seed)
            parameters[arm]=sum(p.numel() for p in m.parameters())
            opt=torch.optim.Adam(m.parameters(),lr=.001)
            rng=torch.Generator(device=DEVICE).manual_seed(20261119)
            best=float('inf')
            for step in range(1,2001):
                tr=torch.randint(len(e),(16,),device=DEVICE,generator=rng)
                wi=torch.randint(2,(16,),device=DEVICE,generator=rng)
                me=torch.randint(32,(16,),device=DEVICE,generator=rng)
                ti=torch.randint(19,(16,),device=DEVICE,generator=rng)
                x,y=e[tr,wi,me,ti],e[tr,wi,me,ti+1]
                loss=(m(x/scale,ti+2)-y/scale).square().mean()
                opt.zero_grad(set_to_none=True);loss.backward();opt.step()
                if step%250==0:
                    m.eval()
                    score=float(errors(dev,lambda x,l:network_predict(x,l,m,scale)).mean())
                    logs.append(dict(arm=arm,seed=seed,step=step,development_mse=score))
                    if score<best:
                        best=score
                        torch.save(dict(model=m.state_dict(),scale=scale,step=step,seed=seed,arm=arm),dest)
                    m.train()
                    print('LEARN',arm,seed,step,score,flush=True)
            dump(OUT/'mechanism_training.json',dict(log=logs,parameters=parameters,scale=scale,
                analytic='A2 exact all-training moments; A3/linear3x3 ridge LS on262144 fixed train pixels'))


def kernel_variant(m,variant):
    v=m.mix.weight.detach().clone()
    if variant=='reverse180': v=v.flip((-1,-2))
    if variant=='flipx': v=v.flip(-1)
    if variant=='flipy': v=v.flip(-2)
    if variant=='symmetrized':
        variants=[torch.rot90(v,k,(-2,-1)) for k in range(4)]
        variants += [z.flip(-1) for z in variants]
        v=torch.stack(variants).mean(0)
    with torch.no_grad():m.mix.weight.copy_(v)


def score():
    analytic=torch.load(EXTRA/'analytic.pt',map_location='cpu',weights_only=False)
    detail=[];kernels=[]
    for split in ('development','test'):
        e=residual(split)
        jobs=[]
        for name in ('A0_zero','A1_persistence'):
            ck=dict(kind=name)
            jobs.append((name,-1,lambda x,l,ck=ck:simple_predict(x,l,ck),False))
        for name,ck in analytic.items():
            jobs.append((name,-1,lambda x,l,ck=ck:simple_predict(x,l,ck),False))
        for seed in SEEDS:
            for arm in ('B','C','A4_symmetric'):
                m=model_new(arm,seed).eval()
                ck=torch.load(EXTRA/f'mechanism_{arm}_{seed}.pt',map_location=DEVICE,weights_only=False)
                m.load_state_dict(ck['model'])
                if split=='test' and arm=='C':
                    k=m.mix.weight.detach().cpu().numpy()[:,0]
                    kernels.append(dict(seed=seed,kernel=k.tolist(),
                        horizontal_antisymmetric_norm=float(np.linalg.norm(k-k[...,::-1])),
                        vertical_antisymmetric_norm=float(np.linalg.norm(k-k[:,::-1,:])),
                        note='raw depthwise mixing is not the effective nonlinear PDE propagator'))
                jobs.append((arm,seed,lambda x,l,m=m,s=ck['scale']:network_predict(x,l,m,s),False))
                if seed==42:
                    jobs.append((arm+'_coordinate_mirror_x',seed,
                        lambda x,l,m=m,s=ck['scale']:network_predict(x,l,m,s),True))
                if arm=='C':
                    for variant in ('reverse180','flipx','flipy','symmetrized'):
                        modified=model_new('C',seed).eval();modified.load_state_dict(ck['model'])
                        kernel_variant(modified,variant)
                        jobs.append(('C_'+variant,seed,lambda x,l,m=modified,s=ck['scale']:network_predict(x,l,m,s),False))
        # Offset signs are image-grid coordinates, not a physical velocity label.
        for dy,dx in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
            def shifted(x,l,dy=dy,dx=dx):
                p=F.pad(x,(1,1,1,1),mode='replicate')[:,:,1+dy:33+dy,1+dx:65+dx]
                return simple_predict(p,l,analytic['A2_scaled'])
            jobs.append((f'offset_y{dy}_x{dx}',-1,shifted,False))
        for name,seed,fn,mirror in jobs:
            v=errors(e,fn,mirror=mirror)
            for i,r in enumerate(protocol()['records'][split]):
                for t in range(19):
                    for c,ch in enumerate(('u','v')):
                        detail.append(dict(split=split,method=name,seed=seed,trajectory=r['file'],
                            Re=r['group'],channel=ch,lead=t+2,mse=float(v[i,t,c])))
            print('MECHANISM',split,name,seed,float(v.mean()),flush=True)
        del e
    csvwrite(OUT/'mechanism_by_trajectory_lead_channel.csv',detail)
    summaries=aggregate(detail,['split','method','seed'],['mse'])
    csvwrite(OUT/'mechanism_summary.csv',summaries)
    csvwrite(OUT/'mechanism_by_Re_channel.csv',aggregate(detail,['split','method','seed','Re','channel'],['mse']))
    csvwrite(OUT/'mechanism_by_lead_channel.csv',aggregate(detail,['split','method','seed','lead','channel'],['mse']))
    dump(OUT/'kernels.json',kernels)
    paired=[]
    trajectories=aggregate(detail,['split','method','seed','trajectory'],['mse'])
    lookup={(r['split'],r['method'],r['seed'],r['trajectory']):r['mse'] for r in trajectories}
    comparisons=[('B','A1_persistence'),('C','A4_symmetric'),('C','B'),('A2_scaled','A1_persistence'),
                 ('A3_linear','A1_persistence'),('C','linear3x3'),
                 ('C_reverse180','C'),('C_flipx','C'),('C_flipy','C'),('C_symmetrized','C')]
    for split in ('development','test'):
        for a,b in comparisons:
            seeds=SEEDS if a.startswith(('B','C')) else (-1,)
            for seed in seeds:
                sb=seed if b in ('B','C','A4_symmetric') else -1
                d=[lookup[split,a,seed,r['file']]-lookup[split,b,sb,r['file']]
                   for r in protocol()['records'][split]]
                paired.append(dict(split=split,contrast=a+' minus '+b,seed=seed,**bootstrap(d)))
    dump(OUT/'mechanism_paired_bootstrap.json',paired)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['fit','score'])
    args=p.parse_args()
    from reliability.backbone import configure
    configure()
    (learn if args.stage=='fit' else score)()
