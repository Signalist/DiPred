"""Small, signed field corrections. Every learned input is available at issue time."""
from pathlib import Path
import sys, os, json, time, argparse
os.environ['TORCH_ALLOW_TF32_CUBLAS_OVERRIDE']='0'
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from functools import lru_cache
W=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(W/'f_risk_query_20261116'))
from common import setup,load,save,dump,digest,model,norm as source_norm,SHARED,MANIFEST,OUT as PRIOR
OUT=W/'runs/error_structure_20261117'
NAMES=['dx','dy','phase','amplitude_u','amplitude_v','bias_u','bias_v']
LIMITS=[1.,1.,1.,.25,.25,.25,.25]

@lru_cache(None)
def norm(device='cuda'):return source_norm(device)

def mse(p,y):
    return ((p-y)*norm(p.device)['std']).square().mean((1,2,3,4))

def pooled(x):
    if x.ndim==5:return F.adaptive_avg_pool3d(x,(4,4,8)).flatten(1)
    return F.adaptive_avg_pool2d(x,(4,8)).flatten(1)

def capture(m,x):
    buffers={k:[] for k in ('mid','up2')};handles=[]
    for k in buffers:
        handles.append(getattr(m.net,k).register_forward_hook(lambda mod,args,out,k=k:buffers[k].append(pooled(out).detach())))
    try:p=m.rollout(x,500000,sign=0)
    finally:
        for h in handles:h.remove()
    assert all(len(v)==4 for v in buffers.values())
    return p,dict(F_last=buffers['up2'][-1],F_mid0=buffers['mid'][-1],F_mid99=buffers['mid'][0])

@torch.no_grad()
def prepare(split):
    dest=OUT/f'{split}.pt'
    if dest.exists():return
    src=load((W/'runs/joint_structure_20261117'/f'{split}_data.pt') if split in ('late','future') else (PRIOR/'data'/f'{split}.pt'))
    x,y,meta=src['x'],src['y'],src['meta'];del src
    m=model();ps=[];feats={};start=time.perf_counter()
    for i in range(0,len(x),8):
        h=x[i:i+8].cuda();p,f=capture(m,h);ps.append(p.cpu())
        f.update(X=pooled(h),P=pooled(p))
        for k,v in f.items():feats.setdefault(k,[]).append(v.cpu())
    p=torch.cat(ps);feats={k:torch.cat(v) for k,v in feats.items()}
    save(dict(x=x,y=y,p=p,features=feats,meta=meta),dest)
    print('PREPARED',split,len(x),round(time.perf_counter()-start,2),flush=True)

def tangent(p):
    dx=torch.gradient(p,dim=4)[0];dy=torch.gradient(p,dim=3)[0];dt=torch.gradient(p,dim=2)[0]
    u=torch.zeros_like(p);v=torch.zeros_like(p);u[:,0]=p[:,0];v[:,1]=p[:,1]
    bu=torch.zeros_like(p);bv=torch.zeros_like(p);bu[:,0]=1;bv[:,1]=1
    return torch.stack([dx,dy,dt,u,v,bu,bv],-1)

def linear_oracle(p,y):
    b=tangent(p);sd=norm(p.device)['std']
    a=(b*sd[...,None]).flatten(1,4);r=((y-p)*sd).flatten(1)
    gram=a.transpose(1,2)@a
    # Scale-aware numerical ridge, not fitted hyperparameters.
    reg=gram.diagonal(dim1=1,dim2=2).clamp_min(1e-8)*1e-6
    c=torch.linalg.solve(gram+torch.diag_embed(reg),(a.transpose(1,2)@r[...,None]))[...,0]
    limits=p.new_tensor(LIMITS);c=c.clamp(-limits,limits)
    return c,b

def transform(p,c):
    """c is [B,7], positive displacement samples P at larger x/y/time."""
    if not c.requires_grad and torch.count_nonzero(c)==0:return p
    nt,ny,nx=p.shape[-3:]
    t,y,x=torch.meshgrid(torch.linspace(-1,1,nt,device=p.device),torch.linspace(-1,1,ny,device=p.device),torch.linspace(-1,1,nx,device=p.device),indexing='ij')
    grid=torch.stack([x,y,t],-1)[None]+c[:,:3,None,None,None].permute(0,2,3,4,1)*p.new_tensor([2/(nx-1),2/(ny-1),2/(nt-1)])
    q=F.grid_sample(p,grid,padding_mode='border',align_corners=True)
    return q*(1+c[:,3:5,None,None,None])+c[:,5:7,None,None,None]

def optimize_oracle(p,y,steps=80):
    with torch.no_grad():c,_=linear_oracle(p,y)
    limits=p.new_tensor(LIMITS)
    c=c.detach().requires_grad_();opt=torch.optim.Adam([c],lr=.025)
    best=mse(p,y).detach();bestc=torch.zeros_like(c)
    for _ in range(steps):
        opt.zero_grad();q=transform(p,c);err=mse(q,y)
        with torch.no_grad():
            better=err<best;best[better]=err[better];bestc[better]=c[better]
        (err/norm(p.device)['std'].square().mean()).mean().backward();opt.step()
        with torch.no_grad():c.clamp_(-limits,limits)
    return bestc.detach(),best.detach()

def diagnostics(split):
    d=load(OUT/f'{split}.pt');rows=[];coefs=[];linears=[]
    for i in range(0,len(d['p']),8):
        p=d['p'][i:i+8].cuda();y=d['y'][i:i+8].cuda()
        c,b=linear_oracle(p,y);q=p+(b*c[:,None,None,None,None,:]).sum(-1)
        oc,err=optimize_oracle(p,y);coefs.append(oc.cpu());linears.append(c.cpu())
        base=mse(p,y);linear=mse(q,y)
        for j,m in enumerate(d['meta'][i:i+8]):
            rows.extend([dict(**m,method=key,mse=float(val[j])) for key,val in [('base',base),('tangent_oracle',linear),('warp_oracle',err)]])
    save(dict(coefficients=torch.cat(coefs),tangent_coefficients=torch.cat(linears),meta=d['meta']),OUT/f'{split}_oracle.pt')
    df=pd.DataFrame(rows);df.to_csv(OUT/f'{split}_oracle.csv',index=False)
    summary=df.groupby(['method','file']).mse.mean().groupby('method').mean()
    print('ORACLE',split,summary.to_dict(),'gain',((1-summary/summary['base'])*100).to_dict(),flush=True)

def project_fit(z,ids):
    fit=z[ids].double();mu=fit.mean(0);sd=fit.std(0).clamp_min(.01);a=(fit-mu)/sd
    # Full SVD avoids a separate randomized feature approximation seed.
    _,_,v=torch.linalg.svd(a,full_matrices=False);basis=v[:64].T
    zz=((z.double()-mu)/sd)@basis
    scale=zz[ids].std(0).clamp_min(.01);zz=zz/scale
    return dict(mean=mu,scale=sd,basis=basis,pc_scale=scale),zz

def project(z,ck):
    return ((z.double()-ck['mean'])/ck['scale'])@ck['basis']/ck['pc_scale']

def describe(d,kind):
    keys=kind.split('+');return torch.cat([d['features'][k] for k in keys],1).cuda()

def ridge(z,target,alpha):
    z=torch.cat([torch.ones_like(z[:,:1]),z],1);reg=torch.eye(z.shape[1],device=z.device,dtype=z.dtype)*alpha*len(z);reg[0,0]=0
    return torch.linalg.solve(z.T@z+reg,z.T@target.double())

def predict(z,w):return (torch.cat([torch.ones_like(z[:,:1]),z],1)@w).float()

def additive_fit(d,ids):
    r=(d['y'][ids]-d['p'][ids]).cuda();z=F.adaptive_avg_pool3d(r,(4,8,16)).flatten(1)
    # No mean mode is removed: the leading singular mode can represent bias.
    _,_,v=torch.linalg.svd(z,full_matrices=False)
    b=F.interpolate(v[:7].reshape(7,2,4,8,16),size=(20,32,64),mode='trilinear',align_corners=False)
    sd=norm()['std'];a=(b*sd).flatten(1).T;gram=a.T@a
    gram+=torch.eye(7,device=a.device)*gram.diag().mean()*1e-6
    return b,torch.linalg.solve(gram,a.T)

def labels(d,basis_inverse):
    return (((d['y']-d['p']).cuda()*norm()['std']).flatten(1)@basis_inverse.T)

def decode(p,c,style,b):
    if style=='warp':return transform(p,c.clamp(-p.new_tensor(LIMITS),p.new_tensor(LIMITS)))
    return p+torch.einsum('bk,kcthw->bcthw',c,b)

def aggregate(error,meta):
    return float(pd.DataFrame(dict(file=[m['file'] for m in meta],mse=error.detach().cpu().numpy())).groupby('file').mse.mean().mean())

def learn():
    train=load(OUT/'train.pt');cal=load(OUT/'calibration.pt');oracle=load(OUT/'train_oracle.pt')['coefficients'].cuda()
    names=sorted({m['file'] for m in train['meta']});rng=np.random.default_rng(42);rng.shuffle(names)
    variants=['X','P','X+P','F_last','F_mid0','F_mid99','X+P+F_last']
    variants += [k for k in ('F_det4_1500','F_det4_6000') if k in train['features']]
    selected=load(OUT/'readouts.pt','cuda') if (OUT/'readouts.pt').exists() else {}
    rows=pd.read_csv(OUT/'selection_grid.csv').to_dict('records') if (OUT/'selection_grid.csv').exists() else []
    cp=cal['p'].cuda();cy=cal['y'].cuda()
    for fraction in (.25,.5,1.):
        included=names[:max(1,round(len(names)*fraction))];ids=torch.tensor([i for i,m in enumerate(train['meta']) if m['file'] in included],device='cuda')
        b,inv=additive_fit(train,ids.cpu());addlabels=labels(train,inv)
        for kind in variants:
            if all(f'{style}/{kind}/{fraction:g}' in selected for style in ('warp','additive')):continue
            proj,z=project_fit(describe(train,kind),ids);zc=project(describe(cal,kind),proj)
            for style,target in [('warp',oracle),('additive',addlabels)]:
                key=f'{style}/{kind}/{fraction:g}';best=None
                for alpha in (.01,.1,1.,10.):
                    w=ridge(z[ids],target[ids],alpha);coef=predict(zc,w)
                    for shrink in (0.,.25,.5,1.):
                        error=mse(decode(cp,coef*shrink,style,b),cy);score=aggregate(error,cal['meta'])
                        rows.append(dict(model=key,alpha=alpha,shrink=shrink,mse=score))
                        if best is None or score<best['calibration_mse']:
                            best=dict(projection=proj,weights=w,alpha=alpha,shrink=shrink,calibration_mse=score,basis=b if style=='additive' else None,kind=kind,style=style,fraction=fraction,fit_files=included,fit_windows=len(ids))
                selected[key]=best
                print('FIT',key,best['alpha'],best['shrink'],best['calibration_mse'],flush=True)
    save(selected,OUT/'readouts.pt');pd.DataFrame(rows).to_csv(OUT/'selection_grid.csv',index=False)
    dump(OUT/'selection.json',{k:{a:(float(b) if isinstance(b,np.floating) else b) for a,b in v.items() if a not in ('projection','weights','basis')} for k,v in selected.items()})

@torch.no_grad()
def evaluate(split):
    d=load(OUT/f'{split}.pt');models=load(OUT/'readouts.pt','cuda');rows=[]
    p=d['p'].cuda();y=d['y'].cuda();base=mse(p,y)
    for name in ['base']+list(models):
        if name=='base':q=p
        else:
            ck=models[name];z=project(describe(d,ck['kind']),ck['projection']);c=predict(z,ck['weights'])*ck['shrink'];q=decode(p,c,ck['style'],ck['basis'])
        errors=mse(q,y);nmse=(q-y).square().mean((1,2,3,4));sd=norm()['std'];mu=norm()['mean']
        kin=(.5*(q*sd+mu).square().sum(1).mean((1,2,3))-.5*(y*sd+mu).square().sum(1).mean((1,2,3))).abs()
        for i,m in enumerate(d['meta']):rows.append(dict(**m,method=name,mse=float(errors[i]),normalized_mse=float(nmse[i]),kinetic_error=float(kin[i])))
    df=pd.DataFrame(rows);df.to_csv(OUT/f'{split}_learned.csv',index=False)
    byfile=df.groupby(['method','file','group'])[['mse','normalized_mse','kinetic_error']].mean().reset_index();byfile.to_csv(OUT/f'{split}_trajectories.csv',index=False)
    summary=byfile.groupby('method')[['mse','normalized_mse','kinetic_error']].mean();summary['gain_percent']=(1-summary.mse/summary.loc['base','mse'])*100
    summary.to_csv(OUT/f'{split}_summary.csv');print(summary.to_string(),flush=True)

def main():
    setup();OUT.mkdir(parents=True,exist_ok=True)
    a=argparse.ArgumentParser();a.add_argument('action',choices=['prepare','diagnose','learn','evaluate']);a.add_argument('--split',default='train',choices=['train','calibration','development','late','future']);args=a.parse_args()
    if args.action=='prepare':prepare(args.split)
    elif args.action=='diagnose':diagnostics(args.split)
    elif args.action=='learn':learn()
    else:evaluate(args.split)

if __name__=='__main__':main()
