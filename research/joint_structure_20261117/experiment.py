"""Frozen XPF marginals, observable-state residual transport, RealPDE only."""
from pathlib import Path
import sys, json, time, argparse, csv
import numpy as np
import torch
import torch.nn.functional as F
W=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(W/'xpf49_followup_20261116'))
from followcommon import (setup,load,save,dump,cache,norm,model,feature,features,
                         models_for,all_bounds,MANIFEST,SHARED,digest)
OUT=W/'runs/joint_structure_20261117'
N=32
FUNCTIONS=['tke_peak','kinetic_mean','enstrophy_grid_peak','speed_max','mvpe']

def seed(s=42):
    torch.manual_seed(s);np.random.seed(s)

def descriptor(x):
    a=x[:,:,-5:].mean(2);b=a-x[:,:,:5].mean(2)
    return torch.cat([F.adaptive_avg_pool2d(v,(4,8)).flatten(1) for v in (a,b)],1)

def functionals(z,p,n):
    # z: samples,2,20,32,64; physical velocity, unit-grid vorticity proxy.
    a=z*n['std']+n['mean'];center=a.mean((-1,-2),keepdim=True)
    tke=.5*(a-center).square().sum(1).mean((-1,-2)).amax(-1)
    kinetic=.5*a.square().sum(1).mean((1,2,3))
    vort=(a[:,1,:,1:-1,2:]-a[:,1,:,1:-1,:-2])/2-(a[:,0,:,2:,1:-1]-a[:,0,:,:-2,1:-1])/2
    ens=vort.square().mean((-1,-2)).amax(-1)
    speed=a.square().sum(1).sqrt().flatten(1).amax(1)
    mvpe=((z-p)*n['std']).square().sum(1).sqrt().flatten(1).amax(1)
    return torch.stack((tke,kinetic,ens,speed,mvpe),1)

def empirical_crps(z,y):
    count=len(z);v=z.sort(0).values
    weights=(2*torch.arange(1,count+1,device=z.device)-count-1).reshape(count,*([1]*(z.ndim-1)))
    return (z-y).abs().mean(0)-(v*weights).sum(0)/(count*count)

def variogram(z,y):
    vals=[]
    for dim in (3,2,1,0):
        dz=torch.diff(z,dim=dim+1).abs().sqrt().mean(0)
        dy=torch.diff(y,dim=dim).abs().sqrt()
        vals.append((dz-dy).square().mean())
    return torch.stack(vals)

def scores(z,y,p,n,stats):
    flat=z.flatten(1);target=y.flatten()
    first=(flat-target).square().mean(1).sqrt().mean()
    second=torch.pdist(flat).sum()/(N*N*np.sqrt(flat.shape[1]))
    v=variogram(z,y);f=functionals(z,p,n);fy=functionals(y[None],p,n)[0]
    fc=empirical_crps(f,fy)/stats['scale'];ev=(f[:,None,:]>stats['thresholds'][None]).float().mean(0)
    truth=(fy[None]>stats['thresholds']).float();br=(ev-truth).square()
    out=dict(es=float(first-second),vs=float(v.mean()),marginal_crps=float(empirical_crps(z,y).mean()),functional_crps=float(fc.mean()),brier=float(br.mean()))
    out.update({f'vs_{k}':float(s) for k,s in zip(('x','y','time','uv'),v)})
    for i,name in enumerate(FUNCTIONS):
        out[f'crps_{name}']=float(fc[i]);out[f'brier_{name}']=float(br[:,i].mean())
    return out,dict(samples=f.cpu(),truth=fy.cpu(),probability=ev.cpu())

def quantile_values(raw,p):
    q=torch.cat((torch.stack([v[0] for v in raw]),torch.stack([v[1] for v in reversed(raw)]))).sort(0).values
    q=torch.cat((q[:49],(q[48:49]+q[49:50])/2,q[49:]))[:,0]
    z=(torch.arange(N,device=p.device)+.5)/N*100-1
    lo=z.floor().long().clamp(0,98);hi=(lo+1).clamp(max=98);f=(z-lo).clamp(0,1)[:,None,None,None,None]
    return p[None]+q[lo]*(1-f)+q[hi]*f

def couple(values,template):
    return values.gather(0,template.argsort(0).argsort(0))

def shift_estimate(source,target):
    """Integer displacement of source to target, bounded y +/-2, x +/-4."""
    s=source-source.mean((-1,-2),keepdim=True);t=target-target.mean((-1,-2),keepdim=True)
    # Gradient-like emphasis suppresses the shared large-scale background.
    s=s-F.avg_pool2d(s,5,1,2);t=t-F.avg_pool2d(t,5,1,2)
    cs=(torch.fft.rfft2(t)[None]*torch.fft.rfft2(s).conj()).sum(1)
    cor=torch.fft.irfft2(cs,s=(32,64))
    yy=torch.arange(-2,3,device=s.device);xx=torch.arange(-4,5,device=s.device)
    local=cor[:,yy[:,None]%32,xx[None,:]%64].flatten(1)
    k=local.argmax(1)
    return torch.stack((xx[k%9],yy[k//9]),1).float()

def transport(r,source_p,p,mode):
    # Align early/late P, with shared displacement for both velocity components.
    d0=shift_estimate(source_p[:,:,:5].mean(2),p[:,:5].mean(1))
    d1=shift_estimate(source_p[:,:,-5:].mean(2),p[:,-5:].mean(1))
    if mode=='static':d0=d1=(d0+d1)/2
    t=torch.linspace(0,1,20,device=r.device)[None,:,None]
    d=d0[:,None]*(1-t)+d1[:,None]*t
    yy,xx=torch.meshgrid(torch.linspace(-1,1,32,device=r.device),torch.linspace(-1,1,64,device=r.device),indexing='ij')
    grid=torch.stack((xx,yy),-1)[None,None]-d[:,:,None,None,:]*r.new_tensor([2/63,2/31])
    a=r.permute(0,2,1,3,4).reshape(-1,2,32,64)
    result=F.grid_sample(a,grid.reshape(-1,32,64,2),align_corners=True,padding_mode='border')
    return result.reshape(N,20,2,32,64).permute(0,2,1,3,4),d

def library():
    path=OUT/'library.pt'
    if path.exists():return load(path,'cuda')
    d=cache('train');n=norm();desc=descriptor(d['x']);mu=desc.mean(0);sd=desc.std(0).clamp_min(.05)
    r=d['y']-d['p'];r=r/r.square().mean((2,3,4),keepdim=True).sqrt().clamp_min(.03)
    names=sorted({v['file'] for v in d['meta']});groups=[torch.tensor([i for i,m in enumerate(d['meta']) if m['file']==name],device='cuda') for name in names]
    seed();order=torch.randperm(len(names),device='cuda')[:N];ids=torch.stack([groups[i][torch.randint(len(groups[i]),(),device='cuda')] for i in order])
    fs=[]
    for i in range(0,len(r),16):fs.append(functionals(d['y'][i:i+16],d['p'][i:i+16],n))
    fs=torch.cat(fs);stats=dict(scale=fs.std(0).clamp_min(1e-6),thresholds=torch.quantile(fs,fs.new_tensor([.75,.9]),dim=0))
    value=dict(residual=r,p=d['p'],desc=(desc-mu)/sd,mean=mu,std=sd,groups=groups,unconditional=ids,stats=stats,meta=d['meta'])
    save({k:([{**m} for m in v] if k=='meta' else v) for k,v in value.items()},path)
    dump(OUT/'library_audit.json',dict(windows=len(r),trajectories=len(names),seed=42,manifest_sha256=digest(MANIFEST),checkpoint_sha256=digest(SHARED),data='original RealPDE only',test_read=False))
    del d
    return value

def matched(lib,x):
    z=(descriptor(x)-lib['mean'])/lib['std'];distance=(lib['desc']-z).square().mean(1)
    # At most one window per training trajectory, equal diversity to unconditional.
    ids=torch.stack([g[distance[g].argmin()] for g in lib['groups']])
    return ids[distance[ids].argsort()[:N]]

def summarize(split):
    import pandas as pd
    files=sorted((OUT/split/'parts').glob('*.pt'));rows=[]
    for f in files:
        v=load(f)
        for method,s in v['scores'].items():rows.append(dict(method=method,**v['meta'],**s))
    df=pd.DataFrame(rows);df.to_csv(OUT/split/'per_window.csv',index=False)
    metrics=[c for c in df if c not in ('method','file','group','start')]
    byfile=df.groupby(['method','file','group'])[metrics].mean().reset_index();byfile.to_csv(OUT/split/'per_trajectory.csv',index=False)
    summary=byfile.groupby('method')[metrics].mean();summary.to_csv(OUT/split/'summary.csv')
    print(summary[['es','vs','functional_crps','brier']].to_string(),flush=True)
    return summary

@torch.no_grad()
def run(split,limit=None):
    setup();seed();OUT.mkdir(parents=True,exist_ok=True)
    selected=None
    if split=='development':selected=json.loads((OUT/'selection.json').read_text())['selected']
    lib=library();d=cache(split);n=norm();pred=model();heads,batches=models_for('XPF')
    paths=OUT/split/'parts';paths.mkdir(parents=True,exist_ok=True)
    methods=['independent','comonotonic','empirical','matched','transport_static','transport_dynamic','ecc32','native32']
    if selected:methods=['independent','comonotonic','empirical','matched','ecc32','native32']+([] if selected=='matched' else [selected])
    total=len(d['x']) if limit is None else min(len(d['x']),limit)
    for i in range(total):
        dest=paths/f'{i:04d}.pt'
        if dest.exists():continue
        begin=time.perf_counter();x=d['x'][i:i+1]
        # Legacy environment forces TF32 despite backend flags. Batch >=2 matches
        # the historical batch-12 cache; duplicate the same observed input only.
        pp,ff=feature(pred,x.expand(2,-1,-1,-1,-1).contiguous(),'diffusion');p,f=pp[:1],ff[:1]
        assert torch.allclose(p[0],d['p'][i],atol=2e-5,rtol=1e-5)
        base,last=features(x,p,f);raw=all_bounds(batches,base,last);values=quantile_values(raw,p[0]);p=p[0]
        ids=matched(lib,x);templates={'empirical':lib['residual'][lib['unconditional']],'matched':lib['residual'][ids]}
        seed(420000+i);templates['independent']=torch.rand_like(values)
        templates['comonotonic']=torch.arange(N,device='cuda')[:,None,None,None,None].expand_as(values)
        shifts={}
        for mode in ('static','dynamic'):
            name='transport_'+mode
            if name in methods:templates[name],shifts[name]=transport(lib['residual'][ids],lib['p'][ids],p,mode)
        torch.cuda.synchronize();t=time.perf_counter()
        native=torch.stack([pred.rollout(x,500000+k,sign=1)[0] for k in range(N)])
        torch.cuda.synchronize();sampling_seconds=time.perf_counter()-t;templates['ecc32']=native
        # Predictive fields are fully constructed without reading the target.
        outputs={name:(native if name=='native32' else couple(values,templates[name])) for name in methods}
        y=d['y'][i];sc={};func={};marginal_error={}
        for name,z in outputs.items():
            sc[name],func[name]=scores(z,y,p,n,lib['stats'])
            if name!='native32':marginal_error[name]=float((z.sort(0).values-values).abs().max())
        save(dict(meta=d['meta'][i],scores=sc,functionals=func,matched_indices=ids.cpu(),shifts={k:v.cpu() for k,v in shifts.items()},marginal_error=marginal_error,sampling_seconds=sampling_seconds,total_seconds=time.perf_counter()-begin),dest)
        if i in (0,total-1):save(dict(values=values.cpu(),samples={k:v.cpu() for k,v in outputs.items()},y=y.cpu(),p=p.cpu(),stats={k:v.cpu() for k,v in lib['stats'].items()},norm={k:v.cpu() for k,v in n.items()}),OUT/split/f'audit_{i:04d}.pt')
        print('DONE',split,i+1,total,'seconds',round(time.perf_counter()-begin,2),flush=True)
    summary=summarize(split)
    if split=='calibration' and total==len(d['x']):
        ref=summary.loc['empirical'];candidate=['matched','transport_static','transport_dynamic']
        objective=(summary.loc[candidate,'es']/ref['es']+summary.loc[candidate,'vs']/ref['vs'])/2
        selected=objective.idxmin()
        dump(OUT/'selection.json',dict(selected=selected,criterion='mean of empirical-normalized ES and VS on calibration only',objective=objective.to_dict(),development_not_read=True,seed=42,completed_windows=total))
        print('SELECTED',selected,flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('split',choices=['calibration','development']);a.add_argument('--limit',type=int);args=a.parse_args();run(args.split,args.limit)
