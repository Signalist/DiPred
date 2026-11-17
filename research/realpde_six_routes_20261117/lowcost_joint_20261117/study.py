"""Low-cost, residual-trained Gaussian dependence at fixed XPF marginals."""
from pathlib import Path
import os,sys,time,json,argparse
os.environ['TORCH_ALLOW_TF32_CUBLAS_OVERRIDE']='0'
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
W=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(W/'error_structure_20261117'))
import structure as st
sys.path.insert(0,str(W/'joint_structure_20261117'))
import experiment as legacy_exp
# Import the archived scoring module under its established module name.
from followcommon import setup,load,save,dump,digest,norm,model,feature,features,models_for,all_bounds,SHARED,MANIFEST
from physics import physical_functionals,divergence,DX
from experiment import quantile_values,couple,empirical_crps,scores,N
OUT=W/'runs/lowcost_joint_20261117'
SOURCE=W/'runs/error_structure_20261117'
KINDS=['X','P','F_last','F_mid0']
AR_STAGE=False
SCOREFIT_STAGE=False

def smooth(z,k=3,replicate=True):
    a=z.reshape(-1,1,*z.shape[-2:]);r=k//2
    if replicate:a=F.avg_pool2d(F.pad(a,(r,r,r,r),mode='replicate'),k,1)
    else:
        a=F.avg_pool2d(a,k,1,r,count_include_pad=False)
    return a.reshape_as(z)

def necc(values,raw,k):
    """All offset tilings, edge-complete truncated patches, averaged.

    Inputs [members,channels,time,height,width]. Pad with +inf so virtual
    entries sort after real entries; fold discards virtual positions.
    """
    n,c,t,h,w=values.shape;pad=k-1
    def patches(z):
        a=F.unfold(F.pad(z.reshape(n*c*t,1,h,w),(pad,pad,pad,pad),value=float('inf')),kernel_size=k)
        return a.reshape(n,c*t,k*k,-1).permute(1,3,0,2).reshape(c*t,-1,n*k*k)
    v=patches(values);r=patches(raw)
    # Chunk time/component batches to limit large width9 temporary allocations.
    pieces=[]
    for i in range(0,c*t,4):
        rank=r[i:i+4].argsort(dim=-1,stable=True).argsort(dim=-1,stable=True)
        q=v[i:i+4].sort(-1).values.gather(-1,rank);pieces.append(q)
    q=torch.cat(pieces).reshape(c*t,-1,n,k*k).permute(2,0,3,1).reshape(n*c*t,k*k,-1)
    return (F.fold(q,output_size=(h,w),kernel_size=k,padding=pad)/(k*k)).reshape(n,c,t,h,w)

@torch.no_grad()
def fit():
    OUT.mkdir(parents=True,exist_ok=True);d=load(SOURCE/'train.pt','cuda');r=(d['y']-d['p']).flatten(1)
    rank=r.argsort(dim=0,stable=True).argsort(dim=0,stable=True).float()
    z=torch.erfinv(2*(rank+.5)/len(r)-1)*np.sqrt(2)
    z-=z.mean(0);z/=z.std(0).clamp_min(1e-6)
    gram=z@z.T/len(z);lam,u=torch.linalg.eigh(gram)
    b=(u[:,-32:].T@z)/np.sqrt(len(z))
    # b.T b is rank32 covariance approximation; normalize each coordinate.
    b=b/b.square().sum(0).sqrt().clamp_min(1e-6)
    proj={};desc={}
    for kind in KINDS:
        proj[kind],desc[kind]=st.project_fit(d['features'][kind],torch.arange(len(r),device='cuda'))
    names=sorted({m['file'] for m in d['meta']});groups=[torch.tensor([i for i,m in enumerate(d['meta']) if m['file']==f],device='cuda') for f in names]
    generator=torch.Generator(device='cuda').manual_seed(42);ids=[]
    for gi in torch.randperm(len(groups),generator=generator,device='cuda')[:32]:
        g=groups[int(gi)];ids.append(g[torch.randint(len(g),(),device='cuda',generator=generator)])
    stats=[];n=norm()
    for i in range(0,len(r),16):stats.append(physical_functionals(d['y'][i:i+16],d['p'][i:i+16],n))
    fs=torch.cat(stats);stats=dict(scale=fs.std(0).clamp_min(1e-6),thresholds=torch.quantile(fs,fs.new_tensor([.75,.9]),dim=0))
    save(dict(gaussianized=z.reshape(-1,2,20,32,64).half(),pca=b.reshape(32,2,20,32,64),projection=proj,descriptor=desc,groups=groups,unconditional=torch.stack(ids),stats=stats,meta=d['meta']),OUT/'library.pt')
    dump(OUT/'fit.json',dict(seed=42,windows=len(r),trajectories=len(names),pca_rank=32,explained_variance_fraction=float(lam[-32:].sum()/lam.sum()),manifest_sha256=digest(MANIFEST),input_source_sha256=digest(SOURCE/'train.pt'),description='Gaussianized true train residual dependence; no held-out targets used.'))
    print('FIT',json.loads((OUT/'fit.json').read_text()),flush=True)

@torch.no_grad()
def fit_conditional_scores():
    """Normalize train residuals by their own forecast marginal quantile map."""
    parent=OUT.parent;d=load(SOURCE/'train.pt','cuda');pred=Predictor(library=False);zs=[]
    levels=(torch.arange(32,device='cuda')+.5)/32
    normal=torch.erfinv(2*levels-1)*np.sqrt(2)
    for i in range(len(d['x'])):
        p,q,desc=pred.issue(d['x'][i:i+1]);y=d['y'][i]
        idx=(q<=y).sum(0);lo=(idx-1).clamp(0,30);hi=lo+1
        qlo=q.gather(0,lo[None])[0];qhi=q.gather(0,hi[None])[0]
        # Linear tails in normal-score coordinates, not clipped uniform ranks.
        z=normal[lo]+(y-qlo)/(qhi-qlo).clamp_min(1e-4)*(normal[hi]-normal[lo])
        zs.append(z.clamp(-6,6).cpu())
        if i%100==0:print('CONDITIONAL_SCORE',i,len(d['x']),flush=True)
    z=torch.stack(zs).cuda().flatten(1);z-=z.mean(0);z/=z.std(0).clamp_min(1e-6)
    gram=z@z.T/len(z);lam,u=torch.linalg.eigh(gram);b=(u[:,-32:].T@z)/np.sqrt(len(z));b=b/b.square().sum(0).sqrt().clamp_min(1e-6)
    lib=load(parent/'library.pt','cuda');lib['gaussianized']=z.reshape(-1,2,20,32,64).half();lib['pca']=b.reshape(32,2,20,32,64)
    save(lib,OUT/'library.pt');dump(OUT/'fit.json',dict(seed=42,windows=len(z),pca_rank=32,explained_variance_fraction=float(lam[-32:].sum()/lam.sum()),normalization='per-window fixed XPF marginal inverse in Gaussian quantile coordinates, linear normal tails, clipped ±6, then train-only centering/scaling; covariance is no longer confounded by unconditional marginal scale',adaptation='motivated by failure of raw-residual calibration quality gate, before reading current development results'))

class Generator:
    def __init__(self,method=None):
        compact=(OUT/'compact_generator.pt')
        if method=='ar__mixed' and compact.exists():self.lib=load(compact,'cuda')
        else:
            self.lib=load(OUT/'library.pt','cuda')
            if method is not None and (method.startswith('pca__') or method.startswith('ar__')):
                self.lib={k:v for k,v in self.lib.items() if k in ('pca','stats')}
        eye=torch.eye(32,device='cuda').reshape(32,1,4,8)
        self.local_sd=F.interpolate(eye,(32,64),mode='bilinear',align_corners=True).square().sum(0).sqrt()
        self.ar=load(OUT/'ar_model.pt','cuda') if (OUT/'ar_model.pt').exists() else None
    def ids(self,kind,desc):
        z=st.project(desc,self.lib['projection'][kind]);distance=(self.lib['descriptor'][kind]-z).square().mean(1)
        ids=torch.stack([g[distance[g].argmin()] for g in self.lib['groups']]);return ids[distance[ids].argsort()[:32]]
    def sample(self,name,descriptors,seed=42,return_variance=False):
        kind,loc=name.split('__');g=torch.Generator(device='cuda').manual_seed(seed)
        if kind.startswith('ar'):
            rho=self.ar['mean']
            if kind=='ar_F_last':
                desc=st.project(descriptors['F_last'],self.lib['projection']['F_last'])
                rho=torch.tanh(st.predict(desc,self.ar['weights'])[0]).clamp(-.995,.995)
            z=torch.randn(32,2,20,32,64,device='cuda',generator=g)
            for axis,r in zip((4,3,2),rho[:3]):
                ix=torch.arange(z.shape[axis],device='cuda');cov=r**(ix[:,None]-ix[None]).abs()
                chol=torch.linalg.cholesky(cov)
                z=(z.movedim(axis,-1)@chol.T).movedim(-1,axis)
            z[:,1]=rho[3]*z[:,0]+(1-rho[3]**2).sqrt()*z[:,1]
            if loc=='mixed':
                weights=torch.randn(32,32,device='cuda',generator=g)
                global_part=(weights@self.lib['pca'].flatten(1)).reshape_as(z)
                z=np.sqrt(.75)*z+.5*global_part
            return (z,torch.ones_like(z[0])) if return_variance else z
        if kind=='pca':b=self.lib['pca']
        else:
            ids=self.lib['unconditional'] if kind=='unconditional' else self.ids(kind,descriptors[kind])
            b=self.lib['gaussianized'][ids].float()
            if loc=='schaake':return b
            b=b-b.mean(0,keepdim=True);b=b/b.square().sum(0,keepdim=True).sqrt().clamp_min(1e-6)
        if loc=='global':
            weights=torch.randn(32,32,device='cuda',generator=g);z=(weights@b.flatten(1)).reshape(32,2,20,32,64)
        else:
            a=torch.randn(32*32,1,4,8,device='cuda',generator=g)
            a=(F.interpolate(a,(32,64),mode='bilinear',align_corners=True)/self.local_sd).reshape(32,32,32,64)
            z=torch.einsum('nkyx,kctyx->nctyx',a,b)
        return (z,b.square().sum(0)) if return_variance else z

class Predictor:
    def __init__(self,library=True,method=None):
        self.pred=model();self.heads,self.batches=models_for('XPF');self.n=norm();self.stats=load(SHARED,'cuda')['statistics'];self.generator=Generator(method) if library else None;self.method=method
    def issue(self,x):
        taps={};handles=[]
        for key,name in [('F_mid0','mid'),('F_last','up2')]:
            handles.append(getattr(self.pred.net,name).register_forward_hook(lambda module,args,out,key=key:taps.__setitem__(key,st.pooled(out))))
        try:p,f=feature(self.pred,x,'diffusion')
        finally:
            for handle in handles:handle.remove()
        # Reproduce the existing fixed marginal head preprocessing.
        from reliability.features import base_features
        raw=base_features(x*self.n['std']+self.n['mean'],p*self.n['std']+self.n['mean'],self.n['mean'],self.n['std'],torch.ones(len(x),3,device=x.device))
        base=((raw-self.stats['base_mean'])/self.stats['base_std']).clamp(-10,10)
        last=((f.half().float()-self.stats['last_mean'])/self.stats['last_std']).clamp(-10,10)
        values=quantile_values(all_bounds(self.batches,base,last),p[0]);taps.update(X=st.pooled(x),P=st.pooled(p))
        return p[0],values,taps
    @torch.no_grad()
    def predict(self,x,method,seed=42):
        if self.method is not None and method!=self.method:raise ValueError('Instantiate a predictor for this method')
        p,values,desc=self.issue(x)
        if '__' in method:return couple(values,self.generator.sample(method,desc,seed))
        native=torch.stack([self.pred.rollout(x,500000+k,sign=1)[0] for k in range(32)])
        return baselines(values,native,method)

def baselines(values,native,name):
    ecc=couple(values,native)
    if name=='ecc32':return ecc
    if name=='smooth':
        delta=ecc-ecc.mean(0,keepdim=True);return couple(values,.5*delta+.5*smooth(delta))
    k=int(name[-1])
    sm=smooth(native,k,replicate=False)
    if name.startswith('secc'):return couple(values,sm)
    result=necc(values,sm,k)
    return couple(values,result) if name.startswith('necc_rank') else result

def methods():
    if SCOREFIT_STAGE:return [json.loads((OUT/'fit.json').read_text())['selected']['method']]
    if AR_STAGE:return ['ar__plain','ar__mixed','ar_F_last__plain','ar_F_last__mixed']
    fast=[f'{k}__{l}' for k in ['pca','unconditional']+KINDS for l in ('global','local')]
    fast += ['unconditional__schaake','F_last__schaake']
    return fast

@torch.no_grad()
def fit_ar():
    lib=load(OUT.parent/'library.pt','cuda');z=lib['gaussianized'].float();targets=[]
    for dim in (4,3,2):
        a=z.narrow(dim,0,z.shape[dim]-1).flatten(1);b=z.narrow(dim,1,z.shape[dim]-1).flatten(1)
        targets.append((a*b).mean(1)/(a.square().mean(1)*b.square().mean(1)).sqrt().clamp_min(1e-8))
    a=z[:,0].flatten(1);b=z[:,1].flatten(1);targets.append((a*b).mean(1)/(a.square().mean(1)*b.square().mean(1)).sqrt().clamp_min(1e-8))
    targets=torch.stack(targets,1).clamp(-.995,.995);fisher=torch.atanh(targets)
    weights=st.ridge(lib['descriptor']['F_last'],fisher,1.)
    save(dict(mean=torch.tanh(fisher.mean(0)),weights=weights,training_rho=targets),OUT/'ar_model.pt')
    link=OUT/'library.pt'
    if not link.exists():link.symlink_to(OUT.parent/'library.pt')
    dump(OUT/'fit.json',dict(seed=42,parameters=4,conditional_readout_parameters=int(weights.numel()),rho_order=['x','array_y','time','uv'],unconditional_rho=torch.tanh(fisher.mean(0)).cpu().tolist(),ridge=1.,mixed_pca_variance=.25,description='Stationary separable AR1 Gaussian innovations fitted to train conditional-score autocorrelation; optional F-conditioned four coefficients and fixed25% PCA covariance mixture.'))
    print('AR_FIT',json.loads((OUT/'fit.json').read_text()),flush=True)

def summarize(split):
    rows=[]
    for path in sorted((OUT/split/'parts').glob('*.pt')):
        d=load(path)
        for method,v in d['scores'].items():rows.append(dict(method=method,**d['meta'],**v))
    df=pd.DataFrame(rows);df.to_csv(OUT/split/'windows.csv',index=False)
    cols=[c for c in df if c not in ('method','file','group','start')]
    tr=df.groupby(['method','file','group'])[cols].mean().reset_index();tr.to_csv(OUT/split/'trajectories.csv',index=False)
    summary=tr.groupby('method')[cols].mean();summary.to_csv(OUT/split/'summary.csv');print(summary[['es','vs','functional_crps']],flush=True)
    return summary

@torch.no_grad()
def run(split,limit=None,reuse_controls=False):
    d=load(SOURCE/f'{split}.pt','cuda');pred=Predictor();stats=pred.generator.lib['stats']
    # Keep the full functional scoring code, replacing only its old image-grid curl.
    original=legacy_exp.functionals
    def funcs(z,p,n):
        out=original(z,p,n);a=z*n['std']+n['mean'];v=(a[:,1,:,1:-1,2:]-a[:,1,:,1:-1,:-2])/(2*DX)+(a[:,0,:,2:,1:-1]-a[:,0,:,:-2,1:-1])/(2*DX)
        out[:,2]=v.square().mean((-1,-2)).amax(-1);return out
    legacy_exp.functionals=funcs
    fixed=['ecc32','smooth','secc3','secc9','necc3','necc9','necc_rank3','necc_rank9']
    fast=methods()
    if split!='calibration':
        sel=json.loads((OUT/'selection.json').read_text());fast=list(dict.fromkeys([sel['selected'],'pca__global','unconditional__schaake','F_last__schaake']));fixed=['ecc32','smooth',sel['secc'],sel['necc'],sel['necc_rank']]
    folder=OUT/split/'parts';folder.mkdir(parents=True,exist_ok=True);total=len(d['x']) if limit is None else min(limit,len(d['x']))
    for i in range(total):
        dest=folder/f'{i:04d}.pt'
        if dest.exists():continue
        begin=time.perf_counter();x=d['x'][i:i+1];p,values,desc=pred.issue(x)
        prior=load(OUT.parent/split/'parts'/f'{i:04d}.pt') if reuse_controls else None
        native=None if prior is not None else torch.stack([pred.pred.rollout(x,500000+k,sign=1)[0] for k in range(32)])
        sc={};func={};errors={};samples={}
        for name in fixed+fast:
            if prior is not None and name in fixed:
                sc[name]=prior['scores'][name];func[name]=prior['functionals'][name];errors[name]=prior['marginal_error'][name]
                if i in (0,total-1):
                    audit_path=OUT.parent/split/f'audit_{i:04d}.pt'
                    if audit_path.exists():samples[name]=load(audit_path)['samples'][name]
                continue
            z=baselines(values,native,name) if name in fixed else couple(values,pred.generator.sample(name,desc,seed=420000+i))
            # Every generated field is formed before the scoring function sees y.
            sc[name],func[name]=scores(z,d['y'][i],p,pred.n,stats)
            a=z*pred.n['std']+pred.n['mean'];div=divergence(a)/DX;truthdiv=divergence(d['y'][i:i+1]*pred.n['std']+pred.n['mean'])/DX
            sc[name]['divergence_power']=float(div.square().mean());sc[name]['truth_divergence_power']=float(truthdiv.square().mean())
            errors[name]=float((z.sort(0).values-values).abs().max())
            if i in (0,total-1):samples[name]=z.cpu()
        save(dict(meta=d['meta'][i],scores=sc,functionals=func,marginal_error=errors,seconds=time.perf_counter()-begin),dest)
        if samples:save(dict(samples=samples,values=values.cpu(),p=p.cpu(),y=d['y'][i].cpu(),stats={k:v.cpu() for k,v in stats.items()},norm={k:v.cpu() for k,v in pred.n.items()}),OUT/split/f'audit_{i:04d}.pt')
        if i%10==0:print('DONE',split,i,total,round(time.perf_counter()-begin,2),flush=True)
    s=summarize(split)
    if split=='calibration' and total==len(d['x']):
        cols=['es','vs','functional_crps'];ratios=s[cols]/s.loc['smooth',cols];objective=ratios.mean(1);chosen=objective.loc[fast].idxmin()
        dump(OUT/'selection.json',dict(selected=chosen,secc=objective.loc[['secc3','secc9']].idxmin(),necc=objective.loc[['necc3','necc9']].idxmin(),necc_rank=objective.loc[['necc_rank3','necc_rank9']].idxmin(),objective=objective.to_dict(),quality_gate=(ratios.loc[fast]<=1.02).all(1).to_dict(),criterion='mean ES/VS/physical functional CRPS relative to smooth, calibration only',seed=42))

if __name__=='__main__':
    setup();a=argparse.ArgumentParser();a.add_argument('action',choices=['fit','calibration','development','late','future']);a.add_argument('--limit',type=int);a.add_argument('--conditional-scores',action='store_true');a.add_argument('--ar',action='store_true');a.add_argument('--scorefit',action='store_true');args=a.parse_args()
    if args.conditional_scores or args.ar or args.scorefit:OUT=OUT/'conditional_scores'
    if args.ar:OUT=OUT/'ar';AR_STAGE=True
    if args.scorefit:OUT=OUT/'scorefit';SCOREFIT_STAGE=True
    OUT.mkdir(parents=True,exist_ok=True)
    if args.action=='fit' and args.scorefit:a.error('Use scorefit.py to fit scoring-rule correlation scales')
    if args.action=='fit':fit_ar() if args.ar else (fit_conditional_scores() if args.conditional_scores else fit())
    else:run(args.action,args.limit,reuse_controls=(args.conditional_scores or args.ar or args.scorefit))
