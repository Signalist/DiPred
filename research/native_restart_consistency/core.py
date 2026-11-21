"""Frozen native sampler, physical-time joins, and distribution/quality statistics."""
import os
os.environ['TORCH_ALLOW_TF32_CUBLAS_OVERRIDE']='0'
import sys,json,csv,hashlib,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'research/residual_dynamics_phase2')]
from common import WORK,DATA,digest,dump,csvwrite
from metrics import empirical_crps,energy_score
from research.residual_dynamics.run import raw,load_backbone
from reliability.backbone import configure
DELIVERY=ROOT/'results/native_restart_consistency'
OUT=Path(os.environ.get('DIPRED_RESTART_OUT',str(DELIVERY)))
CACHE=Path(os.environ.get('DIPRED_RESTART_CACHE',str(WORK/'runs/native_restart_consistency')))
CFG=json.loads((DELIVERY/'config.json').read_text())
CONFIG_HASH=digest(DELIVERY/'config.json')
assert digest(DELIVERY/'PROTOCOL.md')==CFG['protocol_sha256']
RECORDS=CFG['records']
FEATURES=['mean','std','min','max','increment_rms']


def read_csv(path):
    with Path(path).open() as f:return list(csv.DictReader(f))


def tensor_hash(x):return hashlib.sha256(x.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def check_dependencies():
    for name,sha in CFG['dependencies'].items():assert digest(ROOT/name)==sha,name
    assert digest(ROOT/'reliability/checkpoints/current.pt')==CFG['frozen']['packaged_checkpoint_sha256']


def seed_for(split,file,start,repeat,family,k,member):
    payload=[CFG['roots'][repeat],split,file,start,family,k,member]
    return int.from_bytes(hashlib.sha256(json.dumps(payload,separators=(',',':')).encode()).digest()[:8],'little')&((1<<63)-1)


def seeds_for(split,file,start,repeat,family,k):
    return [seed_for(split,file,start,repeat,family,k,i) for i in range(CFG['M'])]


def seeded_noise(history,seeds):
    assert len(history)==len(seeds) and len(set(seeds))==len(seeds)
    return torch.cat([torch.randn((1,*history.shape[1:]),dtype=history.dtype,device=history.device,
             generator=torch.Generator(device=history.device).manual_seed(seed)) for seed in seeds])


@torch.no_grad()
def sample_noise(model,history,noise):
    assert history.shape==noise.shape and history.shape[1:]==(2,20,32,64)
    x=noise.clone()
    for step in reversed(range(4)):
        alpha=model.abar[step]
        labels=torch.full((len(history),),step*33,device=history.device,dtype=torch.long)
        clean=model.net(x,labels,history)
        residual_noise=(x-alpha.sqrt()*clean)/(1-alpha).sqrt()
        previous=model.abar[step-1] if step else x.new_tensor(1.)
        x=previous.sqrt()*clean+(1-previous).sqrt()*residual_noise
    return x


@torch.no_grad()
def generate(model,norm,history,seeds):
    h=(history-norm['mean'])/norm['std'];noise=seeded_noise(h,seeds)
    hashes=[tensor_hash(v) for v in noise]
    assert len(set(hashes))==len(hashes)
    torch.cuda.synchronize();before=time.perf_counter()
    result=sample_noise(model,h,noise)*norm['std']+norm['mean']
    torch.cuda.synchronize()
    return result,dict(seeds=seeds,noise_sha256=hashes,noise_shape=list(noise.shape),history_sha256=tensor_hash(history),output_sha256=tensor_hash(result),seconds=time.perf_counter()-before,network_calls=4,samples=len(history))


def restart_history(history,parent,k):
    # history [C,20,H,W], parent [M,C,20,H,W]; no truth accepted.
    assert 0<=k<20 and history.shape[1]==20 and parent.shape[2]==20
    original=history[None].expand(len(parent),-1,-1,-1,-1)
    return torch.cat([original[:,:,k:],parent[:,:,:k]],2)


def tails(direct,child,k):return direct[:,:,k:],child[:,:,:20-k]


def physical_indices(start,k):
    return dict(history=list(range(start,start+20)),direct=list(range(start+20,start+40)),
                restart_history=list(range(start+k,start+20+k)),child=list(range(start+20+k,start+40+k)),
                compared=list(range(start+20+k,start+40)))


def distances(x,y):
    a=x.double().flatten(1);b=y.double().flatten(1)
    squared=(a.square().sum(1)[:,None]+b.square().sum(1)[None]-2*a@b.T)/a.shape[1]
    result=squared.clamp_min(0).sqrt()
    if torch.equal(a,b):result.fill_diagonal_(0)
    return result


def energy_distance(x,y):
    n,m=len(x),len(y);assert n>1 and m>1
    cross=distances(x,y).mean();xx=distances(x,x);yy=distances(y,y)
    return dict(u=float(2*cross-xx.sum()/(n*(n-1))-yy.sum()/(m*(m-1))),
                v=float(2*cross-xx.mean()-yy.mean()))


def projections(paths):
    # Fixed channel×quadrant×(time mean, late-early); standardized input.
    _,_,length,h,w=paths.shape
    fields=[]
    for ch in range(2):
        for ys in (slice(0,h//2),slice(h//2,h)):
            for xs in (slice(0,w//2),slice(w//2,w)):
                v=paths[:,ch,:,ys,xs].mean((-1,-2));mid=length//2
                fields.extend([v.mean(1),v[:,mid:].mean(1)-v[:,:mid].mean(1)])
    return torch.stack(fields,1)


def covariance(x):
    c=x-x.mean(0);return c.T@c/(len(c)-1)


def discrepancy(b,s):
    ed=energy_distance(b,s);bc=b-b.mean(0);sc=s-s.mean(0)
    bp,sp=projections(b),projections(s);cb,cs=covariance(bp),covariance(sp)
    db=cb.diag().clamp_min(0).sqrt();ds=cs.diag().clamp_min(0).sqrt()
    rb=cb/(db[:,None]*db[None,:]+1e-12);rs=cs/(ds[:,None]*ds[None,:]+1e-12)
    return dict(ed_u=ed['u'],ed_v=ed['v'],mean_rms=float((b.mean(0)-s.mean(0)).square().mean().sqrt()),
        centered_ed_u=energy_distance(bc,sc)['u'],variance_delta=float(s.var(0,unbiased=False).mean()-b.var(0,unbiased=False).mean()),
        projection_cov_difference=float((cb-cs).square().sum().sqrt()/len(cb)),
        projection_correlation_difference=float((rb-rs).square().sum().sqrt()/len(cb)))


@torch.no_grad()
def quality(paths,truth,std):
    # [M,C,T,H,W], truth[C,T,H,W], std[2,1,1,1].
    p=paths.double();y=truth.double();pn=p/std;yn=y/std
    error=p.mean(0)-y
    full=empirical_crps(pn,yn)
    variance=p.var(0,unbiased=False);nvar=pn.var(0,unbiased=False)
    width=torch.quantile(p,.95,dim=0)-torch.quantile(p,.05,dim=0)
    time_crps=empirical_crps(pn.mean(2),yn.mean(1))
    rows=[];cells=[]
    for ch,name in [(None,'overall'),(0,'u'),(1,'v')]:
        sl=slice(None) if ch is None else slice(ch,ch+1)
        rows.append(dict(channel=name,crps=float(full[sl].mean()),energy=float(energy_score(pn[:,sl],yn[sl])),
            mse=float(error[sl].square().mean()),smse=float((error/std)[sl].square().mean()),
            variance=float(variance[sl].mean()),standardized_variance=float(nvar[sl].mean()),
            width90=float(width[sl].mean()),standardized_width90=float((width/std)[sl].mean()),
            time_mean_crps=float(time_crps[sl].mean())))
    for ch,name in enumerate('uv'):
        for lead in range(y.shape[1]):cells.append(dict(channel=name,tail_lead=lead+1,crps=float(full[ch,lead].mean()),mse=float(error[ch,lead].square().mean()),smse=float((error/std)[ch,lead].square().mean()),variance=float(variance[ch,lead].mean()),width90=float(width[ch,lead].mean())))
    return rows,cells


def history_features(h,norm):
    x=((h-norm['mean'])/norm['std']).double()
    return torch.stack([x.mean((2,3,4)),x.std((2,3,4),unbiased=False),x.amin((2,3,4)),x.amax((2,3,4)),x.diff(dim=2).square().mean((2,3,4)).sqrt()],-1)


def aggregate(rows,keys,metrics):
    from collections import defaultdict
    groups=defaultdict(list)
    for row in rows:groups[tuple(row[k] for k in keys)].append(row)
    return [dict(zip(keys,key),**{m:float(np.mean([float(r[m]) for r in rr])) for m in metrics}) for key,rr in groups.items()]
