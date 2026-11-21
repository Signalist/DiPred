"""Coupled oracle/mean/member input interventions; frozen, differentiable sampler."""
import os
os.environ['TORCH_ALLOW_TF32_CUBLAS_OVERRIDE']='0'
import sys,json,time,hashlib,csv
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import h5py
import numpy as np
import torch
from research.native_restart_consistency import core as old
from reliability.backbone import configure
OUT=Path(os.environ.get('DIPRED_ATTRIBUTION_OUT',str(ROOT/'results/restart_input_attribution')))
DELIVERY=ROOT/'results/restart_input_attribution'
CACHE=Path(os.environ.get('DIPRED_ATTRIBUTION_CACHE',str(old.WORK/'runs/restart_input_attribution')))
CFG=json.loads((DELIVERY/'config.json').read_text())
digest,dump,csvwrite,tensor_hash=old.digest,old.dump,old.csvwrite,old.tensor_hash
QUAL=['crps','energy','mse','smse','variance','standardized_variance','width90','standardized_width90','time_mean_crps']
META=['split','trajectory','Re','start','repeat','k']


def check():
    assert digest(DELIVERY/'PROTOCOL.md')==CFG['protocol_sha256']
    for p,h in CFG['dependencies'].items():assert digest(ROOT/p)==h,p


def model_load():
    model,norm,frozen=old.load_backbone(SimpleNamespace(workspace=old.WORK))
    assert frozen==CFG['frozen'] and not model.training
    assert all(not p.requires_grad for p in model.parameters())
    return model,norm


def sampler(model,history,noise):
    """All four steps and history.diff remain on the autograd graph."""
    assert history.shape==noise.shape
    x=noise.clone()
    for step in reversed(range(4)):
        alpha=model.abar[step]
        labels=torch.full((len(history),),step*33,device=history.device,dtype=torch.long)
        clean=model.net(x,labels,history)
        residual=(x-alpha.sqrt()*clean)/(1-alpha).sqrt()
        previous=model.abar[step-1] if step else x.new_tensor(1.)
        x=previous.sqrt()*clean+(1-previous).sqrt()*residual
    return x


def frames(file,begin,end,data=old.DATA):
    with h5py.File(data/file,'r') as f:
        a=np.stack([f[c][begin:end,::2,::2] for c in ('u','v')]).astype('float32')
    assert a.shape==(2,end-begin,32,64) and np.isfinite(a).all()
    return torch.from_numpy(a)


def oracle_prefix(file,start,k,data=old.DATA):
    assert k in CFG['k']
    return frames(file,start+20,start+20+k,data)


def inputs(history,parent,prefix,k):
    # Full20 labels are rejected. No scoring target is accepted by this API.
    assert prefix.shape==(2,k,32,64) and history.shape==(2,20,32,64)
    assert parent.shape==(32,2,20,32,64)
    r=torch.cat([history[:,k:],prefix],1)
    m=torch.cat([history[:,k:],parent[:,:,:k].mean(0)],1)
    s=old.restart_history(history,parent,k)
    assert torch.equal(s[:,:,:20-k],r[None,:,:20-k].expand(32,-1,-1,-1,-1))
    return r,m,s


def normalize(x,norm):return (x-norm['mean'])/norm['std']
def decode(x,norm):return x*norm['std']+norm['mean']


def noise_for(bundle,k,history):
    call=bundle['calls']['S'+str(k)]
    z=old.seeded_noise(history,call['seeds'])
    assert [tensor_hash(v) for v in z]==call['noise_sha256']
    assert not set(call['seeds'])&set(bundle['calls']['A']['seeds'])
    return z


def entries(split=None):
    m=json.loads((old.DELIVERY/'cache_manifest.json').read_text())
    return [r for r in m if r['split']==split] if split else m


def load_bundle(entry):
    p=Path(entry['path']);assert digest(p)==entry['sha256'],str(p)
    b=torch.load(p,map_location='cpu',weights_only=False)
    assert tensor_hash(b['history'])==entry['history_sha256']
    for key,h in entry['outputs'].items():assert tensor_hash(b['paths'][key])==h
    for key in ['split','trajectory','Re','start','repeat']:assert b['identity'][key]==entry[key]
    return b


def rms(x):return float(x.double().square().mean().sqrt())


def tangent_metrics(linear,actual,e,k):
    linear,actual,e=linear.double(),actual.double(),e.double()
    a,b,err=rms(linear),rms(actual),rms(linear-actual);inp=rms(e)
    zero=min(a,b,inp)<CFG['near_zero']
    return dict(input_rms=inp,prefix_rms=rms(e[:,-k:]),linear_rms=a,response_rms=b,error_rms=err,
        relative_error=None if b<CFG['near_zero'] else err/b,
        cosine=None if min(a,b)<CFG['near_zero'] else float((linear*actual).mean())/(a*b),
        linear_gain=None if inp<CFG['near_zero'] else a/inp,
        response_gain=None if inp<CFG['near_zero'] else b/inp,near_zero=int(zero))


def response_rows(r,m,s,std):
    r,m,s=[x.double()/std for x in (r,m,s)]
    d,dm,dd=s-r,m-r,s-m
    assert float((d-dm-dd).abs().max())<1e-12
    rows=[]
    for i in range(len(r)):
        for ch,name in [(slice(None),'overall'),(slice(0,1),'u'),(slice(1,2),'v')]:
            x,y,z=d[i,ch],dm[i,ch],dd[i,ch]
            a,b,c=[float(v.square().mean()) for v in (x,y,z)];cross=float(2*(y*z).mean())
            rows.append(dict(member=i,channel=name,D_rms=a**.5,Dm_rms=b**.5,Dd_rms=c**.5,D_square=a,Dm_square=b,Dd_square=c,cross=cross,mean_minus_member_square=b-c,identity_error=a-b-c-cross))
    ensemble=[]
    for ch,name in [(slice(None),'overall'),(slice(0,1),'u'),(slice(1,2),'v')]:
        x,y,z=[v.mean(0)[ch] for v in (d,dm,dd)]
        ensemble.append(dict(channel=name,D_rms=rms(x),Dm_rms=rms(y),Dd_rms=rms(z),D_square=float(x.square().mean()),Dm_square=float(y.square().mean()),Dd_square=float(z.square().mean()),cross=float(2*(y*z).mean())))
    return rows,ensemble


def signed_rows(r,s,truth,std):
    out=[]
    for scale,scaleval in [('physical',1.),('standardized',std)]:
        mu_r,mu_s,y=r.double().mean(0)/scaleval,s.double().mean(0)/scaleval,truth.double()/scaleval
        d,e=mu_s-mu_r,mu_r-y
        for ch,name in [(slice(None),'overall'),(slice(0,1),'u'),(slice(1,2),'v')]:
            delta=float(((mu_s-y)[ch].square()-e[ch].square()).mean())
            cross=float(2*(e[ch]*d[ch]).mean());square=float(d[ch].square().mean())
            out.append(dict(scale=scale,channel=name,mse_difference=delta,error_response_cross=cross,response_square=square,identity_error=delta-cross-square))
    return out


def event(action,**kw):
    with (OUT/'events.jsonl').open('a') as f:f.write(json.dumps(dict(action=action,utc=time.time(),**kw))+'\n')


def code_hashes():return {str(p.relative_to(ROOT)):digest(p) for p in Path(__file__).parent.glob('*.py')}
