"""Bounded phase-three diagnosis. Run from the DiPred checkout with Workspace/env.sh."""
import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

repo=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(repo/'research/residual_dynamics_phase2'),str(repo)]
from common import (OLD,EXTRA,SEEDS,protocol,read,dump,digest,csvwrite,bootstrap,aux)
from causal import BiasHead,inputs
from mechanisms import fit_linear,simple_predict
from metrics import empirical_crps,energy_score,consecutive_event
from reliability.backbone import configure

out=repo/'results/residual_dynamics_phase3'
ref=repo/'results/residual_dynamics_phase2'
records=protocol()['records']
threshold=read(ref/'causal_training_protocol.json')['event_threshold']
trace=[]
selection=[]
long=[]
blocks=[]
innov=[]
models={}


def load(split):
    f=aux(split)
    ckpath=repo/'reliability/checkpoints/current.pt'
    assert digest(ckpath)==protocol()['frozen']['packaged_checkpoint_sha256']
    norm=torch.load(ckpath,map_location='cuda',weights_only=False)['normalization']
    for k in ('mean','std'):assert torch.equal(f['normalization'][k],norm[k])
    paths=[]; features=[]
    for j,start in enumerate((400,420)):
        file=EXTRA/f'{split}_{start}_native.pt'
        a=next(r for r in read(ref/'alignment_audit.json')['entries'] if (r['split'],r['start'])==(split,start))
        assert digest(file)==a['native_sha256']
        v=torch.load(file,map_location='cuda',weights_only=False)
        prior=next(r for r in read(OLD/'cache_manifest.json') if r['identity']['split']==split and r['identity']['start']==start)
        assert v['identity']==prior['identity'] and v['identity']['files']==[r['file'] for r in records[split]]
        assert v['identity']['seeds']==list(range(500000,500032)) and v['identity']['frozen']==protocol()['frozen']
        for field,key in [('history','history_sha256'),('truth','truth_sha256')]:
            assert hashlib.sha256(f[field][:,j].contiguous().cpu().numpy().tobytes()).hexdigest()==prior[key]
        p=v['paths']
        features.append(inputs(f['history'][:,j],p,None,'current_fields',norm).reshape(len(p),20,20,32,64))
        paths.append(p.permute(0,3,1,2,4,5))
    p=torch.cat(paths)
    y=torch.cat([f['truth'][:,j].transpose(1,2) for j in range(2)])
    std=norm['std'].reshape(2,1,1)
    return SimpleNamespace(p=p,y=y,b=p.mean(2)-y,x=torch.cat(features),std=std,split=split)


def shifted_loss(p,y,q,std):
    return ((p-y.unsqueeze(1)-q.unsqueeze(1))/std).abs().mean()


def tests():
    torch.manual_seed(137)
    p=torch.randn(32,2,4,5,dtype=torch.double)
    y=torch.randn_like(p[0]); std=torch.tensor([.11,.017]).reshape(2,1,1)
    q=torch.randn_like(y,requires_grad=True)
    exact=empirical_crps((p-q)/std,y/std).mean()
    reduced=shifted_loss(p[None],y[None],q[None],std)
    pair=.5*((p[:,None]-p[None,:])/std).abs().mean()
    assert torch.allclose(exact,reduced-pair,atol=1e-12)
    assert torch.allclose(torch.autograd.grad(exact,q)[0],torch.autograd.grad(reduced,q)[0],atol=1e-12)
    print('PASS full empirical CRPS value/gradient against explicit 32x32 pair constant',flush=True)


def propagate(b,coef,mode,initial=None):
    # b shape provides the block geometry. Only teacher and first access its values.
    state=b[:,0] if mode=='first' else (initial if initial is not None else torch.zeros_like(b[:,0]))
    seq=[state]
    for t in range(1,20):
        x=b[:,t-1] if mode=='teacher' else state
        state=simple_predict(x,None,coef)
        seq.append(state)
    return torch.stack(seq,1)


@torch.no_grad()
def score(d,q,name,seed=0):
    crps=[]; mse=[]
    for i,(p,y,shift) in enumerate(zip(d.p,d.y,q)):
        r=records[d.split][i%len(records[d.split])]
        meta=dict(split=d.split,trajectory=r['file'],Re=r['group'],start=400+20*(i//len(records[d.split])),method=name,seed=seed)
        pred=(p-shift[:,None]).transpose(0,1).double()
        truth=y.double()
        c=empirical_crps(pred/d.std,truth/d.std).mean((-1,-2)).cpu().numpy()
        m=(pred.mean(0)-truth).square().mean((-1,-2)).cpu().numpy()
        es=float(energy_score(pred/d.std,truth/d.std))
        observed=int(consecutive_event(truth.transpose(0,1)[None],threshold)[0])
        prob=float(consecutive_event(pred.transpose(1,2),threshold).double().mean())
        blocks.append(dict(**meta,mse=m.mean(),crps=c.mean(),energy=es,event=observed,brier=(prob-observed)**2))
        for t in range(20):
            for k,ch in enumerate('uv'):
                long.append(dict(**meta,lead=t+1,ch=ch,mse=m[t,k],crps=c[t,k]))
        crps.append(c.mean());mse.append(m.mean())
    return float(np.mean(crps)),float(np.mean(mse))


def stage_a(d,coefs):
    results={}
    for kind,coef in coefs.items():
        for mode in ('teacher','first','zero'):
            q=propagate(d.b,coef,mode)
            err=d.b-q
            results[kind,mode]=float(err[:,1:].square().mean())
            score(d,q,kind+'_'+mode)
            if mode=='teacher':
                for i,v in enumerate(err[:,1:]):
                    for k in range(2):
                        a=v[:-1,k].flatten().double();b=v[1:,k].flatten().double()
                        val=float(torch.corrcoef(torch.stack([a,b]))[0,1])
                        innov.append(dict(split=d.split,trajectory=records[d.split][i%len(records[d.split])]['file'],
                            window=i//len(records[d.split]),method=kind,ch='uv'[k],lag1=val))
    return results


class Structured(nn.Module):
    """Same local space/time encoder; T integrates conditional increments with learned decay."""
    def __init__(self,recursive):
        super().__init__()
        self.encoder=nn.Sequential(nn.Conv3d(20,12,3,padding=1),nn.SiLU(),nn.Conv3d(12,2,1))
        nn.init.zeros_(self.encoder[-1].weight);nn.init.zeros_(self.encoder[-1].bias)
        self.recursive=recursive
        if recursive:self.decay=nn.Parameter(torch.ones(2,1,1))
    def forward(self,x):
        v=self.encoder(x.transpose(1,2)).transpose(1,2)
        if not self.recursive:return v
        seq=[v[:,0]]
        for t in range(1,20):seq.append(self.decay.sigmoid()*seq[-1]+v[:,t])
        return torch.stack(seq,1)


def output(m,d,scale,kind):
    with torch.no_grad():
        if kind in ('D','T'):
            return torch.cat([m(v)*scale for v in d.x.split(2)])
        if kind=='init':return m(d.x[:,0])*scale
        return torch.cat([m(v)*scale for v in d.x.flatten(0,1).split(32)]).reshape_as(d.b)


def train(train,dev,kind,seed):
    structured=kind in ('D','T')
    torch.manual_seed(seed)
    m=(Structured(kind=='T') if structured else BiasHead()).cuda()
    scale=float(train.b.square().mean().sqrt())
    opt=torch.optim.Adam(m.parameters(),lr=.001)
    rng=torch.Generator(device='cuda').manual_seed(20261121)
    def criterion(q):
        if kind=='init':return float((dev.b[:,0]-q).square().mean())
        # Constant pair term cancels for checkpoint comparisons.
        return float(torch.stack([shifted_loss(p,y,z,dev.std) for p,y,z in zip(dev.p,dev.y,q)]).mean())
    best=criterion(output(m,dev,scale,kind)); baseline=best
    beststate={k:v.clone() for k,v in m.state_dict().items()}; beststep=0
    budget=500 if structured else 1000
    for step in range(1,budget+1):
        n=2 if structured else 16
        idx=torch.randint(len(train.x),(n,),generator=rng,device='cuda')
        if structured:
            q=m(train.x[idx])*scale
            loss=shifted_loss(train.p[idx].flatten(0,1),train.y[idx].flatten(0,1),q.flatten(0,1),train.std)
        else:
            t=torch.randint(20,(n,),generator=rng,device='cuda')
            if kind=='init':t.zero_()
            q=m(train.x[idx,t])*scale
            diff=train.b[idx,t]-q
            if kind in ('raw','init'):loss=(diff/scale).square().mean()
            elif kind=='std':loss=(diff/train.std).square().mean()
            else:loss=shifted_loss(train.p[idx,t],train.y[idx,t],q,train.std)
        opt.zero_grad(set_to_none=True);loss.backward();opt.step()
        if step%(budget//4)==0:
            value=criterion(output(m,dev,scale,kind))
            trace.append(dict(kind=kind,seed=seed,step=step,development=value))
            if value<best:
                best=value;beststep=step;beststate={k:v.clone() for k,v in m.state_dict().items()}
    dest=out/'weights';dest.mkdir(exist_ok=True)
    file=dest/f'{kind}_{seed}.pt'
    torch.save(dict(selected=beststate,endpoint=m.state_dict(),scale=scale,step=beststep,kind=kind),file)
    selection.append(dict(kind=kind,seed=seed,params=sum(v.numel() for v in m.parameters()),step=beststep,
        strength=int(beststep>0),zero_dev=baseline,best_dev=best,sha256=digest(file)))
    print('selected',kind,seed,beststep,best-baseline,flush=True)
    m.load_state_dict(beststate)
    return m,scale


def main():
    configure();tests()
    tr,dev=load('train'),load('development')
    # fit_linear expects trajectory/window/member/lead/channel/y/x
    b=tr.b[:,None,None]
    coefs={'copy':dict(kind='A1_persistence'),'scale':fit_linear(b,'A2_scaled'),'linear':fit_linear(b,'linear3x3')}
    dump(out/'coefficients.json',{k:{n:v.tolist() if torch.is_tensor(v) else v for n,v in p.items()} for k,p in coefs.items()})
    a=stage_a(dev,coefs)
    gate=(a['linear','first'] < min(a['copy','first'],.95*a['linear','zero']) and a['linear','teacher']<a['copy','teacher'])
    print('initialization gate',gate,a,flush=True)
    native=score(dev,torch.zeros_like(dev.b),'native')
    def fit_set(kinds):
        result=[]
        for k in kinds:
            for seed in SEEDS:
                m,s=train(tr,dev,k,seed);models[k,seed]=(m,s)
                q=output(m,dev,s,k)
                if k=='init':
                    for op,coef in coefs.items():score(dev,propagate(dev.b,coef,'estimated',q),'init_'+op,seed)
                else:
                    v=score(dev,q,k,seed)
                    if k=='crps':result.append(v)
        return result
    val=fit_set(['raw','std','crps']+(['init'] if gate else []))
    cgate=bool(gate and np.all(np.mean(val,0)<np.array(native)))
    dump(out/'gates.json',dict(A=bool(gate),C=cgate,dev_native=native,dev_crps_arm=val))
    if cgate:fit_set(['D','T'])
    test=load('test');stage_a(test,coefs);score(test,torch.zeros_like(test.b),'native')
    for (kind,seed),(m,s) in models.items():
        q=output(m,test,s,kind)
        if kind=='init':
            for op,coef in coefs.items():score(test,propagate(test.b,coef,'estimated',q),'init_'+op,seed)
        else:score(test,q,kind,seed)
        if kind in ('raw','std','crps'):
            ck=torch.load(out/'weights'/f'{kind}_{seed}.pt',weights_only=False)
            m.load_state_dict(ck['endpoint'])
            score(test,output(m,test,s,kind),kind+'_endpoint',seed)
    for name,rows in [('scores',blocks),('lead_channel',long),('innovation',innov),('selection',selection),('training',trace)]:
        csvwrite(out/(name+'.csv'),rows)
    print('completed',len(blocks),len(long),flush=True)


if __name__=='__main__':main()
