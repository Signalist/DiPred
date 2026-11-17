"""Actual native paths and simple F-marginal ECC; reuse each field for all queries."""
from common import *
from query import queries,target_queries
from risk import RiskHead,features
import time,csv
@torch.no_grad()
def empirical_crps(samples,y):
    n=len(samples);ordered=samples.sort(0).values
    weights=(2*torch.arange(1,n+1,device=samples.device)-n-1).reshape(n,*([1]*y.ndim))
    return (samples-y).abs().mean(0)-(ordered*weights).sum(0)/(n*n)
@torch.no_grad()
def ecc(paths,p,w):
    n=len(paths);order=paths.argsort(0);ranks=order.argsort(0);u=(ranks.to(p.dtype)+.5)/n
    out=p[None].expand_as(paths).clone()
    bs=[u-.5,torch.logit(u)/4]
    for k in (.2,.4,.6,.8):bs.append((torch.nn.functional.softplus(8*(u-k))-math_softplus(8*(.5-k)))/8)
    for k,b in enumerate(bs):out+=w[:,:,k][None]*b
    return out

def math_softplus(a):
    import math
    return math.log1p(math.exp(a))
@torch.no_grad()
def main():
    setup();folder=OUT/'query';d=cache('development');m=model();qs=queries();stats=load(folder/'statistics.pt','cuda');thresholds=stats['thresholds'];std=norm()['std'].flatten().repeat(2)
    ck=load(OUT/'risk/main/F.pt');head=RiskHead(ck['width']).cuda().eval();head.load_state_dict(ck['model'])
    parts=folder/'ensemble_parts';parts.mkdir(exist_ok=True)
    for i in range(0,len(d['x']),3):
        path=parts/f'{i:04d}.pt'
        if path.exists():continue
        ix=torch.arange(i,min(i+3,len(d['x'])),device='cuda');x=d['x'][ix]
        torch.cuda.synchronize();begin=time.perf_counter()
        paths=[];times={}
        for k in range(32):
            paths.append(m.rollout(x,500000+k,sign=1))
            if k in (7,31):torch.cuda.synchronize();times[str(k+1)]=time.perf_counter()-begin
        paths=torch.stack(paths);w=head.weights(*features(d,'F',ix));samples={};qtimes={}
        for count in (8,32):
            native=paths[:count];torch.cuda.synchronize();t=time.perf_counter()
            samples[f'noise{count}']=target_queries(native.flatten(0,1),qs).reshape(count,len(ix),len(qs),4).cpu()
            torch.cuda.synchronize();qtimes[f'noise{count}']=time.perf_counter()-t
            t=time.perf_counter();coupled=ecc(native,d['p'][ix],w)
            samples[f'F_ECC{count}']=target_queries(coupled.flatten(0,1),qs).reshape(count,len(ix),len(qs),4).cpu()
            torch.cuda.synchronize();qtimes[f'F_ECC{count}']=time.perf_counter()-t
        save(dict(samples=samples,indices=ix.cpu(),sampling_seconds=times,query_seconds=qtimes),path)
        print('QUERY_ENSEMBLE',i,times,flush=True)
    methods=('noise8','noise32','F_ECC8','F_ECC32');ys=target_queries(d['y'],qs);rows=[];preds={};timing=[]
    for a in methods:
        files=sorted(parts.glob('*.pt'));chunks=[load(p)['samples'][a] for p in files];sample=torch.cat(chunks,1).cuda();n,b,k,_=sample.shape
        score=empirical_crps(sample,ys)*std;lo,hi=torch.quantile(sample,torch.tensor([.05,.95],device='cuda'),dim=0);cov=((ys>=lo)&(ys<=hi)).float();width=(hi-lo)*std
        prob=(sample[:,:,:,None,2:]>thresholds[None,None,None]).float().mean(0);truth=ys[:,:,None,2:]>thresholds[None,None];brier=(prob-truth.float()).square()
        preds[a]=dict(samples=sample.cpu(),event=prob.cpu())
        score,cov,width,prob,truth,brier=[v.cpu() for v in (score,cov,width,prob,truth,brier)]
        for i,meta in enumerate(d['meta']):
            for j,q in enumerate(qs):
                for c in range(2):
                    for stat in range(2):
                        h=c+stat*2;row=dict(method=a,**meta,query=j,category=q['category'],channel=c,statistic=('mean','max')[stat],crps=float(score[i,j,h]),coverage=float(cov[i,j,h]),width=float(width[i,j,h]))
                        if stat:
                            for t in range(3):row.update({f'brier{t}':float(brier[i,j,t,c]),f'prob{t}':float(prob[i,j,t,c]),f'event{t}':int(truth[i,j,t,c])})
                        rows.append(row)
    for p in sorted(parts.glob('*.pt')):
        v=load(p);timing.append(dict(start=int(v['indices'][0]),batch=len(v['indices']),sampling_seconds=v['sampling_seconds'],query_seconds=v['query_seconds']))
    keys=sorted(set().union(*(r.keys() for r in rows)))
    with (folder/'ensemble_per_query.csv').open('w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=keys);wr.writeheader();wr.writerows(rows)
    save(preds,folder/'ensemble_predictions.pt');dump(folder/'ensemble_timing.json',timing)
if __name__=='__main__':main()
