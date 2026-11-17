"""Batch-one latency, loaded models, full field ensembles reused over queries."""
from common import *
from query import QueryHead,queries,inputs,mask,target_queries
import time,csv
@torch.no_grad()
def main():
    setup();d=cache('development');x=d['x'][:1];predictor=model();fstats=load(OUT/'feature_stats.pt','cuda');stats=load(OUT/'query/statistics.pt','cuda');levels=torch.tensor([.05,.5,.95],device='cuda');models={}
    for a in ('X','P','F','XPF'):
        h=QueryHead().cuda().eval();h.load_state_dict(load(OUT/'query'/f'{a}.pt')['model']);models[a]=h
    rows=[];qs=queries()
    for count in (1,8,32,160):
        selected=qs[:count]
        for arm in ('X','P','F','XPF','noise8','noise32'):
            times=[]
            for repeat in range(12):
                torch.cuda.synchronize();start=time.perf_counter()
                if arm.startswith('noise'):
                    size=int(arm[5:]);fields=torch.cat([predictor.rollout(x,500000+i,sign=1) for i in range(size)],0)
                    value=target_queries(fields,selected)
                    # Same three output quantiles; fields reused across all queries.
                    out=torch.quantile(value,levels,dim=0)
                else:
                    p=torch.zeros_like(x);f=torch.zeros(1,48,32,64,device='cuda')
                    if arm!='X':
                        p,f=feature(predictor,x,'diffusion');mu,sd=fstats['f'];f=((f-mu)/sd).clamp(-10,10)
                    z=inputs(dict(x=x,p=p,f=f),selected).flatten(0,1);z[:,:384]=((z[:,:384]-stats['mean'])/stats['std']).clamp(-10,10)
                    out=models[arm].quantiles(z,mask(arm),levels)
                torch.cuda.synchronize();elapsed=time.perf_counter()-start
                if repeat>=2:times.append(elapsed*1000)
            rows.append(dict(method=arm,queries=count,median_ms=float(np.median(times)),p10_ms=float(np.quantile(times,.1)),p90_ms=float(np.quantile(times,.9)),batch=1,repeats=10,outputs='05/50/95 quantiles of both query statistics and both channels'))
            print('BENCHMARK',rows[-1],flush=True)
    with (OUT/'query/latency.csv').open('w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
    dump(OUT/'query/latency_protocol.json',dict(device=torch.cuda.get_device_name(),batch=1,repeats=10,warmup=2,included='history->actual frozen DiPred when needed->pooling->query head or reused ensemble->three quantiles',excluded='model/data loading, transfer to CPU, feedback',ensemble_reused_for_all_queries=True,warning='Single host measurements; no throughput or universal speedup claim. Background GPU keepalive may exist.'))
if __name__=='__main__':main()
