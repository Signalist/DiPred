"""Actual dispatched cost, and replay versus the frozen evaluator."""
from predict import *
import gc

@torch.no_grad()
def main():
    s.setup();pred=BudgetedPredictor();d=s.load(s.SOURCE/'development.pt');ids=[i for i,m in enumerate(d['meta']) if m['start']==0];x=d['x'][ids].cuda();names=[d['meta'][i]['file'] for i in ids];seeds=[420000+i for i in ids]
    obs=s.load(OUT/'development_observables.pt');models=s.load(OUT/'models.pt');sel=json.loads((OUT/'selection.json').read_text());priority=priorities(obs,models,sel)
    cases=[('benefit',q) for q in QUOTAS]+[(p,.5) for p in ('risk','width','random')];rows=[];checks=[]
    original=s.legacy_exp.functionals
    def physical(z,p,n):
        r=original(z,p,n);a=z*n['std']+n['mean'];v=(a[:,1,:,1:-1,2:]-a[:,1,:,1:-1,:-2])/(2*s.DX)+(a[:,0,:,2:,1:-1]-a[:,0,:,:-2,1:-1])/(2*s.DX);r[:,2]=v.square().mean((-1,-2)).amax(-1);return r
    s.legacy_exp.functionals=physical
    low,high=paired_scores('development',obs)
    for policy,q in cases:
        z,info=pred.predict_batch(x,q,policy,True,names,seeds)
        expected=choose([d['meta'][i] for i in ids],priority[policy][ids],q)
        assert np.array_equal(info['expensive'],expected)
        if policy=='benefit' and q==.5:
            for j,i in enumerate(ids):
                audit={k:float(v) for k,v in s.scores(z[j],d['y'][i].cuda(),d['p'][i].cuda(),pred.core.n,pred.core.generator.lib['stats'])[0].items()}
                chosen=high if info['expensive'][j] else low
                error=max(abs(audit[k]-float(chosen[k][i])) for k in METRICS)
                assert error<2e-5,(i,error);checks.append(dict(file=names[j],expensive=bool(info['expensive'][j]),max_score_error=error))
        del z;gc.collect();torch.cuda.empty_cache()
    # Interleave cases so GPU clock/load drift is not confounded with policy.
    measurements={c:[] for c in cases};peaks={c:0. for c in cases};rng=np.random.default_rng(42);order_log=[]
    for repeat in range(7):
        order=rng.permutation(len(cases));order_log.append(order.tolist())
        for j in order:
            policy,q=cases[j];torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
            start=time.perf_counter();z,info=pred.predict_batch(x,q,policy,True,names,seeds);torch.cuda.synchronize()
            measurements[(policy,q)].append(1000*(time.perf_counter()-start));peaks[(policy,q)]=max(peaks[(policy,q)],torch.cuda.max_memory_allocated()/2**20);del z
        print('INTERLEAVED_REPEAT',repeat,flush=True)
    for policy,q in cases:
        times=measurements[(policy,q)];rows.append(dict(policy=policy,quota=q,actual_fraction=q,requests=len(ids),median_batch_ms=float(np.median(times)),median_ms_per_request=float(np.median(times))/len(ids),p10_batch_ms=float(np.quantile(times,.1)),p90_batch_ms=float(np.quantile(times,.9)),peak_mb=peaks[(policy,q)],repeats=7))
        print('TIMING',rows[-1],flush=True)
    s.dump(OUT/'timing_raw.json',dict(order=order_log,cases=cases,times_ms={str(k):v for k,v in measurements.items()},interleaved=True,seed=42))
    pd.DataFrame(rows).to_csv(OUT/'timing.csv',index=False)
    s.dump(OUT/'dispatch_audit.json',dict(passed=True,checks=checks,ranking_matches_feature_cache=True,gpu=torch.cuda.get_device_name(),scope='12 simultaneous requests; serialized GPU service including common forecasts, routing, selected dispatch, samples kept on GPU; no IO/scoring. Mean service cost is batch latency divided by12, not per-user response latency.'))

if __name__=='__main__':main()
