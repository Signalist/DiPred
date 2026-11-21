"""Fixed-k trajectory statistics and prespecified descriptive associations."""
from collections import defaultdict
from scipy.stats import rankdata
from core import *

DIST=['restart_ed_u','sampling_ed_u','excess_ed_u','restart_ed_v','sampling_ed_v','excess_ed_v','restart_mean_rms','sampling_mean_rms','excess_mean_rms','restart_centered_ed_u','sampling_centered_ed_u','excess_centered_ed_u','restart_variance_delta','sampling_variance_delta','excess_variance_delta','restart_projection_cov_difference','sampling_projection_cov_difference','excess_projection_cov_difference','restart_projection_correlation_difference','sampling_projection_correlation_difference','excess_projection_correlation_difference','native_spread','native_temporal_rms']
QUAL=['crps','energy','mse','smse','variance','standardized_variance','width90','standardized_width90','time_mean_crps']


def interval(values):
    a=np.asarray(values,dtype=float);rng=np.random.default_rng(123)
    samples=a[rng.integers(len(a),size=(10000,len(a)))].mean(1)
    q=np.quantile(samples,[.025,.975,.05/6,1-.05/6])
    return dict(mean=float(a.mean()),ci95_low=float(q[0]),ci95_high=float(q[1]),ci98_333_low=float(q[2]),ci98_333_high=float(q[3]),positive=int((a>0).sum()),negative=int((a<0).sum()),trajectories=len(a))


def intervals(rows,keys,metrics):
    tr=aggregate(rows,[*keys,'trajectory'],metrics);groups=defaultdict(list)
    for r in tr:groups[tuple(r[k] for k in keys)].append(r)
    result=[]
    for key,rr in groups.items():
        rr=sorted(rr,key=lambda r:r['trajectory'])
        for metric in metrics:result.append(dict(zip(keys,key),metric=metric,**interval([r[metric] for r in rr])))
    return result


def spearman(x,y):
    rx=rankdata(x,axis=-1);ry=rankdata(y,axis=-1)
    rx=rx-rx.mean(axis=-1,keepdims=True);ry=ry-ry.mean(axis=-1,keepdims=True)
    denom=np.sqrt((rx*rx).sum(-1)*(ry*ry).sum(-1))
    return np.divide((rx*ry).sum(-1),denom,out=np.full_like(denom,np.nan,dtype=float),where=denom>0)


def associations(dist,quality):
    dt=aggregate(dist,['split','k','trajectory'],DIST)
    qt=aggregate([r for r in quality if r['channel']=='overall'],['split','k','trajectory','method'],QUAL)
    lookup={(r['split'],r['k'],r['trajectory'],r['method']):r for r in qt};groups=defaultdict(list)
    for r in dt:groups[r['split'],r['k']].append(r)
    result=[]
    for (split,k),rows in groups.items():
        rows=sorted(rows,key=lambda r:r['trajectory']);n=len(rows);rng=np.random.default_rng(123);idx=rng.integers(n,size=(10000,n))
        for predictor in ['excess_ed_u','native_spread','native_temporal_rms']:
            x=np.array([r[predictor] for r in rows])
            for target in ['B_crps','B_smse','S_minus_B_crps']:
                y=[]
                for r in rows:
                    b=lookup[split,k,r['trajectory'],'B'];s=lookup[split,k,r['trajectory'],'S']
                    y.append(s['crps']-b['crps'] if target=='S_minus_B_crps' else b[target[2:]])
                y=np.array(y);rho=float(spearman(x,y));boot=spearman(x[idx],y[idx]);valid=boot[np.isfinite(boot)]
                lo,hi=np.quantile(valid,[.025,.975]) if len(valid) else [np.nan,np.nan]
                result.append(dict(split=split,k=k,predictor=predictor,target=target,rho=rho,ci95_low=float(lo),ci95_high=float(hi),trajectories=n,valid_bootstrap=len(valid)))
    return result


def main():
    assert (OUT/'COMPLETE.json').exists(),'Incomplete fixed list must be reported as incomplete, not silently subset-analyzed'
    d=read_csv(OUT/'distribution.csv');q=read_csv(OUT/'quality.csv');h=read_csv(OUT/'history_stats.csv')
    assert len(d)==840 and len(q)==10080
    for keys,name in [(['split','k'],'distribution_summary'),(['split','k','repeat'],'distribution_by_repeat'),(['split','k','Re'],'distribution_by_Re'),(['split','k','trajectory','Re'],'distribution_by_trajectory')]:csvwrite(OUT/(name+'.csv'),aggregate(d,keys,DIST))
    for keys,name in [(['split','k'],'distribution_intervals'),(['split','k','repeat'],'distribution_intervals_by_repeat'),(['split','k','Re'],'distribution_intervals_by_Re')]:csvwrite(OUT/(name+'.csv'),intervals(d,keys,DIST))
    for keys,name in [(['split','k','method','channel'],'quality_summary'),(['split','k','repeat','method','channel'],'quality_by_repeat'),(['split','k','Re','method','channel'],'quality_by_Re'),(['split','k','trajectory','Re','method','channel'],'quality_by_trajectory')]:csvwrite(OUT/(name+'.csv'),aggregate(q,keys,QUAL))
    index={(r['split'],r['trajectory'],r['start'],r['repeat'],r['k'],r['channel'],r['method']):r for r in q};delta=[]
    for r in q:
        if r['method']!='S':continue
        for base in ['A','B','C']:
            other=index[r['split'],r['trajectory'],r['start'],r['repeat'],r['k'],r['channel'],base]
            delta.append(dict({k:r[k] for k in ['split','trajectory','Re','start','repeat','k','channel']},contrast='S minus '+base,**{m:float(r[m])-float(other[m]) for m in QUAL}))
    csvwrite(OUT/'quality_differences.csv',delta)
    csvwrite(OUT/'quality_paired.csv',intervals(delta,['split','k','channel','contrast'],QUAL))
    csvwrite(OUT/'quality_paired_by_Re.csv',intervals(delta,['split','k','channel','contrast','Re'],QUAL))
    csvwrite(OUT/'quality_paired_by_repeat.csv',intervals(delta,['split','k','channel','contrast','repeat'],QUAL))
    csvwrite(OUT/'associations.csv',associations(d,q))
    hm=FEATURES+['max_abs_z','fraction_abs_z_above3']+[name+'_z' for name in FEATURES]
    # Pseudo parents are averaged inside each trajectory/window/repeat first.
    hist=aggregate(h,['split','trajectory','Re','start','repeat','k','channel'],hm)
    csvwrite(OUT/'history_by_trajectory_window.csv',hist)
    csvwrite(OUT/'history_summary.csv',aggregate(hist,['split','k','channel'],hm))
    csvwrite(OUT/'history_by_Re.csv',aggregate(hist,['split','Re','k','channel'],hm))
    leads=read_csv(OUT/'lead_channel.csv')
    csvwrite(OUT/'lead_channel_summary.csv',aggregate(leads,['split','k','method','channel','tail_lead'],['crps','mse','smse','variance','width90']))
    # Fixed criterion, no model selection or new sampling is permitted.
    ci=intervals(d,['split','k'],['excess_ed_u']);rep=aggregate(d,['split','k','repeat'],['excess_ed_u']);re=aggregate(d,['split','k','Re'],['excess_ed_u'])
    decisions=[]
    for k in ['2','5','10']:
        enough=all(next(r for r in ci if r['split']==sp and r['k']==k)['ci98_333_low']>0 for sp in ['development','test'])
        enough &= all(r['excess_ed_u']>0 for r in rep if r['split'] in ['development','test'] and r['k']==k)
        enough &= sum(r['excess_ed_u']>0 for r in re if r['split']=='test' and r['k']==k)>=2
        decisions.append(dict(k=int(k),stable_distribution_excess=bool(enough)))
    dump(OUT/'statistical_decisions.json',dict(criteria='predeclared simultaneous intervals and both repeats/development/test/Re signs',by_k=decisions))
    print('ANALYZED840 distribution comparisons/10080 quality rows; fixed-k, trajectory paired statistics',flush=True)

if __name__=='__main__':main()
