"""Paired original-trajectory statistics; seeds never add physical sample size."""
from core import *
from research.provenance_uncertainty_conditioning.metrics import EXTRA
from collections import defaultdict
METRICS=QUAL+EXTRA


def aggregate(rows,keys,metrics):
    sums={};counts={}
    for row in rows:
        key=tuple(row[k] for k in keys)
        if key not in sums:sums[key]=np.zeros(len(metrics));counts[key]=np.zeros(len(metrics),dtype=np.int64)
        for i,m in enumerate(metrics):
            v=row.get(m)
            if v not in ('',None):sums[key][i]+=float(v);counts[key][i]+=1
    return [dict(zip(keys,key),**{m:float(sums[key][i]/counts[key][i]) if counts[key][i] else None for i,m in enumerate(metrics)}) for key in sums]


def intervals(rows,keys,metrics):
    grouped=defaultdict(list)
    for row in rows:grouped[tuple(row[k] for k in keys)].append(row)
    output=[]
    for key,rr in grouped.items():
        rr=sorted(rr,key=lambda r:r['trajectory']);assert len({r['trajectory'] for r in rr})==len(rr)
        for metric in metrics:
            a=np.array([float(r[metric]) for r in rr if r.get(metric) not in ('',None)])
            if not len(a):continue
            rng=np.random.default_rng(123);means=a[rng.integers(len(a),size=(10000,len(a)))].mean(1)
            lo,hi,alo,ahi=np.quantile(means,[.025,.975,.05/6,1-.05/6])
            output.append(dict(zip(keys,key),metric=metric,mean=float(a.mean()),ci95_low=float(lo),ci95_high=float(hi),ci98_333_low=float(alo),ci98_333_high=float(ahi),trajectories=len(a),negative=int((a<0).sum()),positive=int((a>0).sum())))
    return output


def main():
    assert (OUT/'EVALUATION_COMPLETE.json').exists(),'Incomplete evaluation cannot be silently subset-reported'
    assert not (OUT/'ANALYSIS.json').exists()
    started=time.monotonic();rows=read_csv(OUT/'scores.csv.gz')
    keybase=['split','trajectory','Re','start','repeat','k','view','channel']
    ix={tuple(r[k] for k in keybase)+ (r['policy'],r['arm'],r['seed']):r for r in rows}
    assert len(ix)==len(rows)
    differences=[]
    for r in rows:
        own=tuple(r[k] for k in keybase)
        if r['policy']=='native':
            targets=[('S minus B','native','B','-1')] if r['arm']=='S' else []
        else:
            targets=[(r['arm']+' minus B','native','B','-1')]
            if r['view']!='direct':targets.append((r['arm']+' minus S','native','S','-1'))
            if r['arm']=='C_mask':targets.append(('C_mask minus C_aug',r['policy'],'C_aug',r['seed']))
            if r['arm']=='C_unc':targets.append(('C_unc minus C_mask',r['policy'],'C_mask',r['seed']))
        for contrast,policy,arm,seed in targets:
            other=ix[own+(policy,arm,seed)]
            delta={m:float(r[m])-float(other[m]) if r[m] not in ('',None) and other[m] not in ('',None) else None for m in METRICS}
            differences.append(dict({k:r[k] for k in keybase},policy=r['policy'],seed=r['seed'],contrast=contrast,**delta))
    out=Table(OUT/'paired_window_differences.csv.gz');out.write(differences);out.close()
    trkeys=['split','trajectory','Re','k','view','channel','policy','arm','seed']
    trajectory=aggregate(rows,trkeys,METRICS);csvwrite(OUT/'scores_by_trajectory_seed.csv',trajectory)
    means=aggregate(trajectory,[k for k in trkeys if k!='seed'],METRICS)
    for r in means:r['seed']='mean3' if r['policy']!='native' else '-1'
    csvwrite(OUT/'scores_by_trajectory.csv',means)
    for extra,suffix in [([], 'summary'),(['Re'],'by_Re'),(['seed'],'by_seed')]:
        source=trajectory if 'seed' in extra else means
        csvwrite(OUT/f'scores_{suffix}.csv',aggregate(source,['split','k','view','channel','policy','arm']+extra,METRICS))
    csvwrite(OUT/'scores_by_window.csv',aggregate(rows,['split','start','repeat','k','view','channel','policy','arm','seed'],METRICS))
    dtkeys=['split','trajectory','Re','k','view','channel','policy','seed','contrast']
    dt=aggregate(differences,dtkeys,METRICS)
    averaged=aggregate([r for r in dt if r['policy']!='native'],[k for k in dtkeys if k!='seed'],METRICS)
    for r in averaged:r['seed']='mean3'
    alltr=dt+averaged;csvwrite(OUT/'paired_by_trajectory.csv',alltr)
    base=['split','k','view','channel','policy','seed','contrast']
    paired=intervals(alltr,base,METRICS);csvwrite(OUT/'paired_intervals.csv',paired)
    byre=intervals(alltr,base+['Re'],METRICS);csvwrite(OUT/'paired_intervals_by_Re.csv',byre)
    print('ANALYZED trajectory/seed pairing',len(paired),'interval rows',flush=True)
    # Large lead files are reduced online, preserving original trajectory weights.
    leadtr=[]
    for split in ['test','extension']:
        for seed in ['native','42','43','44']:
            path=OUT/f'leads_{split}_{seed}.csv.gz'
            with gzip.open(path,'rt') as f:
                leadtr.extend(aggregate(csv.DictReader(f),['split','trajectory','Re','k','view','channel','policy','arm','seed','tail_lead'],QUAL))
    table=Table(OUT/'leads_by_trajectory.csv.gz');table.write(leadtr);table.close()
    csvwrite(OUT/'lead_summary.csv',aggregate(leadtr,['split','k','view','channel','policy','arm','seed','tail_lead'],QUAL))
    decisions=[]
    for contrast in ['C_aug minus S','C_mask minus C_aug','C_unc minus C_mask']:
        for k in ['2','5','10']:
            tests=[]
            for split in ['test','extension']:
                chosen=[r for r in paired if r['split']==split and r['k']==k and r['view']=='tail' and r['policy']=='scientific' and r['contrast']==contrast and r['metric']=='crps']
                mainrow=next(r for r in chosen if r['channel']=='overall' and r['seed']=='mean3')
                seedmeans=[r['mean'] for r in chosen if r['channel']=='overall' and r['seed'] in ['42','43','44']]
                channels=[r['mean'] for r in chosen if r['channel'] in ['u','v'] and r['seed']=='mean3']
                remeans=[r['mean'] for r in byre if r['split']==split and r['k']==k and r['view']=='tail' and r['policy']=='scientific' and r['contrast']==contrast and r['metric']=='crps' and r['channel']=='overall' and r['seed']=='mean3']
                tests.append(dict(split=split,interval_upper=mainrow['ci98_333_high'],all_seed_negative=all(x<0 for x in seedmeans),both_channels_negative=all(x<0 for x in channels),negative_Re=sum(x<0 for x in remeans)))
            stable=all(r['interval_upper']<0 and r['all_seed_negative'] and r['both_channels_negative'] and r['negative_Re']>=2 for r in tests)
            decisions.append(dict(contrast=contrast,k=int(k),stable_increment=stable,by_split=tests))
    dump(OUT/'DECISIONS.json',decisions)
    dump(OUT/'ANALYSIS.json',dict(score_rows=len(rows),window_difference_rows=len(differences),trajectory_seed_rows=len(trajectory),paired_interval_rows=len(paired),unit='original trajectory; average windows/repeats, then seeds within each trajectory for mean3',bootstrap=10000,bootstrap_seed=123,seconds=time.monotonic()-started,utc=time.time()))
    print('ANALYSIS COMPLETE; fixed k, all seeds/Re/channels/views/policies',flush=True)

if __name__=='__main__':main()
