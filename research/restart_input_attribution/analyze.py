"""Trajectory-first summaries and paired fixed-k bootstrap; no sample selection."""
from core import *
from collections import defaultdict


def avg(rows,keys,metrics):
    grouped=defaultdict(list)
    for r in rows:grouped[tuple(r[k] for k in keys)].append(r)
    output=[]
    for key,rr in grouped.items():
        row=dict(zip(keys,key))
        for m in metrics:
            values=[float(r[m]) for r in rr if r.get(m) not in ('',None)]
            row[m]=float(np.mean(values)) if values else None
        output.append(row)
    return output


def ci(rows,keys,metrics):
    tr=avg(rows,keys+['trajectory'],metrics);grouped=defaultdict(list)
    for r in tr:grouped[tuple(r[k] for k in keys)].append(r)
    out=[]
    for key,rr in grouped.items():
        rr=sorted(rr,key=lambda r:r['trajectory'])
        for m in metrics:
            a=np.array([r[m] for r in rr if r[m] is not None]);n=len(a)
            if n:
                rng=np.random.default_rng(123);samples=a[rng.integers(n,size=(10000,n))].mean(1)
                lo,hi,blo,bhi=np.quantile(samples,[.025,.975,.05/6,1-.05/6])
                out.append(dict(zip(keys,key),metric=m,mean=float(a.mean()),ci95_low=lo,ci95_high=hi,ci98_333_low=blo,ci98_333_high=bhi,trajectories=n,positive=int((a>0).sum()),negative=int((a<0).sum())))
    return out


def main():
    assert (OUT/'COMPLETE.json').exists(),'Do not silently analyze incomplete fixed lists'
    assert not (OUT/'ANALYSIS.json').exists(),'Preserve completed statistics'
    q=old.read_csv(OUT/'quality.csv');assert len(q)==3456
    families=[('quality',q,['method','channel'],QUAL),
      ('response',old.read_csv(OUT/'response_members.csv'),['channel'],['D_rms','Dm_rms','Dd_rms','D_square','Dm_square','Dd_square','cross','mean_minus_member_square','identity_error']),
      ('response_ensemble',old.read_csv(OUT/'response_ensemble.csv'),['channel'],['D_rms','Dm_rms','Dd_rms','D_square','Dm_square','Dd_square','cross']),
      ('signed_mse',old.read_csv(OUT/'signed_mse.csv'),['scale','channel'],['mse_difference','error_response_cross','response_square','identity_error'])]
    jfile=OUT/'jvp.csv'
    if jfile.exists():
        j=old.read_csv(jfile)
        families.append(('jvp',j,['dose','channel'],['input_rms','prefix_rms','linear_rms','response_rms','error_rms','relative_error','cosine','linear_gain','response_gain','near_zero']))
    for name,rows,extra,metrics in families:
        for level,suffix in [([], 'summary'),(['Re'],'by_Re'),(['trajectory','Re'],'by_trajectory')]:
            csvwrite(OUT/f'{name}_{suffix}.csv',avg(rows,['split','k']+level+extra,metrics))
        csvwrite(OUT/f'{name}_intervals.csv',ci(rows,['split','k']+extra,metrics))
        csvwrite(OUT/f'{name}_intervals_by_Re.csv',ci(rows,['split','k','Re']+extra,metrics))
    index={tuple(r[k] for k in META+['channel','method']):r for r in q};delta=[]
    for r in q:
        if r['method']!='S':continue
        for a,b in [('S','R'),('M','R'),('S','M'),('R','B')]:
            x=index[tuple(r[k] for k in META+['channel'])+(a,)];y=index[tuple(r[k] for k in META+['channel'])+(b,)]
            delta.append(dict({k:r[k] for k in META+['channel']},contrast=a+' minus '+b,**{m:float(x[m])-float(y[m]) for m in QUAL}))
    csvwrite(OUT/'quality_differences.csv',delta)
    csvwrite(OUT/'quality_differences_by_trajectory.csv',avg(delta,['split','k','trajectory','Re','channel','contrast'],QUAL))
    csvwrite(OUT/'quality_paired.csv',ci(delta,['split','k','channel','contrast'],QUAL))
    csvwrite(OUT/'quality_paired_by_Re.csv',ci(delta,['split','k','channel','contrast','Re'],QUAL))
    leads=old.read_csv(OUT/'lead_channel.csv')
    for extra,suffix in [([], 'summary'),(['Re'],'by_Re'),(['trajectory','Re'],'by_trajectory')]:
        csvwrite(OUT/f'lead_channel_{suffix}.csv',avg(leads,['split','k']+extra+['method','channel','tail_lead'],QUAL))
    decisions=[]
    if jfile.exists():
        tr=avg([r for r in j if r['dose']=='full' and r['channel']=='overall'],['split','k','trajectory'],['relative_error','cosine'])
        for row in tr:row['credible']=row['relative_error'] is not None and row['cosine'] is not None and row['relative_error']<=.2 and row['cosine']>=.95
        csvwrite(OUT/'jvp_full_amplitude_criterion.csv',tr)
        decisions=tr
    dump(OUT/'ANALYSIS.json',dict(utc=time.time(),unit='original trajectory; windows/repeats/members averaged within trajectory',bootstrap=10000,seed=123,intervals=[.95,1-.05/3],jvp_near_zero_rows=sum(int(r['near_zero']) for r in j) if jfile.exists() else None,full_amplitude_credible_trajectory_k=sum(r['credible'] for r in decisions),full_amplitude_total_trajectory_k=len(decisions)))
    print('ANALYZED paired original trajectories, all fixed k/channels/Re/leads',flush=True)

if __name__=='__main__':main()
