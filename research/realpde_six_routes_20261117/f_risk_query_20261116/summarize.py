from common import *
import pandas as pd

def main():
    setup();std=norm('cpu')['std'].flatten().numpy();folder=OUT/'query'
    q=pd.read_csv(folder/'per_query.csv');e=folder/'ensemble_per_query.csv'
    if e.exists():q=pd.concat([q,pd.read_csv(e)],ignore_index=True)
    for key in ('crps','width'):q['normalized_'+key]=q[key]/q.channel.map(dict(enumerate(std)))
    keys=['method','category','statistic','channel']
    sm=q.groupby(keys)[['crps','normalized_crps','coverage','width','normalized_width']].mean().reset_index();sm.to_csv(folder/'summary.csv',index=False)
    event=q[q.statistic=='max'].copy();event['brier']=event[['brier0','brier1','brier2']].mean(axis=1);event['prevalence']=event[['event0','event1','event2']].mean(axis=1)
    es=event.groupby(['method','category','channel'])[['brier','prevalence']].mean().reset_index();es.to_csv(folder/'event_summary.csv',index=False)
    # Per-trajectory paired differences; queries/pixels are not independent units.
    paired=[]
    for metric,table in [('normalized_crps',q),('brier',event)]:
        if metric=='brier':table=table.copy();table['statistic']='max'
        for (cat,stat),block in table.groupby(['category','statistic']):
            pivot=block.groupby(['method','file'])[metric].mean().unstack(0)
            for a in pivot:
                for b in ('X','P'):
                    if a==b:continue
                    diff=(pivot[b]-pivot[a]).values;rng=np.random.default_rng(42)
                    boots=diff[rng.integers(len(diff),size=(10000,len(diff)))].mean(1)
                    paired.append(dict(metric=metric,category=cat,statistic=stat,baseline=b,method=a,improvement=float(diff.mean()),ci_low=float(np.quantile(boots,.025)),ci_high=float(np.quantile(boots,.975)),trajectories=len(diff),positive_trajectories=int((diff>0).sum())))
    pd.DataFrame(paired).to_csv(folder/'paired_trajectory.csv',index=False)
    rel=[]
    for (a,cat,c),block in event.groupby(['method','category','channel']):
        for t in range(3):
            prob=block[f'prob{t}'];truth=block[f'event{t}']
            for j in range(10):
                sel=(prob>=j/10)&((prob<(j+1)/10) if j<9 else (prob<=1))
                if sel.any():rel.append(dict(method=a,category=cat,channel=c,threshold_index=t,bin=j,count=int(sel.sum()),predicted=float(prob[sel].mean()),frequency=float(truth[sel].mean())))
    pd.DataFrame(rel).to_csv(folder/'reliability_bins.csv',index=False)
    for path in (OUT/'risk').glob('*/per_group.csv'):
        d=pd.read_csv(path)
        for k in ('crps','width','interval'):d['normalized_'+k]=d[k]/d.channel.map(dict(enumerate(std)))
        s=d.groupby(['method','mode','region','channel'])[['crps','normalized_crps','mse','coverage','width','normalized_width','interval','normalized_interval']].mean().reset_index();s.to_csv(path.parent/'summary.csv',index=False)
        allrows=s[s.region=='all'].groupby(['method','mode'])[['normalized_crps','coverage','normalized_width','normalized_interval']].mean()
        print('RISK',path.parent.name,'\n',allrows.to_string(),flush=True)
    print('QUERY\n',sm.groupby(['method','category','statistic'])[['normalized_crps','coverage']].mean().to_string())
    print('EVENT\n',es.groupby(['method','category'])[['brier','prevalence']].mean().to_string())
if __name__=='__main__':main()
