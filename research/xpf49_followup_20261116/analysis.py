from followcommon import *
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
 setup();folder=OUT/'query';std=norm('cpu')['std'].flatten().numpy();q=pd.read_csv(folder/'per_query.csv');ecc=folder/'ecc_per_query.csv'
 if ecc.exists():q=pd.concat((q,pd.read_csv(ecc)),ignore_index=True)
 old=pd.read_csv(PRIOR/'query/ensemble_per_query.csv');old=old[old.method.isin(['noise8','noise32'])];q=pd.concat((q,old),ignore_index=True)
 q['normalized_crps']=q.crps/q.channel.map(dict(enumerate(std)));q['normalized_width']=q.width/q.channel.map(dict(enumerate(std)))
 agg=q.groupby(['method','category','statistic','channel'])[['normalized_crps','coverage','normalized_width']].mean().reset_index();agg.to_csv(folder/'summary.csv',index=False)
 ev=q[q.statistic=='max'].copy();ev['brier']=ev[['brier0','brier1','brier2']].mean(axis=1);ev['prevalence']=ev[['event0','event1','event2']].mean(axis=1)
 ev.groupby(['method','category','channel'])[['brier','prevalence']].mean().reset_index().to_csv(folder/'event_summary.csv',index=False)
 comparisons=[]
 for (cat,stat),d in q.groupby(['category','statistic']):
  pivot=d.groupby(['method','file']).normalized_crps.mean().unstack(0)
  for a,b in [('XPFQ','XPF'),('XPQ','XP'),('XPF','XP'),('XPFQ','XP'),('XPF49_ECC32_online','noise32')]:
   if a not in pivot or b not in pivot:continue
   diff=(pivot[b]-pivot[a]).values;rng=np.random.default_rng(42);bstrap=diff[rng.integers(len(diff),size=(10000,len(diff)))].mean(1)
   comparisons.append(dict(category=cat,statistic=stat,method=a,baseline=b,improvement=diff.mean(),ci_low=np.quantile(bstrap,.025),ci_high=np.quantile(bstrap,.975),wins=(diff>0).sum(),trajectories=len(diff)))
 pd.DataFrame(comparisons).to_csv(folder/'paired_trajectory.csv',index=False)
 rel=[]
 for (arm,cat,c),d in ev.groupby(['method','category','channel']):
  for t in range(3):
   prob=d[f'prob{t}'];truth=d[f'event{t}']
   for j in range(10):
    sel=(prob>=j/10)&(prob<((j+1)/10) if j<9 else prob<=1)
    if sel.any():rel.append(dict(method=arm,category=cat,channel=c,threshold=t,bin=j,count=int(sel.sum()),predicted=prob[sel].mean(),frequency=truth[sel].mean()))
 pd.DataFrame(rel).to_csv(folder/'reliability.csv',index=False)
 p=load(folder/'predictions.pt');truth=p['calibration_targets'][:,:,None,2:]>p['thresholds'][None,None];dtruth=p['targets'][:,:,None,2:]>p['thresholds'][None,None];decisions=[]
 for arm,probs in p['calibration'].items():
  seen=[v['id'] for v in p['queries'] if v['category']=='seen'];cp=probs[:,seen];ct=truth[:,seen];dp=p['predictions'][arm]['event']
  for c in range(2):
   for t in range(3):
    cut=torch.quantile(cp[:,:,t,c][~ct[:,:,t,c]],.9)
    for cat in ('seen','new_space','new_time','new_both'):
     ids=[v['id'] for v in p['queries'] if v['category']==cat];actual=dtruth[:,ids,t,c];alert=dp[:,ids,t,c]>cut;pos=int(actual.sum());neg=int((~actual).sum());fn=int((~alert&actual).sum());fp=int((alert&~actual).sum())
     decisions.append(dict(method=arm,category=cat,channel=c,threshold=t,calibration_cut=float(cut),positive=pos,negative=neg,fn=fn,fp=fp,fnr=fn/pos if pos else None,fpr=fp/neg if neg else None))
 pd.DataFrame(decisions).to_csv(folder/'decisions.csv',index=False)
 a=pd.read_csv(OUT/'audit/per_group.csv');g=OUT/'audit/grid250_per_group.csv'
 if g.exists():a=pd.concat((a,pd.read_csv(g)),ignore_index=True)
 audit=a.groupby(['arm','mode','center','region','channel'])[['crps','median_mse','coverage','width','interval_score']].mean().reset_index();audit.to_csv(OUT/'audit/summary.csv',index=False)
 av=audit[(audit.region=='all')&(audit['center']=='original')].groupby(['arm','mode'])[['crps','median_mse','coverage','width','interval_score']].mean()
 fig,axes=plt.subplots(1,2,figsize=(11,4));b=agg.groupby(['method','category','statistic']).normalized_crps.mean()
 for i,st in enumerate(('mean','max')):
  v=b.xs(('new_both',st),level=('category','statistic')).sort_values();axes[i].bar(range(len(v)),v.values,color=['#376b8e' if 'XPF' in x else '#b6865c' for x in v.index]);axes[i].set_xticks(range(len(v)));axes[i].set_xticklabels(v.index,rotation=50,ha='right');axes[i].set_title('Unseen region and time: '+st);axes[i].set_ylabel('Standardized CRPS')
 fig.tight_layout();(OUT/'figures').mkdir(exist_ok=True);fig.savefig(OUT/'figures/query.png',dpi=160);fig.savefig(OUT/'figures/query.pdf');plt.close(fig)
 print('AUDIT',av.to_string());print('QUERY_NEW',b.xs('new_both',level='category').to_string());print('PAIRED',pd.DataFrame(comparisons).query('category=="new_both"').to_string(index=False))
if __name__=='__main__':main()
