"""Equal per-issue quota regret, ranking, calibration and seed uncertainty."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
O=Path(__file__).resolve().parent.parent/'runs/recoverability_20261118'
d=pd.read_csv(O/'windows.csv');p=pd.read_csv(O/'policies.csv');rows=[];seed_summary=[];byre=[]
for (split,seed,quota),v in p.groupby(['split','seed','quota']):
 t=v.groupby(['policy','file','group'],as_index=False).mse.mean();wide=t.pivot(index='file',columns='policy',values='mse');b=wide['benefit'];oracle=wide['oracle'];risk=wide['risk'];random=wide['random_expected'];
 assert wide.notna().all().all();rows.append(dict(split=split,seed=seed,quota=quota,benefit_mean=float(b.mean()),random_mean=float(random.mean()),risk_mean=float(risk.mean()),oracle_mean=float(oracle.mean()),regret=float((b-oracle).mean()),random_regret=float((random-oracle).mean()),gain_vs_risk_pct=float((1-b.mean()/risk.mean())*100)))
 q=d[(d.split==split)&(d.seed==seed)].copy();q=q.groupby(['file','group'],as_index=False)[['base_mse','warp_realized_gain','benefit_prediction','risk_prediction','rescrct_realized_gain']].mean()
 for (a,bname) in [('base_mse','warp_realized_gain'),('benefit_prediction','warp_realized_gain'),('risk_prediction','warp_realized_gain'),('benefit_prediction','rescrct_realized_gain')]:
  if quota==.5:byre.append(dict(split=split,seed=seed,x=a,y=bname,n=len(q),spearman=float(spearmanr(q[a],q[bname]).correlation)))
res=pd.DataFrame(rows);res.to_csv(O/'decision_regret.csv',index=False);pd.DataFrame(byre).to_csv(O/'trajectory_correlations.csv',index=False)
for (split,quota),v in res.groupby(['split','quota']):
 primary=pd.read_csv(O/'primary.csv');z=primary[(primary.split==split)&(primary.quota==quota)];seed_summary.append(dict(split=split,quota=quota,seeds=len(z),seed_mean_gain_percent=float(z.gain_percent.mean()),seed_sd_gain_percent=float(z.gain_percent.std(ddof=1)),mean_absolute_delta=float(z.delta.mean()),seed_sd_absolute_delta=float(z.delta.std(ddof=1)),min_worst_re_delta=float(z.worst_group_delta.min())))
pd.DataFrame(seed_summary).to_csv(O/'seed_summary.csv',index=False);print('DECISION_ANALYSIS',len(rows),flush=True)
