"""Fixed three-seed mean; trajectory bootstrap and Re group summary kept separate."""
from pathlib import Path
import numpy as np
import pandas as pd
O=Path(__file__).resolve().parent.parent/'runs/recoverability_20261118'
p=pd.read_csv(O/'policy_trajectories.csv');out=[];groups=[]
for split,d in p[(p.quota==.5)&p.policy.isin(['benefit','risk','random_expected','oracle'])].groupby('split'):
 t=d.groupby(['file','group','policy'],as_index=False).mse.mean();w=t.pivot(index=['file','group'],columns='policy',values='mse');assert w.notna().all().all();a=w.benefit.to_numpy();b=w.random_expected.to_numpy();rng=np.random.default_rng(42);i=rng.integers(0,len(a),(10000,len(a)));draw=(b[i]-a[i]).mean(1);lo,hi=np.quantile(draw,[.025,.975]);out.append(dict(split=split,n_trajectories=len(w),seeds_averaged=3,benefit_mean=float(a.mean()),random_mean=float(b.mean()),delta=float((b-a).mean()),relative_gain_pct=float((1-a.mean()/b.mean())*100),ci_low_absolute=float(lo),ci_high_absolute=float(hi),n_re=int(w.index.get_level_values('group').nunique()),significant_uncorrected=bool(lo>0 or hi<0)))
 for group,v in w.groupby(level='group'):groups.append(dict(split=split,group=group,trajectories=len(v),delta=float((v.random_expected-v.benefit).mean()),gain_percent=float((1-v.benefit.mean()/v.random_expected.mean())*100)))
pd.DataFrame(out).to_csv(O/'primary_seed_average.csv',index=False);pd.DataFrame(groups).to_csv(O/'primary_by_re.csv',index=False);print('PRIMARY_SEED_AVERAGE',out,flush=True)
