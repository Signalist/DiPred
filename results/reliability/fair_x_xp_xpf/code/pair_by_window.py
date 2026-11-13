"""Window-resolved paired comparisons, bootstrapped over trajectory IDs."""
import csv
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'runs/fair_x_xp_xpf_20261113'
DEST=ROOT/'github/DiPred/results/reliability/fair_x_xp_xpf'
arms=('X','XP','XPF');seeds=(42,123,2026);modes=('raw','fixed_cqr','online')
data={}
for seed in seeds:
    for arm in arms:
        p=torch.load(OUT/'evaluated'/f'{arm}_{seed}.pt',map_location='cpu',weights_only=False)
        for mode in range(3):
            rows=[r for r in p['windows'] if r['mode']==mode]
            data[(arm,seed,mode)]=np.stack([r['values'] for r in rows]).astype(np.float64)
rng=np.random.default_rng(20261114);indices=rng.integers(0,12,size=(10000,12))
columns=('calibration','seed','channel','baseline','comparison','metric','window_start','mean_delta','ci95_low','ci95_high','trajectory_win_rate','trajectories','bootstrap_repetitions')
with (DEST/'paired_by_window.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=columns);w.writeheader()
    for mode,label in enumerate(modes):
        for seed in (*seeds,'pooled'):
            for channel,index in (('u',0),('v',1),('both',slice(None))):
                for left,right in (('X','XP'),('XP','XPF'),('X','XPF')):
                    for metric,k in (('crps',0),('interval_score90',3)):
                        def values(arm):
                            ss=seeds if seed=='pooled' else (seed,)
                            v=np.stack([data[(arm,s,mode)][:,:,index,k] for s in ss]).mean(0)
                            return v.mean(-1) if channel=='both' else v
                        delta=values(left)-values(right)
                        assert delta.shape==(41,12)
                        boot=delta[:,indices].mean(-1)
                        low=np.quantile(boot,.025,axis=1);high=np.quantile(boot,.975,axis=1)
                        for i in range(41):
                            w.writerow(dict(calibration=label,seed=seed,channel=channel,baseline=left,comparison=right,metric=metric,window_start=(i+1)*20,mean_delta=float(delta[i].mean()),ci95_low=float(low[i]),ci95_high=float(high[i]),trajectory_win_rate=float((delta[i]>0).mean()),trajectories=12,bootstrap_repetitions=10000))
print('Window paired bootstrap completed.')
