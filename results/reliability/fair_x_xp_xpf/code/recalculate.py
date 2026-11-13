"""Recompute summaries and paired intervals using only the published CSVs."""
import csv
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np

root = Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).resolve().parent.parent
with (root/'per_window.csv').open(newline='') as f:
    rows=list(csv.DictReader(f))
values=defaultdict(list)
for r in rows:
    values[(r['method'],r['seed'],r['calibration'],r['channel'])].append(float(r['crps']))
with (root/'summary.csv').open(newline='') as f:
    summary=list(csv.DictReader(f))
max_error=0.
for r in summary:
    if r['channel']=='both':
        a=values[(r['method'],r['seed'],r['calibration'],'u')]+values[(r['method'],r['seed'],r['calibration'],'v')]
    else:
        a=values[(r['method'],r['seed'],r['calibration'],r['channel'])]
    max_error=max(max_error,abs(np.mean(a)-float(r['crps'])))
print('Maximum CRPS summary discrepancy:',max_error)
assert max_error < 1e-8
for metric in ('coverage90','width90','interval_score90','adjacent_crossing','any_crossing'):
    for item in summary:
        relevant=[float(r[metric]) for r in rows if r['method']==item['method'] and r['seed']==item['seed'] and r['calibration']==item['calibration'] and (item['channel']=='both' or r['channel']==item['channel'])]
        assert abs(np.mean(relevant)-float(item[metric]))<1e-8
print('All summary interval and crossing metrics match per-window data.')
s=np.load(root/'prediction_snapshot.npz');q=s['q'];y=s['y'];tau=s['levels']
p=tau.reshape((1,len(tau))+(1,)*(q.ndim-2));e=y[:,None]-q
crps=.004*(.5*(np.maximum(q[:,0]-y,0)+np.maximum(y-q[:,-1],0))+np.maximum(p*e,(p-1)*e).sum(1))
crps*=s['normalization_std'].reshape(1,2,1,1,1)
max_error=np.abs(crps-s['crps']).max()
print('Maximum snapshot CRPS discrepancy:',max_error)
assert np.allclose(crps,s['crps'],atol=2e-6,rtol=3e-5)
print('Checks passed.')

# Recompute paired comparisons, resampling trajectory IDs (not windows).
by_trajectory=defaultdict(list)
for r in rows:
    for channel in (r['channel'],'both'):
        for metric in ('crps','interval_score90'):
            key=(r['calibration'],r['seed'],channel,r['method'],r['trajectory_id'],metric)
            by_trajectory[key].append(float(r[metric]))
with (root/'paired_comparison.csv').open(newline='') as f:
    paired=list(csv.DictReader(f))
rng=np.random.default_rng(20261113)
max_pair_error=0.
for r in paired:
    seeds=('42','123','2026') if r['seed']=='pooled' else (r['seed'],)
    ids=sorted({key[4] for key in by_trajectory if key[0]==r['calibration']})
    def scores(method):
        return np.array([np.mean([np.mean(by_trajectory[(r['calibration'],seed,r['channel'],method,trajectory,r['metric'])]) for seed in seeds]) for trajectory in ids])
    d=scores(r['baseline'])-scores(r['comparison'])
    boot=d[rng.integers(0,len(ids),size=(10000,len(ids)))].mean(1)
    actual=(float(d.mean()),float(np.quantile(boot,.025)),float(np.quantile(boot,.975)),float((d>0).mean()))
    stated=tuple(float(r[k]) for k in ('mean_delta','ci95_low','ci95_high','trajectory_win_rate'))
    max_pair_error=max(max_pair_error,max(abs(a-b) for a,b in zip(actual,stated)))
print('Maximum paired bootstrap discrepancy:',max_pair_error)
assert max_pair_error < 1e-7  # Float32 saved window means versus float64 reconstruction.
