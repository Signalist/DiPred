"""Independently reconstruct paired warm contrasts and stratified summaries from CSV."""
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np

out=Path(__file__).resolve().parents[2]/'results/residual_dynamics_stream'
rows=list(csv.DictReader((out/'per_window.csv').open()))
spec=json.loads((out/'protocol.json').read_text())
files=spec['files']
assert len(rows)==len(files)*20*10==2400
idx={(r['file'],int(r['start']),r['condition'],int(r['seed'])):r for r in rows}
assert len(idx)==len(rows)
metrics=('mse','crps','energy','brier')
trajectory=[]
for file in files:
    for start in spec['starts']:
        native=idx[file,start,'native',0]
        for seed in spec['seeds']:
            on=idx[file,start,'on',seed]
            for cond in ('on','off','zero_present'):
                r=idx[file,start,cond,seed]
                assert r['event']==native['event']
                assert r['state']==('cold' if start==440 else 'warm')
                assert all(np.isfinite(float(r[k])) for k in metrics)
                assert abs((float(r['prob'])-float(r['event']))**2-float(r['brier']))<1e-12
                if start==440: assert all(on[k]==r[k] for k in metrics)
    for ref in ('native','off','zero_present'):
        diffs={k:[] for k in metrics}
        for start in spec['starts'][1:]:
            for seed in spec['seeds']:
                on=idx[file,start,'on',seed]
                other=idx[file,start,ref,0 if ref=='native' else seed]
                for k in metrics:diffs[k].append(float(on[k])-float(other[k]))
        trajectory.append(dict(file=file,Re=file.split('_')[0],contrast='on-'+ref,**{k:np.mean(v) for k,v in diffs.items()}))
for r in json.loads((out/'paired.json').read_text()):
    if r['seeds']!=spec['seeds']:continue
    v=np.array([t[r['metric']] for t in trajectory if t['contrast']==r['contrast']])
    assert abs(v.mean()-r['mean_delta'])<1e-12
    assert np.allclose(np.quantile(np.random.default_rng(123).choice(v,(10000,len(v))).mean(1),[.025,.975]),r['ci95'],rtol=1e-10,atol=1e-12)
with (out/'per_trajectory_contrast.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=trajectory[0]);w.writeheader();w.writerows(trajectory)
print('PASS: 2400 unique forecasts, same targets, cold switch equality, 12 paired means and bootstrap intervals')
native=[r for r in rows if r['condition']=='native']
positive=[r for r in native if int(r['event'])]
print('Event positives:',len(positive),'of',len(native),'across',len({r['file'] for r in positive}),'trajectories')
for cond in ('native','off','zero_present','on'):
    warm=[r for r in rows if r['condition']==cond and r['state']=='warm']
    print(cond, {k:float(np.mean([float(r[k]) for r in warm])) for k in metrics})
