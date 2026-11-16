"""Validate a scenario release using only its published small data tables."""
import csv,hashlib,json,sys,gzip
from collections import defaultdict
from pathlib import Path
import numpy as np
root=Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).resolve().parent.parent
manifest=json.loads((root/'manifest.json').read_text())
for name,expected in manifest['files_sha256'].items():
    h=hashlib.sha256()
    with (root/name).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    assert h.hexdigest()==expected,name
for name in ('preflight.json','audit.json','field_audit.json'):
    assert json.loads((root/name).read_text())['passed'],name
key=('method','seed','calibration','trajectory_id','window_start','channel')
score=defaultdict(list);count=0
for folder in sorted((root/'runs').iterdir()):
    p=folder/'per_window.csv'; p=p if p.exists() else folder/'per_window.csv.gz'
    with (gzip.open(p,'rt',newline='') if p.suffix=='.gz' else p.open(newline='')) as f:
        win={tuple(r[k] for k in key):r for r in csv.DictReader(f)}
    p=folder/'per_frame.csv'; p=p if p.exists() else folder/'per_frame.csv.gz'
    with (gzip.open(p,'rt',newline='') if p.suffix=='.gz' else p.open(newline='')) as f:
        seen=set()
        for r in csv.DictReader(f):
            k=tuple(r[z] for z in key);assert k in win and k not in seen;seen.add(k)
            expected=float(win[k]['crps']);actual=sum(float(r[f'crps_frame_{i:02d}']) for i in range(1,21))/20
            assert abs(actual-expected)<1e-6
        assert len(seen)==len(win)
    for r in win.values():
        for channel in (r['channel'],'all'):
            for metric in ('crps','coverage90','width90','interval_score90','adjacent_crossing','any_crossing'):
                score[(r['method'],r['seed'],r['calibration'],channel,metric)].append(float(r[metric]))
    count+=len(win)
with (root/'summary.csv').open(newline='') as f:
    for r in csv.DictReader(f):
        for metric in ('crps','coverage90','width90','interval_score90','adjacent_crossing','any_crossing'):
            values=score[(r['method'],r['seed'],r['calibration'],r['channel'],metric)]
            assert abs(float(np.mean(values))-float(r[metric]))<1e-8,(r['method'],metric)
s=np.load(root/'prediction_snapshot.npz');q=s['q'];y=s['y'];levels=s['levels']
t=levels.reshape((1,len(levels))+(1,)*(q.ndim-2));e=y[:,None]-q
pin=np.maximum(t*e,(t-1)*e)
value=.004*(.5*(np.maximum(q[:,0]-y,0)+np.maximum(y-q[:,-1],0))+pin.sum(axis=1))
value*=s['normalization_std'].reshape(1,-1,1,1,1)
assert np.allclose(value,s['crps'],rtol=3e-5,atol=2e-6)
print('PASS: hashes, audits, per-frame alignment, summaries, and snapshot CRPS;',count,'window/channel rows')
