"""Independent checks of split, masks, causal ordering, tables, and CRPS."""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent.parent
sys.path[:0]=[str(ROOT),str(ROOT/'fair_x_xp_xpf')]
from fair_x_xp_xpf.model import FairQuantileHead
from evaluate import OUT, ARMS, SEEDS
from clean_reliability_common import digest, manifest
DEST=ROOT/'github/DiPred/results/reliability/fair_x_xp_xpf'


def read(path):
    with path.open(newline='') as f:return list(csv.DictReader(f))


def main():
    checks={}
    checks['genuine_frozen_four_step_features']=json.loads((OUT/'feature_audit.json').read_text())['passed']
    files=manifest()['files']; names=tuple(files)
    checks['trajectory_and_Re_disjoint']=all(
        {r['file'] for r in files[a]}.isdisjoint(r['file'] for r in files[b]) and
        {r['group'] for r in files[a]}.isdisjoint(r['group'] for r in files[b])
        for i,a in enumerate(names) for b in names[i+1:])
    hashes=[];checks['matched_initialization_and_parameter_count']=True
    for seed in SEEDS:
        saved=[torch.load(OUT/f'seed_{seed}'/f'{arm}.pt',map_location='cpu',weights_only=False) for arm in ARMS]
        checks['matched_initialization_and_parameter_count'] &= len({s['init_sha256'] for s in saved})==1 and len({s['parameters'] for s in saved})==1 and all(s['steps']==2000 for s in saved)
        hashes += [digest(OUT/f'seed_{seed}'/f'{arm}.pt') for arm in ARMS]
    checks['nine_independent_weight_files']=len(set(hashes))==9
    model=FairQuantileHead().eval()
    x=torch.randn(1,2,20,32,64);p=torch.randn_like(x);f=torch.randn(1,48,32,64)
    levels=torch.tensor([.002,.05,.5,.95,.998])
    with torch.no_grad():
        xout=model.quantiles(x,p,f,torch.tensor([[0.,0.]]),levels)
        xnoise=model.quantiles(x,p*100+10,f*100-10,torch.tensor([[0.,0.]]),levels)
        xpout=model.quantiles(x,p,f,torch.tensor([[1.,0.]]),levels)
        xpnoise=model.quantiles(x,p,f*100-10,torch.tensor([[1.,0.]]),levels)
    checks['missing_slots_invariant']=bool(torch.equal(xout,xnoise) and torch.equal(xpout,xpnoise))
    checks['raw_architecture_noncrossing']=bool((xout[1:]>=xout[:-1]-1e-6).all())
    checks['causal_issue_reveal']=True
    checks['all_model_results_complete']=True
    for seed in SEEDS:
        for arm in ARMS:
            part=torch.load(OUT/'evaluated'/f'{arm}_{seed}.pt',map_location='cpu',weights_only=False)
            events=part['events'];checks['all_model_results_complete'] &= len(part['windows'])==123 and len(part['frames'])==123
            checks['causal_issue_reveal'] &= len(events)==84 and all(events[2*i]['event']=='issued' and events[2*i+1]['event']=='revealed' and events[2*i]['start']==events[2*i+1]['start']==i*20 and events[2*i]['time']<=events[2*i+1]['time'] for i in range(42))
    windows=read(DEST/'per_window.csv');frames=read(DEST/'per_frame.csv');summary=read(DEST/'summary.csv')
    checks['expected_row_counts']=(len(windows)==27*41*12*2 and len(frames)==27*41*12*2*20 and len(summary)==81)
    keycols=('method','seed','calibration','trajectory_id','window_start','channel')
    acc=defaultdict(list)
    for r in frames:acc[tuple(r[k] for k in keycols)].append(float(r['crps']))
    checks['per_frame_matches_per_window']=len(acc)==len(windows) and all(len(acc[tuple(r[k] for k in keycols)])==20 and abs(np.mean(acc[tuple(r[k] for k in keycols)])-float(r['crps']))<1e-6 for r in windows)
    grouped=defaultdict(list)
    for r in windows:
        for channel in (r['channel'],'both'):grouped[(r['method'],r['seed'],r['calibration'],channel)].append(r)
    checks['summary_matches_windows']=True
    for r in summary:
        rows=grouped[(r['method'],r['seed'],r['calibration'],r['channel'])]
        for col in ('crps','coverage90','width90','interval_score90','adjacent_crossing','any_crossing'):
            checks['summary_matches_windows'] &= abs(np.mean([float(v[col]) for v in rows])-float(r[col]))<1e-8
    s=np.load(DEST/'prediction_snapshot.npz');q=s['q'];y=s['y'];tau=s['levels']
    expanded=tau.reshape((1,len(tau))+(1,)*(q.ndim-2));err=y[:,None]-q
    pin=np.maximum(expanded*err,(expanded-1)*err)
    first=np.maximum(q[:,0]-y,0);last=np.maximum(y-q[:,-1],0)
    calc=.004*(.5*(first+last)+pin.sum(axis=1))
    scale=s['normalization_std'].reshape(2);calc*=scale.reshape(1,2,1,1,1)
    checks['snapshot_crps_independent_recomputation']=bool(np.allclose(calc,s['crps'],rtol=3e-5,atol=2e-6))
    checks['snapshot_quantiles_nondecreasing']=bool(np.diff(q,axis=1).min()>=-1e-6)
    for channel in ('u','v'):
        data=np.genfromtxt(DEST/f'cdf_{channel}.csv',delimiter=',',skip_header=1)
        checks[f'cdf_{channel}_valid']=bool(np.isfinite(data).all() and np.all(np.diff(data[:,0])>0) and np.diff(data[:,1:],axis=0).min()>=-1e-6 and data[:,1:].min()>=-1e-5 and data[:,1:].max()<=1+1e-5)
    result=dict(checks=checks,passed=all(checks.values()),qualification='Test trajectories have been reused in earlier development; findings are exploratory. Split CQR pools correlated observations and has no trajectory-level guarantee.')
    (DEST/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)
    if not result['passed']:raise RuntimeError('Audit failure')

if __name__=='__main__':main()
