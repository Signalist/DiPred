"""Frozen-head feedback switch, twenty contiguous blocks after phase 2."""
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

repo = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(repo/'research/residual_dynamics_phase2'), str(repo)]
from common import WORK, DATA, EXTRA, SEEDS, protocol, digest, dump, csvwrite, bootstrap
from causal import BlockPredictor, BiasHead
from metrics import empirical_crps, energy_score, consecutive_event
from reliability.backbone import configure
from research.residual_dynamics.run import raw, load_backbone

out = repo/'results/residual_dynamics_stream'
starts = list(range(440, 821, 20))
metrics = ('crps','energy','mse','brier')


def score(paths, truth, std, threshold):
    p, y = paths.double(), truth.double()
    z, target = p/std, y/std
    prob = float(consecutive_event(p,threshold).double().mean())
    event = int(consecutive_event(y[None],threshold)[0])
    return dict(crps=float(empirical_crps(z,target).mean()),energy=float(energy_score(z,target)),
                mse=float((p.mean(0)-y).square().mean()),prob=prob,event=event,brier=(prob-event)**2)


@torch.no_grad()
def main():
    configure()
    records = protocol()['records']['test']
    manifest = json.loads((WORK/'runs/clean_direct20_20261107/manifest.json').read_text())
    all_real = {p.name for p in DATA.glob('*.h5')}
    seen = {r['file'] for group in manifest['files'].values() for r in group}
    extra_real = sorted(all_real-seen)
    sim = DATA.parent/'train_sim'
    sim_groups = {p.stem.split('_')[0] for p in sim.glob('*.h5')}
    threshold = json.loads((repo/'results/residual_dynamics_phase2/causal_training_protocol.json').read_text())['event_threshold']
    weights = {s:out/'checkpoints'/f'causal_with_memory_{s}.pt' for s in SEEDS}
    expected = json.loads((out/'checkpoints/SHA256.json').read_text())
    assert all(digest(path) == expected[path.name] for path in weights.values())
    dump(out/'protocol.json',dict(starts=starts,files=[r['file'] for r in records],heads={s:digest(p) for s,p in weights.items()},
        scope='Historical test trajectories; follow-on blocks distinct from both phase2 windows. New stream starts cold.',
        endpoints='warm macro standardized empirical CRPS and full-field20 RMS Energy Score; both lower',
        contrasts=['on-off: feedback field and availability vs masked','on-zero_present: value vs zero at fixed availability','on-native: total benefit'],
        seeds=list(SEEDS),pairs='average over windows within trajectory; average fixed head seeds before trajectory bootstrap',
        threshold=threshold,real_files=len(all_real),outside_manifest=extra_real,duplicates=manifest['duplicates'],
        extra_sim_groups=sorted(sim_groups-{p.split('_')[0] for p in all_real}),sim_used=False))
    model,norm,_=load_backbone(SimpleNamespace(workspace=WORK))
    predictors={}
    for seed,path in weights.items():
        ck=torch.load(path,map_location='cuda',weights_only=False)
        head=BiasHead().cuda().eval().requires_grad_(False)
        head.load_state_dict(ck['model'])
        for cond in ('on','off','zero_present'):
            predictors[seed,cond]=BlockPredictor('with_memory' if cond!='off' else 'current_fields',norm,head,ck['scale'])
    predictors[0,'native']=BlockPredictor('native',norm)
    history=raw(DATA,records,starts[0])
    rows=[]
    for index,start in enumerate(starts):
        h=(history-norm['mean'])/norm['std']
        paths=torch.stack([model.rollout(h,s,sign=1)*norm['std']+norm['mean'] for s in range(500000,500032)],1)
        issued={}
        for key,predictor in predictors.items():
            if key[1]=='zero_present' and index:
                predictor.previous=torch.zeros_like(predictor.previous)
            issued[key]=predictor.predict(history,paths)
        if index==0:
            for s in SEEDS:
                assert torch.equal(issued[s,'on'],issued[s,'off']) and torch.equal(issued[s,'on'],issued[s,'zero_present'])
        # Only now read this block's observations. They become next block's history.
        truth=raw(DATA,records,start+20)
        for (seed,cond),value in issued.items():
            for i,r in enumerate(records):
                rows.append(dict(file=r['file'],Re=r['group'],start=start,state='warm' if index else 'cold',
                    seed=seed,condition=cond,**score(value[i],truth[i],norm['std'][0],threshold)))
            predictors[seed,cond].reveal(truth)
        history=truth
        print('complete',index+1,'/20',start,flush=True)
    csvwrite(out/'per_window.csv',rows)
    report=[]
    for subset in (SEEDS,*( (s,) for s in SEEDS)):
        for cond in ('on','off','zero_present','native'):
            for state in ('warm','cold'):
                selected=[r for r in rows if r['condition']==cond and r['state']==state and (r['seed'] in subset or cond=='native')]
                report.append(dict(seeds=list(subset),condition=cond,state=state,**{k:np.mean([r[k] for r in selected]) for k in metrics}))
    dump(out/'summary.json',report)
    intervals=[]
    for subset in (SEEDS,*((s,) for s in SEEDS)):
        for reference in ('off','zero_present','native'):
            for k in metrics:
                diffs=[]
                for r in records:
                    def mean(cond):
                        return np.mean([v[k] for v in rows if v['file']==r['file'] and v['state']=='warm' and v['condition']==cond
                                        and (cond=='native' or v['seed'] in subset)])
                    diffs.append(mean('on')-mean(reference))
                intervals.append(dict(seeds=list(subset),contrast='on-'+reference,metric=k,**bootstrap(diffs)))
    dump(out/'paired.json',intervals)


if __name__=='__main__':
    main()
