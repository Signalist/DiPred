"""Paths, immutable protocol, grouping, and tensors shared by phase-two studies."""
import csv
import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
WORK = REPO.parents[1]
OLD = REPO / 'results/residual_dynamics'
OUT = Path(os.environ.get('DIPRED_PHASE2_OUT', REPO / 'results/residual_dynamics_phase2'))
CACHE = WORK / 'runs/residual_dynamics_20261119'
EXTRA = Path(os.environ.get('DIPRED_PHASE2_CACHE', WORK / 'runs/residual_dynamics_phase2'))
DATA = WORK.parent / 'OLD/REALPDE/realpdebench-competition/data/train_real'
SEEDS = (42, 43, 44)
DEVICE = 'cuda'


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def read(path):
    return json.loads(path.read_text())


def csvwrite(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        w.writeheader()
        w.writerows(rows)


def protocol():
    return read(OLD / 'protocol.json')


def events(name, **kwargs):
    with (OUT/'events.jsonl').open('a') as f:
        f.write(json.dumps(dict(event=name, **kwargs))+'\n')


def residual(split, device=DEVICE):
    arrays = []
    for start in protocol()['starts']:
        v = torch.load(CACHE/f'{split}_{start}.pt', map_location='cpu', weights_only=False)
        assert v['identity']['files'] == [r['file'] for r in protocol()['records'][split]]
        # [trajectory,window,member,frame,channel,y,x]
        arrays.append(v['residual'].permute(0, 1, 3, 2, 4, 5))
    return torch.stack(arrays, 1).contiguous().to(device)


def components(e):
    bias = e.mean(2, keepdim=True)
    return bias, e-bias


def aux(split):
    return torch.load(EXTRA/f'{split}_fields.pt', map_location=DEVICE, weights_only=False)


def coordinates(n, h, w, leads, device):
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, h, device=device),
                            torch.linspace(-1, 1, w, device=device), indexing='ij')
    return torch.cat([torch.stack([xx, yy])[None].expand(n,-1,-1,-1),
                      ((leads.float()-2)/18*2-1)[:,None,None,None].expand(n,1,h,w)], 1)


def bootstrap(deltas):
    v = np.asarray(deltas, dtype=float)
    rng = np.random.default_rng(123)
    means = v[rng.integers(len(v),size=(10000,len(v)))].mean(1)
    return dict(mean_delta=float(v.mean()), ci95=np.quantile(means,[.025,.975]).tolist(),
                improved=int((v<0).sum()), trajectories=len(v))


def aggregate(rows, keys, values):
    groups = {}
    for row in rows:
        key = tuple(row[k] for k in keys)
        groups.setdefault(key, []).append(row)
    return [dict(zip(keys,key), **{v: float(np.mean([r[v] for r in items])) for v in values},
                 count=len(items)) for key,items in groups.items()]


def freeze_protocol():
    OUT.mkdir(parents=True, exist_ok=True)
    EXTRA.mkdir(parents=True, exist_ok=True)
    spec = dict(reference='06096dd463f50341231a2bf2ca5724e8f6c36af8',
        head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        old_results={str(p.relative_to(REPO)):digest(p) for p in sorted(OLD.iterdir()) if p.is_file()},
        backbone_sha256=digest(REPO/'reliability/checkpoints/current.pt'),
        seeds=list(SEEDS), steps=2000, minibatch=16, optimizer='Adam lr=.001',
        selection='development trajectory-macro original-scale MSE, every250 steps',
        primary_mechanism='paired trajectory MSE: B-persistence and C-symmetric',
        primary_causal='standardized empirical ensemble CRPS and full-field20 RMS Energy Score',
        scoring='trajectory equally weighted; channel/lead/Re also reported; 10000 paired trajectory bootstrap',
        causal_event='spatial v RMS exceeds train-truth frame90th percentile for >=3 consecutive frames',
        shuffle_seeds=list(range(7100,7110)), no_test_tuning=True, no_remote_push=True,
        directional_claim='grid offsets only; no reliable physical propagation direction label',
        limitations='historically reused test,12 trajectories/3Re; seed variability separated from trajectory bootstrap')
    if (OUT/'preregistered.json').exists():
        prior=read(OUT/'preregistered.json')
        assert all(prior[k]==spec[k] for k in ('reference','old_results','backbone_sha256','seeds','steps'))
    else:
        dump(OUT/'preregistered.json',spec)
    return spec
