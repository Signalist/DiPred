"""Shared frozen protocol for the first valid v2 retraining comparison."""
import csv, hashlib, json, os, time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from research.trust_causal_audit_v2.common import ROOT, DATA, OUT as AUDIT_OUT, MEMBERS, digest, tensor_hash, state_hash, load_backbone, records
from research.trust_causal_audit_v2.data import Key, CacheIndex, H5Reader, prepare_inputs, model_inputs, batch_noise
from research.trust_causal_audit_v2.model import TrustDiffusion, native_sample, crps_parts, energy_score

OUT = ROOT/'results/trust_routing_retrain_v2'
CACHE = ROOT/'.cache/trust_routing_retrain_v2'
OLD_CACHE = ROOT/'.cache/error_guided_spatial_routing'
SEED = 42
STEPS = 1000
SCHEDULE_SEED = 20261129
CANDIDATES = (0, 250, 500, 1000)


def dump(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def csvwrite(path, rows):
    if not rows:return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)


def load_protocol():
    ready=json.loads((AUDIT_OUT/'VERIFICATION.json').read_text())
    assert ready['passed'] and ready['READY_FOR_RETRAINING']
    assert ready['optimizer_updates']==0 and ready['formal_test_comparisons']==0
    scale=json.loads((AUDIT_OUT/'RISK_NORMALIZATION.json').read_text())
    manifest=json.loads((AUDIT_OUT/'WINDOW_MANIFEST.json').read_text())
    cache_manifest=json.loads((AUDIT_OUT/'CACHE_MANIFEST.json').read_text())
    assert len(manifest)==173 and sum(x['split']=='train' for x in manifest)==137 and sum(x['split']=='development' for x in manifest)==36
    assert scale['maps']==137 and scale['split']=='train'
    return ready,scale,manifest,cache_manifest


def state_files():
    return {
      'audit_verification':digest(AUDIT_OUT/'VERIFICATION.json'),
      'window_manifest':digest(AUDIT_OUT/'WINDOW_MANIFEST.json'),
      'risk_normalization':digest(AUDIT_OUT/'RISK_NORMALIZATION.json'),
      'cache_manifest':digest(AUDIT_OUT/'CACHE_MANIFEST.json'),
      'v2_data':digest(ROOT/'research/trust_causal_audit_v2/data.py'),
      'v2_model':digest(ROOT/'research/trust_causal_audit_v2/model.py'),
      'v2_common':digest(ROOT/'research/trust_causal_audit_v2/common.py'),
      'checkpoint':digest(ROOT/'reliability/checkpoints/current.pt')}


def load_indexes(split, starts, rs):
    indexes={}
    files=[x['file'] for x in rs[split]]
    for start in starts:
        p=OLD_CACHE/f'{split}_{start}_native.pt'
        assert p.exists(), f'missing frozen feedback cache: {p}'
        payload=torch.load(p,map_location='cpu',weights_only=False)
        indexes[(split,start)] = CacheIndex(payload,split,start,files,str(p.relative_to(ROOT)))
    return indexes


def load_entries(split, rows, norm, scale, indexes):
    """Use v2 publication preparation; target read follows input preparation."""
    rs=records();reader=H5Reader(DATA);entries=[]
    for row in rows:
        key=Key(row['split'],row['trajectory_id'],int(row['start']))
        previous=indexes[(split,key.start-20)].lookup(Key(split,key.trajectory_id,key.start-20))
        prepared=prepare_inputs(reader,key,previous,norm,scale)
        # This read is a training/development label only, after route inputs exist.
        target=reader(key.trajectory_id,key.start+20,key.start+40)
        assert tensor_hash(prepared['risk'])==row['risk_sha256']
        assert tensor_hash(prepared['raw_risk'])==row['raw_risk_sha256']
        entries.append(dict(row=row,key=key,prepared=prepared,target=target,
                            previous=previous,history_reader_requests=list(reader.reads[-2:])))
    return entries


def split_rows(manifest, split):
    return [r for r in manifest if r['split']==split]


def files_for(split):
    return [x['file'] for x in records()[split]]


def noise_for(entry, device='cuda'):
    noise, identities=batch_noise(entry['key'],files_for(entry['key'].split),device=device)
    return noise,identities


def to_inputs(entry, noise, norm):
    return model_inputs(entry['prepared'],noise)


def metrics(normalized_paths, target_raw, norm):
    std=norm['std'];mean=norm['mean']
    target=(target_raw-mean)/std
    c,first,pair=crps_parts(normalized_paths,target)
    raw=normalized_paths*std+mean
    raw_es=energy_score(raw,target_raw)
    std_es=energy_score(normalized_paths,target)
    err=raw.mean(0)-target_raw
    width=torch.quantile(raw,.95,0)-torch.quantile(raw,.05,0)
    time_crps=crps_parts(normalized_paths.mean(2),target.mean(1))[0]
    rows=[]
    for channel,name in ((None,'overall'),(0,'u'),(1,'v')):
        sl=slice(None) if channel is None else slice(channel,channel+1)
        ch_crps=crps_parts(normalized_paths[:,sl],target[sl])[0]
        rows.append(dict(channel=name,crps=float(ch_crps),energy=float(std_es if channel is None else energy_score(normalized_paths[:,sl],target[sl])),
            energy_raw=float(raw_es if channel is None else energy_score(raw[:,sl],target_raw[sl])),
            mse=float(err[sl].square().mean()),smse=float((err/sl_std(std,sl)).square().mean()),
            variance=float(raw[:,sl].var(0,unbiased=False).mean()),std_variance=float(normalized_paths[:,sl].var(0,unbiased=False).mean()),
            width90=float(width[sl].mean()),std_width90=float((width[sl]/sl_std(std,sl)).mean()),
            time_mean_crps=float(time_crps if channel is None else crps_parts(normalized_paths[:,sl].mean(2),target[sl].mean(1))[0])))
    return rows


def sl_std(std, sl):
    return std[sl]


def aggregate_trajectory(rows, metric_names):
    by=defaultdict(list)
    for row in rows:by[(row['arm'],row['step'],row['trajectory_id'])].append(row)
    tr=[]
    for (arm,step,trajectory),items in by.items():
        tr.append(dict(arm=arm,step=step,trajectory_id=trajectory,Re=items[0]['Re'],
          **{m:float(np.mean([float(x[m]) for x in items])) for m in metric_names}))
    by_arm=defaultdict(list)
    for x in tr:by_arm[(x['arm'],x['step'])].append(x)
    macro=[]
    for (arm,step),items in by_arm.items():
        macro.append(dict(arm=arm,step=step,trajectories=len(items),
          **{m:float(np.mean([x[m] for x in items])) for m in metric_names}))
    return tr,macro
