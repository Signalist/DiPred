"""Audit independent XP weights, CQR setup, and matched full-test curves."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import torch
from evaluate_xp import ROOT, OUT, REFERENCE, conformal_rank
from clean_reliability_common import manifest


def tensor_hash(state):
    h=hashlib.sha256()
    for key,value in sorted(state.items()):
        h.update(key.encode());h.update(value.contiguous().numpy().tobytes())
    return h.hexdigest()


def main():
    info=json.loads((OUT/'training_complete.json').read_text())
    assert info['passed'] and info['heads']==249 and info['seed']==42 and info['steps']==2000
    hashes=[]
    for k in range(1,250):
        model=torch.load(OUT/'models'/f'XP-P{k:03d}'/'model.pt',map_location='cpu',weights_only=False)
        assert model['k']==k and model['alpha']==k/500 and model['steps']==2000 and model['train_pairs']==819
        assert all(torch.isfinite(t).all() for t in model['model'].values())
        hashes.append(tensor_hash(model['model']))
    assert len(set(hashes))==249
    result=OUT/'results';summary=json.loads((result/'summary.json').read_text())
    assert summary['passed'] and summary['starts']==list(range(20,821,20))
    assert summary['observations_per_channel']==20152320 and len(summary['windows'])==41
    calibration=torch.load(OUT/'cqr_offsets.pt',map_location='cpu',weights_only=False)
    assert calibration['offsets'].shape==(249,2) and torch.isfinite(calibration['offsets']).all()
    assert calibration['calibration_trajectories']==10 and conformal_rank(10,.004) is None
    splits=manifest()['files']
    for first,second in (('train','calibration'),('train','test'),('calibration','test')):
        assert {r['file'] for r in splits[first]}.isdisjoint({r['file'] for r in splits[second]})
        assert {r['group'] for r in splits[first]}.isdisjoint({r['group'] for r in splits[second]})
    methods=('xpf16_1','xpf16_02','noise8','noise32','noise256','noise2048')
    for c in ('u','v'):
        a=np.load(result/f'cdf_{c}.npz');b=np.load(REFERENCE/f'cdf_{c}.npz')
        assert a['count']==b['count']==20152320 and np.array_equal(a['x'],b['x'])
        for method in methods:assert np.array_equal(a[method],b[method])
        y=a['xp02_cqr']
        assert len(y)==701 and np.isfinite(y).all() and y.min()>=-1e-7 and y.max()<=1+1e-7
        assert np.diff(y).min()>=-1e-7
        table=np.loadtxt(result/f'cdf_{c}.csv',delimiter=',',skiprows=1)
        assert table.shape==(701,8) and np.allclose(table[:,0],a['x']) and np.allclose(table[:,-1],y)
    subprocess.run([sys.executable,str(ROOT/'xp02_cqr/test_xp.py')],check=True)
    (result/'audit.json').write_text(json.dumps(dict(passed=True,independent_heads=249,
      weights_unique_sha256=hashes,calibration_n=calibration['n'],calibration_trajectory_count=10,
      cqr_extreme_trajectory_only_correction='infinite: insufficient independent trajectories',
      cdf_aligned_to_prior=True,cdf_observations_per_channel=20152320),indent=2)+'\n')
    print('XP_CQR_AUDIT_PASSED',flush=True)

if __name__=='__main__':main()
