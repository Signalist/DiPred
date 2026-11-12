"""Audit causal sequencing, data alignment and two reused study baselines."""
import json
import sys
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from clean_reliability_common import manifest
OUT=ROOT/'runs/controlled_xp_xpf_20261112'


def main():
    summary=json.loads((OUT/'summary.json').read_text())
    events=json.loads((OUT/'events.json').read_text())
    r=np.load(OUT/'results.npz')
    arms=('xp_fixed_cqr','xp_online','xpf_fixed_cqr','xpf_online','xp_raw','xpf_raw')
    assert tuple(r['arms'])==arms and r['cdf'].shape==(6,2,701)
    assert r['intervals'].shape==(41,6,249,12,2,5) and r['crps'].shape==(41,6,12,2)
    assert summary['observations_per_channel']==20152320 and summary['calibration_trajectories']==10
    assert r['starts'].tolist()==list(range(20,821,20)) and len(events)==84
    for i,start in enumerate(range(0,821,20)):
        issue,reveal=events[2*i:2*i+2]
        assert issue['event']=='issued_all_six' and reveal['event']=='revealed_complete_20'
        assert issue['start']==reveal['start']==start and issue['time']<=reveal['time']
    for first,second in (('train','calibration'),('train','test'),('calibration','test')):
        assert {v['group'] for v in manifest()['files'][first]}.isdisjoint({v['group'] for v in manifest()['files'][second]})
    assert np.isfinite(r['intervals']).all() and np.isfinite(r['crps']).all() and (r['crps']>=-1e-7).all()
    assert np.isfinite(r['cdf']).all() and np.diff(r['cdf'],axis=-1).min()>=-1e-7
    # Two arms were evaluated earlier; reusing their aligned CDFs checks the replay.
    diffs={}
    for c,channel in enumerate(('u','v')):
        old_xp=np.load(ROOT/'runs/xp02_cqr_20261112/results'/f'cdf_{channel}.npz')
        old_xpf=np.load(ROOT/'runs/xpf16_02grid_2048_20261112/full_test_cdf'/f'cdf_{channel}.npz')
        assert np.array_equal(r['x'],old_xp['x']) and np.array_equal(r['x'],old_xpf['x'])
        diffs[channel]=dict(xp_fixed=float(np.max(np.abs(r['cdf'][0,c]-old_xp['xp02_cqr']))),
                            xpf_online=float(np.max(np.abs(r['cdf'][3,c]-old_xpf['xpf16_02']))))
        assert max(diffs[channel].values())<2e-5,diffs
    # The two frozen calibration inputs must exactly match earlier releases.
    cal=torch.load(OUT/'calibration.pt',map_location='cpu',weights_only=False)
    prior_xp=torch.load(ROOT/'runs/xp02_cqr_20261112/cqr_offsets.pt',map_location='cpu',weights_only=False)
    prior_xpf=torch.load(ROOT/'runs/xpf16_02grid_2048_20261112/calibration.pt',map_location='cpu',weights_only=False)
    assert torch.equal(cal['xp_fixed_cqr'],prior_xp['offsets'])
    assert torch.equal(cal['xpf_online'],prior_xpf['xpf'])
    (OUT/'audit.json').write_text(json.dumps(dict(passed=True,causal_events=len(events),
       original_cdf_max_abs_difference=diffs,observations_per_channel=20152320),indent=2)+'\n')
    print('CONTROL_AUDIT_PASSED',flush=True)

if __name__=='__main__':main()
