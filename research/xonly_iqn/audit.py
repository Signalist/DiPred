"""Audit data split, X-only inputs, causal online updates and aligned CDFs."""
import json,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from clean_reliability_common import manifest,digest
OUT=ROOT/'runs/xonly_iqn_20261112'
REF=ROOT/'github/DiPred/results/reliability/xp02_cqr'


def main():
    s=json.loads((OUT/'summary.json').read_text());t=json.loads((OUT/'training.json').read_text())
    e=json.loads((OUT/'events.json').read_text());r=np.load(OUT/'metrics.npz')
    assert s['passed'] and t['seed']==42 and t['steps']==2000 and t['examples']==819
    assert digest(OUT/'model.pt')==t['model_sha256']
    assert s['calibration_trajectories']==10 and s['observations_per_channel']==20152320
    assert r['cdf'].shape==(2,2,701) and r['crps'].shape==(41,2,12,2)
    assert len(e)==84 and s['starts']==list(range(20,821,20))
    for i,start in enumerate(range(0,821,20)):
        a,b=e[2*i:2*i+2]
        assert a['event']=='issued_both' and b['event']=='revealed_complete_20' and a['start']==b['start']==start and a['time']<=b['time']
    groups=manifest()['files']
    for a,b in (('train','calibration'),('train','test'),('calibration','test')):
        assert {v['group'] for v in groups[a]}.isdisjoint({v['group'] for v in groups[b]})
    for c,channel in enumerate(('u','v')):
        new=np.load(OUT/f'cdf_{channel}.npz');old=np.load(REF/f'cdf_{channel}.npz')
        assert np.array_equal(new['x'],old['x']) and new['count']==old['count']==20152320
        for k in ('xp02_cqr','xpf16_1','xpf16_02','noise8','noise32','noise256','noise2048'):
            assert np.array_equal(new[k],old[k])
        assert np.array_equal(new['xonly_iqn_raw'],r['cdf'][0,c]) and np.array_equal(new['xonly_iqn_online'],r['cdf'][1,c])
        for k in ('xonly_iqn_raw','xonly_iqn_online'):
            assert np.isfinite(new[k]).all() and np.diff(new[k]).min()>=-1e-7
    assert np.isfinite(r['crps']).all() and (r['crps']>=-1e-7).all()
    audit=dict(passed=True,fit_windows=819,seed=42,model_sha256=t['model_sha256'],calibration_trajectories=10,
        test_trajectories=12,test_windows=41,observations_per_channel=20152320,causal_events=84,
        warning='Re groups disjoint, but development/test identities used in earlier exploration; no formal coverage guarantee')
    (OUT/'audit.json').write_text(json.dumps(audit,indent=2)+'\n');print('XONLY_IQN_AUDIT_PASSED',flush=True)

if __name__=='__main__':main()
