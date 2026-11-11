"""Independent audits of model completeness, coverage arithmetic and causality."""
import json
import hashlib
import subprocess
import sys

import numpy as np
import torch
from clean_reliability_common import digest
from train49 import OUT, configs


def main():
    cfgs=configs()
    state={}
    parameter_hashes={}
    for c in cfgs:
        path=OUT/'models'/c['id']/'model.pt'
        packed=torch.load(path,map_location='cpu',weights_only=False)
        assert packed['config']==c and packed['parameters']==34004 and packed['training_pairs']==819
        assert packed['loss'][-1]['step']==2000
        assert all(torch.isfinite(v).all() for v in packed['model'].values())
        state[c['id']]=digest(path)
        hasher=hashlib.sha256()
        for name,tensor in sorted(packed['model'].items()):
            hasher.update(name.encode());hasher.update(tensor.contiguous().numpy().tobytes())
        parameter_hashes[c['id']]=hasher.hexdigest()
    assert len(set(state.values()))==49 and len(set(parameter_hashes.values()))==49
    latency=json.loads((OUT/'latency.json').read_text())
    assert len(latency['heads'])==49 and all(r['head_seconds_per_12_trajectory_batch']>0 for r in latency['heads'])
    portable=json.loads((OUT/'portable49_audit.json').read_text())
    assert portable['passed'] and portable['max_metric_difference'] <= 1/2048+1e-7
    noise_batching=json.loads((OUT/'batched_noise_audit.json').read_text())
    assert noise_batching['passed']
    batching=json.loads((OUT/'batch_training_audit.json').read_text())
    assert batching['passed'] and batching['maximum_one_step_parameter_difference']<2e-6
    summary={}
    for split in ('development','test'):
        directory=OUT/'evaluation'/split
        meta=json.loads((directory/'run.json').read_text())
        assert meta['starts']==list(range(0,821,20))
        events=[json.loads(x) for x in (directory/'events.jsonl').read_text().splitlines()]
        assert len(events)==84
        for i,start in enumerate(meta['starts']):
            first,second=events[2*i:2*i+2]
            assert first['event']=='issued' and second['event']=='revealed_full20'
            assert first['start']==second['start']==start and first['time']<=second['time']
        max_formula=0.;max_coverage=0.
        for start in meta['starts']:
            a=np.load(directory/f'{start:04d}.npz')['metrics']
            assert a.shape==(2,4,49,len(meta['records']),2,20,9) and np.isfinite(a).all()
            assert (a[...,3]>=-1e-6).all()
            for k in range(49):
                q=(k+1)/100
                # Proper interval-score identity for distinct lower/upper quantiles.
                err=np.max(np.abs(a[:,:,k,...,4]-(a[:,:,k,...,5]+a[:,:,k,...,6])/q))
                max_formula=max(max_formula,float(err))
                cov=np.max(np.abs(a[:,:,k,...,2]-(1-a[:,:,k,...,7]-a[:,:,k,...,8])))
                max_coverage=max(max_coverage,float(cov))
        assert max_formula<1e-4,max_formula
        # Bound equality/float rounding may cause inclusion identities at zero-width.
        summary[split]=dict(windows=42,max_interval_score_identity_error=max_formula,
                            max_inclusive_coverage_identity_error=max_coverage,
                            projection=meta['projection'])
    for name in ('test_protocol.py','test_scores.py'):
        subprocess.run([sys.executable,str(__import__('pathlib').Path(__file__).with_name(name))],check=True)
    report=dict(passed=True,models=49,unique_model_sha256=state,unique_parameter_sha256=parameter_hashes,split=summary,
                batched_training_independence=batching, batched_noise_roundoff=noise_batching, portable_inference=portable,
                calibration='fit on calibration identities; complete future revealed only after issue',
                historical_test_caveat='test identities used during earlier exploratory research')
    (OUT/'independent_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print('AUDIT49_PASSED',flush=True)

if __name__=='__main__':main()
