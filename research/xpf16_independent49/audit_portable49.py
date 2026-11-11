"""Match portable Q01/Q05/Q49 predictions and feedback to evaluation replay."""
import json,sys
import numpy as np
import torch
from clean_reliability_common import ROOT,configure,manifest
from experiment_reliability_transport_e12 import geometry,history,target
from scores import interval_metrics
from train49 import OUT
sys.path.insert(0,str(ROOT/'github/DiPred'))
from reliability.independent49_predict import XPFIndependentPredictor

configure()
records=manifest()['files']['development']
p=XPFIndependentPredictor(geometry(records),OUT/'portable_weights',OUT/'initial_offsets.pt',levels=(1,5,49))
maximum=0.
differences={}
for start in (0,20):
    r=p.predict(history('development',start))
    truth=target('development',start)
    residual=(truth-p.norm['mean'])/p.norm['std']-p.pending
    reference=np.load(OUT/'evaluation/development'/f'{start:04d}.npz')['metrics']
    for j,k in enumerate((1,5,49)):
        for mode,(a,b) in enumerate((('raw_lower','raw_upper'),('calibrated_lower','calibrated_upper'))):
            lo=(r[a][:,j]-r['prediction'])/p.norm['std']
            hi=(r[b][:,j]-r['prediction'])/p.norm['std']
            scores=interval_metrics(lo,hi,residual,k/100)
            for i,key in enumerate(('lower_coverage','upper_coverage','coverage','width','interval_score','lower_pinball','upper_pinball','lower_miss','upper_miss')):
                value=scores[key].mean((-1,-2)).cpu().numpy()
                error=float(np.max(np.abs(value-reference[mode,0,k-1,:,:,:,i])))
                differences[key]=max(differences.get(key,0),error)
                maximum=max(maximum,error)
    p.reveal(truth)
print('per-metric differences',differences)
assert maximum<=1/2048+1e-7,maximum
(OUT/'portable49_audit.json').write_text(json.dumps(dict(passed=True,levels=[1,5,49],windows=[0,20],max_metric_difference=maximum,per_metric=differences),indent=2)+'\n')
print('PORTABLE49_AUDIT_PASSED',maximum)
