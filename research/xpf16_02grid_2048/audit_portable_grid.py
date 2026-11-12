"""Compare portable subset with the issued development stream and feedback."""
import json,sys
import numpy as np
import torch
from clean_reliability_common import ROOT,configure,manifest
from experiment_reliability_transport_e12 import geometry,history,target
from train_grid import OUT
sys.path.insert(0,str(ROOT/'xpf16_fullpercent_study'))
from scores import pinball
sys.path.insert(0,str(ROOT/'github/DiPred'))
from reliability.grid02_predict import Grid02Predictor

configure();tmp=OUT/'portable_weights';tmp.mkdir(exist_ok=True)
ids=(1,5,249,250)
for k in ids:
    dest=tmp/f'XPF16-P{k:03d}.pt'
    if not dest.exists():dest.symlink_to((OUT/'models'/f'XPF16-P{k:03d}'/'model.pt').resolve())
records=manifest()['files']['development']
p=Grid02Predictor(geometry(records),tmp,calibration=OUT/'calibration.pt',tail_ids=ids[:-1])
indices=(0,4,248,249,250,494,498)
err_cov=0.;err_pin=0.
for start in (0,20):
    out=p.predict(history('development',start))
    truth=target('development',start)
    residual=(truth-p.norm['mean'])/p.norm['std']-p.pending
    ref=np.load(OUT/'evaluation/development'/f'{start:04d}.npz')['marginals']
    for arm,key in enumerate(('raw_quantiles','calibrated_quantiles')):
        q=(out[key]-out['prediction'].unsqueeze(1))/p.norm['std'].unsqueeze(1)
        for j,ix in enumerate(indices):
            e=residual-q[:,j]
            tau=(ix+1)/500
            coverage=(e<=0).float().mean((-1,-2)).cpu().numpy()
            loss=pinball(e,tau).mean((-1,-2)).cpu().numpy()
            err_cov=max(err_cov,float(np.max(np.abs(coverage-ref[arm,ix,:,:,:,0]))))
            err_pin=max(err_pin,float(np.max(np.abs(loss-ref[arm,ix,:,:,:,1]))))
    p.reveal(truth)
assert err_cov<=1/2048+1e-7 and err_pin<2e-6,(err_cov,err_pin)
(OUT/'portable_audit.json').write_text(json.dumps(dict(passed=True,tail_ids=ids[:-1],median=True,windows=[0,20],
    max_coverage_difference=err_cov,max_pinball_difference=err_pin,note='One grid point tolerance covers floating-point batch-order effects.'),indent=2)+'\n')
print('GRID_PORTABLE_AUDIT_PASSED',err_cov,err_pin)
