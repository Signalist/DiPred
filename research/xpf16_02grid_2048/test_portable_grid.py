"""Development smoke for portable subset; fake median only in scratch space."""
import sys,torch
from clean_reliability_common import ROOT,configure,manifest
from experiment_reliability_transport_e12 import geometry,history,target
from train_grid import OUT
sys.path.insert(0,str(ROOT/'github/DiPred'))
from reliability.grid02_predict import Grid02Predictor

configure();tmp=OUT/'portable_smoketest';tmp.mkdir(exist_ok=True)
for k in (1,5):
    a=torch.load(OUT/'models'/f'XPF16-P{k:03d}'/'model.pt',map_location='cpu',weights_only=False)
    torch.save(a,tmp/f'XPF16-P{k:03d}.pt')
a=torch.load(OUT/'models/XPF16-P005/model.pt',map_location='cpu',weights_only=False)
a['config']['lower']=.5
torch.save(a,tmp/'XPF16-P250.pt')
torch.save({'xpf':torch.zeros(249,2,2),'xpf_median':torch.zeros(2)},tmp/'calibration.pt')
records=manifest()['files']['development'][:2]
p=Grid02Predictor(geometry(records),tmp,calibration=tmp/'calibration.pt',tail_ids=(1,5))
for start in (0,20):
    r=p.predict(history('development',start)[:2])
    assert len(r['levels'])==5 and r['levels']==(.002,.01,.5,.99,.998)
    assert r['raw_quantiles'].shape==(2,5,2,20,32,64)
    assert torch.isfinite(r['calibrated_quantiles']).all()
    p.reveal(target('development',start)[:2])
assert p.grid_controller.updates==2 and p.median_controller.updates==2
print('PORTABLE_GRID_SUBSET_SMOKE_PASSED; median weights are intentionally fake and unpublished')
