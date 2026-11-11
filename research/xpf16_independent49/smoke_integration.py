"""Development-only integration smoke test; not a formal study result."""
import sys
import torch
from clean_reliability_common import ROOT, configure, manifest
from experiment_reliability_transport_e12 import geometry, normalization, history, target
from experiment_fullblock_memory_risk import OUT as MEMORY, load
from experiment_selected16_quantile import new_head
from train49 import OUT
from calibration import TwoTailCalibrator
from scores import interval_metrics
from evaluate49 import prepare, empirical
sys.path.insert(0,str(ROOT/'github/DiPred'))
from reliability.selected16_predict import RequestedThreePredictor

configure()
records=manifest()['files']['development']
p=RequestedThreePredictor(geometry(records),'selected16','cuda')
norm=normalization()
h,point,base,last=prepare(p,history('development',0))
raw=[]
for k in (1,5):
    head=new_head('compact16',load(MEMORY/'statistics.pt')['memory_scale'])
    head.load_state_dict(load(OUT/'models'/f'XPF16-Q{k:02d}'/'model.pt')['model'])
    with torch.no_grad():
        lo,hi=head.bounds(base,last,None,None,False)
    control=TwoTailCalibrator(k/100,torch.zeros(2,2,device='cuda'),len(records))
    issued=control.issue(lo,hi)
    raw.append((control,issued))
with torch.no_grad():
    paths=torch.stack([p.model.rollout(h,seed,sign=1) for seed in range(500000,500256)])
    for n in (8,32,256):
        lo=empirical(paths[:n],.01)
        hi=empirical(paths[:n],.99)
        assert torch.isfinite(lo).all() and torch.all(lo<=hi)
truth=(target('development',0)-norm['mean'])/norm['std']-point
for control,(lo,hi) in raw:
    m=interval_metrics(lo,hi,truth,control.alpha)
    assert all(torch.isfinite(v).all() for v in m.values())
    control.reveal(truth)
print('INTEGRATION_SMOKE_PASSED: development Q01,Q05 and actual 8/32/256 paths')
