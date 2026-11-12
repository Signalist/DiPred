"""Verify vectorized grid produces the same results as separate controllers."""
import sys
import torch
from clean_reliability_common import ROOT
from calibration_grid import GridCalibrator
sys.path.insert(0,str(ROOT/'xpf16_fullpercent_study'))
from calibration import TwoTailCalibrator

torch.manual_seed(42)
a=torch.tensor([.002,.01,.25,.498])
initial=torch.randn(4,2,2)*.1
multiple=GridCalibrator(a,initial,2)
singles=[TwoTailCalibrator(float(v),initial[i],2) for i,v in enumerate(a)]
for _ in range(2):
    lower=torch.randn(4,2,2,20,8,12)
    upper=lower+.3
    target=torch.randn(2,2,20,8,12)
    ml,mu=multiple.issue(lower,upper)
    for i,s in enumerate(singles):
        sl,su=s.issue(lower[i],upper[i])
        assert torch.equal(sl,ml[i]) and torch.equal(su,mu[i])
    multiple.reveal(target)
    for s in singles:s.reveal(target)
    for i,s in enumerate(singles):
        assert torch.allclose(s.offsets,multiple.offsets[i],atol=3e-6)
print('GRID_CALIBRATION_MATCHES_SEPARATE_STATES')
