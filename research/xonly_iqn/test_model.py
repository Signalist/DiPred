"""Verify X-only quantile shapes, conditioning and forecast-only feedback."""
import sys,torch
from pathlib import Path
from model import XOnlyIQN,quantile_huber
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT/'xpf16_02grid_2048'))
from calibration_grid import GridCalibrator

torch.manual_seed(42)
model=XOnlyIQN().eval()
x=torch.randn(2,2,20,4,8)
with torch.no_grad():
    a=model(x,torch.tensor([.1,.1]));b=model(x,torch.tensor([.9,.9]))
assert a.shape==x.shape and not torch.equal(a,b)
tau=torch.tensor([.1,.9]);loss=quantile_huber(a,x,tau)
assert torch.isfinite(loss) and loss>=0
cal=GridCalibrator(torch.tensor([.05],dtype=torch.float64),torch.zeros(1,2,2),2)
lo=torch.zeros(1,2,2,20,4,8);hi=torch.ones_like(lo)
cal.issue(lo,hi)
try:
    cal.issue(lo,hi)
    raise AssertionError('Future truth was not required')
except RuntimeError:pass
cal.reveal(torch.zeros(2,2,20,4,8));assert cal.updates==1
print('XONLY_IQN_TESTS_PASSED')
