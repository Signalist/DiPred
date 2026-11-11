"""Deterministic unit checks for levels, scores, and online causality."""
import torch
from calibration import TwoTailCalibrator
from train49 import configs, pinball

cs=configs()
assert len(cs)==49 and cs[0]['id']=='XPF16-Q01' and cs[-1]['id']=='XPF16-Q49'
for k,c in enumerate(cs,1):
    assert abs(c['lower']-k/100)<1e-12 and abs(c['upper']-(1-k/100))<1e-12
    assert abs(c['target_coverage']-(1-2*k/100))<1e-12
x=torch.arange(40.).reshape(1,2,20,1,1)
for k in (1,5,25,49):
    c=TwoTailCalibrator(k/100,torch.zeros(2,2),1)
    l,u=c.issue(torch.zeros_like(x),torch.ones_like(x))
    try:
        c.issue(l,u)
        raise AssertionError('issue before reveal accepted')
    except RuntimeError:
        pass
    try:
        c.reveal(x[:,:,:19])
        raise AssertionError('partial reveal accepted')
    except RuntimeError:
        pass
    c.reveal(x)
    l,u=c.issue(torch.zeros_like(x),torch.ones_like(x))
    assert torch.all(l<=u)
    c.reveal(x)
assert torch.equal(pinball(torch.tensor([-2.,1.]),.2),torch.tensor([1.6,.2]))
print('PROTOCOL_TESTS_PASSED')
