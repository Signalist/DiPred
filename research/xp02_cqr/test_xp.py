"""CQR rank, sign, feature independence and interval projection checks."""
import torch
from xp_head import XPHead, BASE_INDICES
from evaluate_xp import conformal_rank

assert conformal_rank(99,.1)==90
assert conformal_rank(12,.004) is None
assert conformal_rank(1000,.004)==997
lower=torch.tensor([0.,0.,0.]);upper=torch.tensor([1.,1.,1.]);truth=torch.tensor([.5,-.2,1.4])
scores=torch.maximum(lower-truth,truth-upper)
assert torch.allclose(scores,torch.tensor([-.5,.2,.4]))
torch.manual_seed(42)
model=XPHead().eval()
base=torch.randn(1,14,20,4,8)
history=torch.randn(1,2,20,4,8)
with torch.no_grad():
    old=model(base,history)
    changed=base.clone()
    omitted=[i for i in range(14) if i not in BASE_INDICES]
    changed[:,omitted]+=1000
    new=model(changed,history)
assert all(torch.equal(a,b) for a,b in zip(old,new))
assert (old[0]<old[1]).all()
print('XP_CQR_TESTS_PASSED')
