"""Check all 499 levels and sample CRPS against independent formulas."""
import torch
from distribution_grid import quantiles_and_pair,sample_crps

torch.manual_seed(42)
paths=torch.randn(2048,2,2,3)
point=torch.randn(2,2,3)
truth=torch.randn_like(point)
scale=torch.rand_like(point)+.1
levels=torch.arange(1,500,dtype=torch.float64)/500
q,pair=quantiles_and_pair(paths,point,levels,chunk=5)
ref=torch.quantile(paths,levels.float(),dim=0)-point
assert torch.allclose(q,ref,atol=3e-6)
assert torch.all(q[1:]>=q[:-1])
result=sample_crps(paths,point,truth,scale,pair,chunk=5)
x=paths[:,0,0,0].double()-point[0,0,0].double();y=truth[0,0,0].double();g=scale[0,0,0].double()
pairs=(x[:,None]-x[None,:]).abs().sum();n=len(x)
expected=((x-y).abs().mean()-pairs/(2*n*n),(x-y).abs().mean()-pairs/(2*n*(n-1)),
          (x*g-y).abs().mean()-g*pairs/(2*n*n),(x*g-y).abs().mean()-g*pairs/(2*n*(n-1)))
for k,v in zip(('raw_empirical','raw_fair','proxy_empirical','proxy_fair'),expected):
    assert abs(float(result[k][0,0,0]-v))<1e-6
print('GRID_499_QUANTILES_AND_2048_CRPS_TESTS_PASSED')
from distribution_grid import rearrange_quantiles,quantile_distribution_scores
v=torch.arange(1,500,dtype=torch.float64).reshape(499,1)/500
v[[0,1]]=v[[1,0]]
sorted_q,cross,any_cross=rearrange_quantiles(v,chunk=1)
assert float(cross)>0 and float(any_cross)==1 and torch.all(sorted_q[1:]>=sorted_q[:-1])
constant=torch.full((499,2),2.0)
y=torch.tensor([-1.0,5.0])
crps,wis=quantile_distribution_scores(constant,y)
assert torch.allclose(crps,(y-2).abs(),atol=1e-5)
assert torch.isfinite(wis).all()
print('GRID_REARRANGEMENT_AND_QUANTILE_CRPS_TESTS_PASSED')
from distribution_grid import quantile_pit
q=(torch.arange(1,500,dtype=torch.float32)/500).reshape(499,1)
for y,expected in ((.5,.5),(.002,.001),(.001,0.),(.999,1.)):
    actual=float(quantile_pit(q,torch.tensor([y]),chunk=1))
    assert abs(actual-expected)<1e-6,(y,actual,expected)
print('GRID_PIT_INVERSION_TESTS_PASSED')
