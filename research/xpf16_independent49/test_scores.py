import torch
from scores import interval_metrics,distribution
q=torch.arange(1,100,dtype=torch.float32).reshape(99,1)
q98=torch.cat((q[:49],q[50:]),0)
y=torch.tensor([50.])
r=distribution(q98,y,True)
assert 0<=float(r['pit'])<=1 and float(r['crossing'])==0
assert float(r['crps'])>=0 and float(r['wis'])>=0
m=interval_metrics(torch.tensor([1.]),torch.tensor([3.]),torch.tensor([5.]),.1)
assert abs(float(m['interval_score'])-22)<1e-6
x=q98.clone(); x[[0,1]]=x[[1,0]]
assert float(distribution(x,y,False)['crossing'])>0
assert float(distribution(x,y,True)['crossing'])==0
assert float(distribution(x,y,True)['crossing_before'])>0
print('DISTRIBUTION_SCORE_TESTS_PASSED')
# For a degenerate forecast Q(tau)=2, CRPS is exactly |Y-2|.
constant=torch.full((98,3),2.0)
y=torch.tensor([-1.0,2.0,5.0])
d=distribution(constant,y,True)
assert torch.allclose(d['crps'],(y-2).abs(),atol=1e-6)
# Independent weighted-interval-score definition on a numerical example.
y=torch.tensor([7.])
q=torch.linspace(-1,8,98).reshape(98,1)
r=distribution(q,y,True)
med=(q[48]+q[49])/2
wis=.5*abs(y-med)
for k in range(1,50):
    a=k/100; lo=q[k-1]; hi=q[98-k]
    wis += a*(hi-lo+torch.relu(lo-y)/a+torch.relu(y-hi)/a)
assert torch.allclose(r['wis'],wis/49.5,atol=1e-6)
print('ANALYTICAL_CRPS_WIS_TESTS_PASSED')
# Q1..Q99 for an almost-uniform quantile function; Q0=Q1 creates a 1% atom.
q=torch.cat((torch.arange(1,50),torch.arange(51,100))).float().reshape(98,1)/100
for observation,expected in ((.5,.5),(.01,.005),(.005,0.),(.995,1.)):
    got=float(distribution(q,torch.tensor([observation]),True)['pit'])
    assert abs(got-expected)<1e-5,(observation,got,expected)
print('PIT_INVERSION_TESTS_PASSED')
