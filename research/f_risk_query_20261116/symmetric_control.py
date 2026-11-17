from common import *
from risk import train
from evaluate_risk import run
setup()
dump(OUT/'symmetric_protocol.json',dict(seed=42,arms=['F','P'],steps=2000,width=16,hypothesis='Both the median and the symmetric quantile-distribution mean equal P; separate pure scale/tail shape from skew-induced mean shift.',quantile_basis='(basis(tau)-basis(1-tau))/2',development_checkpoint_selection=False))
train(('F','P'),16,2000,'symmetric',1.,True)
run('symmetric')
