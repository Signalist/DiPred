"""Fit 256-path online proxy initial state on calibration identities only."""
import sys
import torch
from clean_reliability_common import ROOT, configure, manifest
from experiment_reliability_transport_e12 import geometry, normalization, history, target
from causal_shape_calibration import initial_state
from train49 import OUT
from evaluate49 import prepare
sys.path.insert(0,str(ROOT/'github/DiPred'))
from reliability.selected16_predict import RequestedThreePredictor

@torch.no_grad()
def main():
    configure()
    path=OUT/'noise256_calibration.pt'
    if path.exists():
        print('CALIBRATION_256_EXISTS',flush=True);return
    norm=normalization();records=manifest()['files']['calibration']
    p=RequestedThreePredictor(geometry(records),'selected16','cuda')
    ratios=[]
    for start in range(0,161,20):
        observed=history('calibration',start)
        h,point,base,last=prepare(p,observed)
        paths=torch.stack([p.model.rollout(h,seed,sign=1) for seed in range(500000,500256)])
        radius=1.6448536269514722*paths.std(0).clamp_min(1e-6)
        truth=(target('calibration',start)-norm['mean'])/norm['std']
        ratios.append(((truth-point).abs()/radius).cpu())
        print('CALIBRATION_256',start,flush=True)
    state=initial_state(torch.cat(ratios).cuda(),'separable_integral')
    torch.save({k:v.cpu() for k,v in state.items()},path)
    print('CALIBRATION_256_COMPLETE',flush=True)

if __name__=='__main__':main()
