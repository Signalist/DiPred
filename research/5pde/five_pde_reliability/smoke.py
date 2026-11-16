"""Small development-only shape, causality and numerical pipeline check."""
import json,sys,torch
from common import OUT,Scenario,configure,predict,records
from evaluate import ALPHA,LEVELS,bounds,quantile_crps,sorted_cdf_sum
from calibration_grid import GridCalibrator,MedianCalibrator
from distribution_grid import rearrange_quantiles
from fair_x_xp_xpf.model import FairQuantileHead

@torch.no_grad()
def main():
    configure();name='cylinder';s=Scenario(name);rs=[r for r in records(name) if r['split']=='development'][:2]
    p=s.network().eval().requires_grad_(False);p.load_state_dict(torch.load(OUT/'backbones'/name/'model.pt',map_location='cuda',weights_only=False)['model'])
    x=s.batch(rs,[0,0]);point,feature=predict(p,x,s.mask,True)
    torch.manual_seed(1);head=FairQuantileHead().cuda().eval();mask=torch.ones(2,2,device='cuda')
    q=head.quantiles(x,point,feature,mask,LEVELS.cuda().float())
    cal=dict(fixed=torch.zeros(249,2,device='cuda'))
    controller=GridCalibrator(ALPHA.cuda(),torch.zeros(249,2,2,device='cuda'),2)
    median=MedianCalibrator(torch.zeros(2,device='cuda'),2)
    forecasts,_=bounds(q,cal,controller,median)
    y=s.batch(rs,[0,0],True)
    for values in forecasts:
        ordered,crossing,_=rearrange_quantiles(values)
        score=quantile_crps(ordered,y,LEVELS.cuda().float())
        grid=torch.linspace(-.3,.3,11,device='cuda',dtype=torch.float64)
        cdf=sorted_cdf_sum((ordered[:,:,0]-y[:,0][None]).flatten(1),LEVELS,grid)
        assert score.isfinite().all() and cdf.isfinite().all() and (cdf[1:]>=cdf[:-1]-1e-5).all()
    controller.reveal(y);median.reveal(y)
    print('SMOKE_OK',tuple(q.shape),controller.updates,flush=True)

if __name__=='__main__':main()
