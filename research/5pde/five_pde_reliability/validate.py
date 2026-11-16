"""Fixed-protocol, nonselective development assessment before any test replay."""
import argparse,json,sys,time
import numpy as np
import torch
from common import OUT,Scenario,configure,predict,records
from evaluate import ARMS,MASKS,load_head

@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('scenario');args=parser.parse_args()
    configure();name=args.scenario;scenario=Scenario(name);rs=[r for r in records(name) if r['split']=='development']
    backbone=scenario.network().eval().requires_grad_(False)
    backbone.load_state_dict(torch.load(OUT/'backbones'/name/'model.pt',map_location='cuda',weights_only=False)['model'])
    max_start=(min(r['frames'] for r in rs)-40)//20
    starts=sorted({int(round(v))*20 for v in np.linspace(0,max_start,9)})
    models={(arm,seed):load_head(name,arm,seed) for seed in (42,123,2026) for arm in ARMS}
    scores={(arm,seed):[] for arm,seed in models}
    for start in starts:
        x=scenario.batch(rs,[start]*len(rs));p,f=predict(backbone,x,scenario.mask,True)
        forecasts={}
        for (arm,seed),(head,state,mean,std) in models.items():
            feature=((f-mean)/std).clamp(-10,10)
            if arm!='XPF':feature=torch.zeros_like(feature)
            point=torch.zeros_like(p) if arm=='X' else p
            mask=torch.tensor(MASKS[arm],device='cuda').expand(len(x),2)
            forecasts[(arm,seed)]=head.quantiles(x,point,feature,mask,torch.tensor([.05,.5,.95],device='cuda'))
        # Only after all nine models issue is the future observed.
        y=scenario.batch(rs,[start]*len(rs),True)
        tau=torch.tensor([.05,.5,.95],device='cuda')[:,None,None,None,None,None]
        for key,q in forecasts.items():
            e=y[None]-q
            value=(torch.maximum(tau*e,(tau-1)*e)*scenario.std)[:,:,:scenario.channels].mean()
            scores[key].append(float(value))
    output=dict(scenario=name,selection='none; all fixed final checkpoints',
        trajectory_ids=[r['id'] for r in rs],starts=starts,
        mean_pinball_05_50_95={f'{arm}_{seed}':float(np.mean(value)) for (arm,seed),value in scores.items()},time=time.time())
    (OUT/'validation').mkdir(parents=True,exist_ok=True)
    (OUT/'validation'/f'{name}.json').write_text(json.dumps(output,indent=2)+'\n')
    print('VALIDATION_COMPLETE',name,flush=True)

if __name__=='__main__':main()
