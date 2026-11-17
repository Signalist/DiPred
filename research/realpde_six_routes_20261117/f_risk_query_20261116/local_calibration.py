"""Predefined overlapping-group calibration; empirical pilot, no guarantee."""
from common import *
from risk import RiskHead,features
from evaluate_risk import groups
import csv
@torch.no_grad()
def main():
    setup();cal=cache('calibration');dev=cache('development');threshold=load(OUT/'gradient_threshold.pt','cuda');std=norm()['std'];levels=torch.tensor([.05,.95],device='cuda')
    folder=OUT/'local';folder.mkdir(exist_ok=True)
    dump(folder/'protocol.json',dict(shapes=['constant','P','F','XPF'],strategies=['global','groups'],coverage_targets=[.8,.85,.9,.95],fit_split='calibration',evaluate_split='development',group_definitions='same 10 prediction-defined groups as audit',method='maximum calibrated multiplicative scale over overlapping groups; no new-method or formal guarantee claim'))
    rows=[];parameters={};cmasks=groups(cal['p'],threshold);dmasks=groups(dev['p'],threshold)
    for arm in ('constant','P','F','XPF'):
        if arm=='constant':cl=ch=torch.ones_like(cal['p']);dl=dh=torch.ones_like(dev['p'])
        else:
            ck=load(OUT/'risk/main'/f'{arm}.pt');m=RiskHead(ck['width']).cuda().eval();m.load_state_dict(ck['model'])
            ab=[]
            for data in (cal,dev):
                chunks=[]
                for i in range(0,len(data['x']),6):
                    ids=torch.arange(i,min(i+6,len(data['x'])),device='cuda');chunks.append(m.quantiles(*features(data,arm,ids),levels))
                q=torch.cat(chunks,dim=1);ab.append(((data['p']-q[0]).clamp_min(1e-5),(q[1]-data['p']).clamp_min(1e-5)))
            (cl,ch),(dl,dh)=ab
        residual=cal['y']-cal['p'];score=torch.where(residual>=0,residual/ch,-residual/cl)
        for target in (.8,.85,.9,.95):
            factors={}
            for group,sel in cmasks.items():
                factors[group]=torch.stack([torch.quantile(score[:,c][sel[:,c]],target) for c in range(2)])
            for strategy in ('global','groups'):
                factor=torch.zeros_like(dev['p'])
                for group in (['all'] if strategy=='global' else list(dmasks)):
                    v=factors[group][None,:,None,None,None];factor=torch.maximum(factor,torch.where(dmasks[group],v,torch.zeros_like(v)))
                lo=dev['p']-dl*factor;hi=dev['p']+dh*factor;y=dev['y'];metrics=dict(coverage=((y>=lo)&(y<=hi)).float(),width=(hi-lo),interval=(hi-lo)+20*((lo-y).clamp_min(0)+(y-hi).clamp_min(0)))
                parameters[f'{arm}_{strategy}_{target}']={g:v.tolist() for g,v in factors.items()}
                for group,sel in dmasks.items():
                    counts=sel.sum((2,3,4));values={k:((v*sel).sum((2,3,4))/counts.clamp_min(1)).cpu().numpy() for k,v in metrics.items()};counts=counts.cpu().numpy()
                    for i,meta in enumerate(dev['meta']):
                        for c in range(2):
                            if counts[i,c]:rows.append(dict(shape=arm,strategy=strategy,target=target,region=group,channel=c,**meta,**{k:float(v[i,c]) for k,v in values.items()}))
        print('LOCAL_CALIBRATION',arm,flush=True)
    with (folder/'per_group.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    dump(folder/'parameters.json',parameters)
if __name__=='__main__':main()
