"""Matched 2x2 XP/XPF by fixed CQR or causal online calibration."""
import csv
import json
import math
import os
import sys
import time
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent.parent
for sub in ('xp02_cqr','xpf16_02grid_2048','xpf16_fullpercent_study','xpf16_cdf_fulltest','github/DiPred'):
    sys.path.insert(0,str(ROOT/sub))
from clean_reliability_common import configure,manifest,digest
from experiment_reliability_transport_e12 import geometry,history,target,normalization
from experiment_selected16_quantile import new_head
from reliability.selected16_predict import RequestedThreePredictor
from evaluate49 import prepare,make_head_batches,all_bounds
from evaluate_xp import load_batches as load_xp,predict as predict_xp,fields,conformal_rank
from calibration_grid import GridCalibrator,per_level_quantiles
from distribution_grid import rearrange_quantiles,quantile_distribution_scores
from cdf_stream import quantile_cdf_sum
from scores import interval_metrics

OUT=ROOT/'runs/controlled_xp_xpf_20261112'
XP=ROOT/'runs/xp02_cqr_20261112'
XPF=ROOT/'runs/xpf16_02grid_2048_20261112'
ARMS=('xp_fixed_cqr','xp_online','xpf_fixed_cqr','xpf_online','xp_raw','xpf_raw')
METRICS=('coverage','width','interval_score','lower_miss','upper_miss')
ALPHA=torch.arange(1,250,dtype=torch.float64)/500
PROBS=torch.tensor([k/500 for k in range(1,250)]+[1-k/500 for k in range(249,0,-1)],dtype=torch.float64)


def load_xpf():
    heads=[]
    for k in range(1,250):
        h=new_head('compact16',None)
        h.load_state_dict(torch.load(XPF/'models'/f'XPF16-P{k:03d}'/'model.pt',map_location='cuda',weights_only=False)['model'])
        heads.append(h.eval().requires_grad_(False))
    return make_head_batches(heads)


def predict_xpf(batches,base,last):
    pairs=all_bounds(batches,base,last)
    return torch.stack([x[0] for x in pairs]),torch.stack([x[1] for x in pairs])


def online_initial(blocks,levels):
    values=torch.cat(blocks,dim=1).permute(0,2,1,3,4,5).flatten(2)
    output=[]
    for start in range(0,249,8):
        output.append(per_level_quantiles(values[start:start+8].cuda(),levels[start:start+8].cuda()).cpu())
    return torch.cat(output)


def fixed_initial(blocks):
    values=torch.cat(blocks,dim=1).permute(0,2,1,3,4,5).flatten(2)
    n=values.shape[-1]
    result=[]
    for i in range(249):
        rank=conformal_rank(n,2*(i+1)/500)
        result.append(values[i].cuda().sort(dim=-1).values[:,rank-1].cpu())
    return torch.stack(result)


@torch.no_grad()
def fit(xp,xpf,norm):
    path=OUT/'calibration.pt'
    if path.exists():return torch.load(path,map_location='cuda',weights_only=False)
    records=manifest()['files']['calibration'];p=RequestedThreePredictor(geometry(records),'selected16','cuda')
    xp_left=[];xp_right=[];xpf_scores=[]
    for start in range(0,161,20):
        obs=history('calibration',start);h,point,base,last=prepare(p,obs)
        a,b=predict_xp(xp,base,h);c,d=predict_xpf(xpf,base,last)
        # Every predicted bound is finalized before the calibration target is read.
        residual=(target('calibration',start)-norm['mean'])/norm['std']-point
        xp_left.append((residual.unsqueeze(0)-a)[...,::4,::4].cpu())
        xp_right.append((residual.unsqueeze(0)-b)[...,::4,::4].cpu())
        xpf_scores.append(torch.maximum(c-residual.unsqueeze(0),residual.unsqueeze(0)-d)[...,::4,::4].cpu())
        print('CONTROL_CALIBRATION',start,flush=True)
    xp_init=torch.stack((online_initial(xp_left,ALPHA),online_initial(xp_right,1-ALPHA)),1)
    xpf_fixed=fixed_initial(xpf_scores)
    xp_fixed=torch.load(XP/'cqr_offsets.pt',map_location='cpu',weights_only=False)['offsets']
    xpf_init=torch.load(XPF/'calibration.pt',map_location='cpu',weights_only=False)['xpf']
    result=dict(xp_online=xp_init,xpf_online=xpf_init,xp_fixed_cqr=xp_fixed,xpf_fixed_cqr=xpf_fixed,
                calibration_trajectories=len(records),starts=list(range(0,161,20)),
                note='fixed CQR and online initial scores pooled over correlated 4x-subsampled pixels')
    tmp=path.with_suffix('.tmp');torch.save(result,tmp);os.replace(tmp,path)
    return torch.load(path,map_location='cuda',weights_only=False)


def fixed_bounds(lo,hi,offset):
    offset=offset[:,None,:,None,None,None]
    a,b=lo-offset,hi+offset
    invalid=a>b;mid=.5*(a+b)
    return torch.where(invalid,mid,a),torch.where(invalid,mid,b),invalid


def interval_summary(lo,hi,y):
    alpha=ALPHA.to(y.device,y.dtype)
    result=[]
    for i in range(0,249,12):
        a=alpha[i:i+12,None,None,None,None,None]
        m=interval_metrics(lo[i:i+12],hi[i:i+12],y.unsqueeze(0),a)
        result.append(torch.stack([m[k].mean((3,4,5)) for k in METRICS],dim=-1).cpu().numpy())
    return np.concatenate(result)


def plot_curves(out,grid,cdf):
    labels=('XP + fixed CQR','XP + online','XPF + fixed CQR','XPF + online','XP raw','XPF raw')
    for channel,c in (('u',0),('v',1)):
        fig,ax=plt.subplots(figsize=(7.4,5))
        for i,label in enumerate(labels):ax.plot(grid,cdf[i,c],label=label,lw=1.8)
        ax.axvline(0,color='gray',ls=':',lw=.7);ax.grid(alpha=.2);ax.legend()
        ax.set(xlabel='Forecast − truth (dataset original scale)',ylabel='Mean predictive CDF',title=f'{channel}: matched 2 × 2 control')
        fig.tight_layout();fig.savefig(out/f'cdf_{channel}.png',dpi=220);fig.savefig(out/f'cdf_{channel}.pdf');plt.close(fig)


@torch.no_grad()
def main():
    configure();OUT.mkdir(exist_ok=True,parents=True)
    assert (XP/'training_complete.json').exists() and (XPF/'training_complete.json').exists()
    norm=normalization();xp=load_xp();xpf=load_xpf();cal=fit(xp,xpf,norm)
    records=manifest()['files']['test'];p=RequestedThreePredictor(geometry(records),'selected16','cuda')
    controllers=[GridCalibrator(ALPHA.cuda(),cal[name],len(records)) for name in ('xp_online','xpf_online')]
    grid=torch.linspace(-.35,.35,701,device='cuda',dtype=torch.float64)
    sums=torch.zeros(len(ARMS),2,701,device='cuda',dtype=torch.float64)
    intervals=[];crps=[];cross=[];projections=[];starts=[];events=[];count=0
    for start in range(0,821,20):
        obs=fields('test',start,records);h,point,base,last=prepare(p,obs)
        a,b=predict_xp(xp,base,h);c,d=predict_xpf(xpf,base,last)
        af,bf,bad_xp=fixed_bounds(a,b,cal['xp_fixed_cqr'])
        ao,bo=controllers[0].issue(a,b)
        cf,df,bad_xpf=fixed_bounds(c,d,cal['xpf_fixed_cqr'])
        co,do=controllers[1].issue(c,d)
        # The whole next 20-frame truth is read after all four forecasts issue.
        events.append(dict(event='issued_all_six',start=start,time=time.time()))
        y=(fields('test',start,records,True)-norm['mean'])/norm['std']-point
        if start:
            irows=[];crows=[];xrows=[]
            for index,(lo,hi) in enumerate(((af,bf),(ao,bo),(cf,df),(co,do),(a,b),(c,d))):
                irows.append(interval_summary(lo,hi,y))
                q=torch.cat((lo,hi.flip(0)))
                qsorted,adjacent,_=rearrange_quantiles(q)
                xrows.append(float(adjacent.mean()))
                for channel in range(2):
                    e=((qsorted[:,:,channel].double()-y[:,channel].double().unsqueeze(0))*norm['std'][0,channel,0,0,0].double()).flatten(1)
                    sums[index,channel]+=quantile_cdf_sum(e,PROBS,grid)
                midpoint=.5*(qsorted[248]+qsorted[249])
                q499=torch.cat((qsorted[:249],midpoint.unsqueeze(0),qsorted[249:]))
                score,_=quantile_distribution_scores(q499,y)
                crows.append(score.mean((2,3,4)).cpu().numpy())
                del q,qsorted,q499,score
            intervals.append(np.stack(irows));crps.append(np.stack(crows));cross.append(xrows)
            projections.append([float(bad_xp.float().mean()),float(bad_xpf.float().mean())])
            starts.append(start);count+=y[:,0].numel()
        controllers[0].reveal(y);controllers[1].reveal(y)
        events.append(dict(event='revealed_complete_20',start=start,time=time.time()))
        print('CONTROL_TEST',start,flush=True)
    assert starts==list(range(20,821,20)) and count==len(records)*41*20*32*64
    intervals=np.stack(intervals);crps=np.stack(crps);cross=np.array(cross);cdf=(sums/count).cpu().numpy();x=grid.cpu().numpy()
    (OUT/'events.json').write_text(json.dumps(events,indent=2)+'\n')
    assert np.isfinite(cdf).all() and (np.diff(cdf,axis=-1)>=-1e-7).all()
    np.savez_compressed(OUT/'results.npz',x=x,cdf=cdf,intervals=intervals,crps=crps,crossing=cross,projections=projections,starts=starts,arms=ARMS,metrics=METRICS)
    with (OUT/'intervals.csv').open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['method','lower_probability','target_coverage','channel',*METRICS])
        for a,name in enumerate(ARMS):
            for i in range(249):
                for c,channel in enumerate(('u','v')):writer.writerow([name,(i+1)/500,1-2*(i+1)/500,channel,*intervals[:,a,i,:,c].mean((0,1))])
    with (OUT/'distribution.csv').open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['method','channel','approx_crps','adjacent_crossing'])
        for a,name in enumerate(ARMS):
            for c,channel in enumerate(('u','v')):writer.writerow([name,channel,crps[:,a,:,c].mean(),cross[:,a].mean()])
    plot_curves(OUT,x,cdf)
    report=dict(passed=True,arms=ARMS,observations_per_channel=count,starts=starts,calibration_trajectories=cal['calibration_trajectories'],
       cqr='fixed separate-per-channel Romano nonconformity quantile, voxel-time pooled; no exchangeability guarantee',
       online='same alpha-dependent two-tail adaptive rule, .5 gain, independent per-model trajectory-channel state',
       notes='no model retraining; same DiPred and same 41 post-cold-start test windows; XP and XPF differ in architecture, initialization and inputs; comparison does not isolate F alone',
       cdf='498 sorted quantiles, linear probability interpolation, endpoint atoms; median interpolated for equal CRPS quadrature',
       fixed_projected_interval_fraction=np.mean(projections,axis=0).tolist(),mean_adjacent_crossing=cross.mean(axis=0).tolist(),
       records=records,mean_crps=crps.mean(axis=(0,2)).tolist(),source_sha256={str(p.relative_to(ROOT)):digest(p) for p in (Path(__file__).resolve(),XP/'training_complete.json',XPF/'training_complete.json')})
    (OUT/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print('CONTROL_EVALUATION_COMPLETE',flush=True)

if __name__=='__main__':main()
