"""Fit split-CQR on calibration trajectories; stream full test mean CDF."""
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from functorch import combine_state_for_ensemble, vmap
from xp_head import XPHead

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'xpf16_fullpercent_study'))
sys.path.insert(0,str(ROOT/'xpf16_02grid_2048'))
sys.path.insert(0,str(ROOT/'xpf16_cdf_fulltest'))
sys.path.insert(0,str(ROOT/'github/DiPred'))
from clean_reliability_common import configure, manifest, dump, digest
from experiment_reliability_transport_e12 import geometry, history, target, normalization
from export_aligned_test_distribution import from_hdf
from evaluate49 import prepare
from reliability.selected16_predict import RequestedThreePredictor
from cdf_stream import quantile_cdf_sum

OUT=ROOT/'runs/xp02_cqr_20261112'
REFERENCE=ROOT/'github/DiPred/results/reliability/xpf16_02grid_2048/full_test_cdf'
PROBS=torch.tensor([k/500 for k in range(1,250)]+[1-k/500 for k in range(249,0,-1)],dtype=torch.float64)


def conformal_rank(n,alpha):
    """Romano CQR order statistic; alpha is the total interval miscoverage."""
    rank=math.ceil((n+1)*(1-alpha))
    return rank if rank<=n else None


def load_batches():
    heads=[]
    for k in range(1,250):
        h=XPHead().cuda().eval().requires_grad_(False)
        h.load_state_dict(torch.load(OUT/'models'/f'XP-P{k:03d}'/'model.pt',map_location='cuda',weights_only=False)['model'])
        heads.append(h)
    batches=[]
    for i in range(0,len(heads),7):
        f,p,b=combine_state_for_ensemble(heads[i:i+7])
        batches.append((f,p,b))
    return batches


def predict(batches,base,history_fields):
    pairs=[]
    for f,parameters,buffers in batches:
        lower,upper=vmap(lambda p,b:f(p,b,base,history_fields),in_dims=(0,0))(parameters,buffers)
        pairs.append((lower,upper))
    return torch.cat([p[0] for p in pairs]),torch.cat([p[1] for p in pairs])


def fields(split,start,records,future=False):
    if start<=760:return target(split,start) if future else history(split,start)
    return from_hdf(records,start+(20 if future else 0))


@torch.no_grad()
def calibrate(batches,norm):
    file=OUT/'cqr_offsets.pt'
    if file.exists():return torch.load(file,map_location='cuda',weights_only=False)
    records=manifest()['files']['calibration']
    p=RequestedThreePredictor(geometry(records),'selected16','cuda')
    blocks=[]
    for start in range(0,161,20):
        observed=history('calibration',start)
        h,point,base,_=prepare(p,observed)
        lo,hi=predict(batches,base,h)
        # Future data are read only after the frozen XP prediction exists.
        y=(target('calibration',start)-norm['mean'])/norm['std']-point
        scores=torch.maximum(lo-y.unsqueeze(0),y.unsqueeze(0)-hi)[...,::4,::4]
        blocks.append(scores.cpu())
        print('XP_CQR_CALIBRATION',start,flush=True)
    # Per-channel pooled observations are dependent; this is an empirical
    # pointwise approximation, not a trajectory-exchangeability guarantee.
    values=torch.cat(blocks,dim=1).permute(0,2,1,3,4,5).flatten(2)
    n=values.shape[-1];offsets=[]
    for k in range(1,250):
        rank=conformal_rank(n,2*k/500)
        if rank is None:raise RuntimeError('Insufficient calibration samples')
        sorted_scores=values[k-1].cuda().sort(dim=-1).values
        offsets.append(sorted_scores[:,rank-1].cpu())
    offsets=torch.stack(offsets).cuda()
    tmp=file.with_suffix('.tmp')
    torch.save(dict(offsets=offsets,n=n,calibration_trajectories=len(records),
                    scores='max(lower-y, y-upper)',correlated_pool=True),tmp)
    os.replace(tmp,file)
    return torch.load(file,map_location='cuda',weights_only=False)


def test_curves(curves):
    for c in curves:
        if not np.isfinite(c).all() or np.diff(c).min() < -1e-7 or c.min() < -1e-7 or c.max() > 1+1e-7:
            raise RuntimeError('Invalid CDF')


def plot(output,channel,x,curves):
    fig,ax=plt.subplots(figsize=(7.4,5.2))
    colors=('#b31e31','#237ab6','#9887ad','#50a886','#d28b34','#656565','#000000')
    labels=('XPF16-1','XPF16-0.2','8-noise','32-noise','256-noise','2048-noise','XP-0.2 + CQR')
    for i,(name,label) in enumerate(zip(curves,labels)):
        ax.plot(x,curves[name],label=label,color=colors[i],lw=2.4 if name=='xp02_cqr' else 1.4)
    ax.axvline(0,color='gray',lw=.7,ls=':')
    ax.set(xlabel='Forecast value − truth (dataset original scale)',ylabel='Full-test mean CDF',title=f'{channel} channel: online XPF/noise vs fixed XP + CQR')
    ax.legend(fontsize=8);ax.grid(alpha=.2);fig.tight_layout()
    fig.savefig(output/f'cdf_{channel}.png',dpi=240);fig.savefig(output/f'cdf_{channel}.pdf');plt.close(fig)


@torch.no_grad()
def main():
    configure()
    assert (OUT/'training_complete.json').exists()
    output=OUT/'results';output.mkdir(exist_ok=True)
    norm=normalization();batches=load_batches();cal=calibrate(batches,norm)
    correction=cal['offsets'][:,None,:,None,None,None].cuda()
    records=manifest()['files']['test']
    p=RequestedThreePredictor(geometry(records),'selected16','cuda')
    grid=torch.linspace(-.35,.35,701,dtype=torch.float64,device='cuda')
    sums=torch.zeros((2,len(grid)),device='cuda',dtype=torch.float64)
    count=0;rows=[];total_cross=0;total_locations=0;projected=0;total_pairs=0
    for start in range(0,821,20):
        observed=fields('test',start,records)
        h,point,base,_=prepare(p,observed)
        lo,hi=predict(batches,base,h)
        # Classical fixed split-CQR applies the same scalar to both tails.
        lower=lo-correction;upper=hi+correction
        bad=lower>upper;projected+=int(bad.sum());total_pairs+=bad.numel()
        mid=.5*(lower+upper)
        lower=torch.where(bad,mid,lower);upper=torch.where(bad,mid,upper)
        truth=(fields('test',start,records,True)-norm['mean'])/norm['std']
        residual=truth-point
        if start:
            quantiles=torch.cat((lower,upper.flip(0)))
            total_cross+=int((quantiles[1:]<quantiles[:-1]).sum());total_locations+=quantiles[0].numel()
            for channel in range(2):
                q=((quantiles[:,:,channel].double()-residual[:,channel].double().unsqueeze(0))*norm['std'][0,channel,0,0,0].double()).flatten(1)
                sums[channel]+=quantile_cdf_sum(q,PROBS,grid)
            cover=((residual.unsqueeze(0)>=lower)&(residual.unsqueeze(0)<=upper)).float().mean((1,3,4,5))
            rows.append(dict(start=start,coverage=cover.cpu().tolist()))
            count+=residual[:,0].numel()
        print('XP_CQR_TEST',start,flush=True)
    assert count==len(records)*41*20*32*64
    x=grid.cpu().numpy();curves=(sums/count).cpu().numpy();test_curves(curves)
    for channel,i in (('u',0),('v',1)):
        ref=np.load(REFERENCE/f'cdf_{channel}.npz')
        assert np.array_equal(x,ref['x']) and ref['count']==count
        old=('xpf16_1','xpf16_02','noise8','noise32','noise256','noise2048')
        combined={key:ref[key] for key in old};combined['xp02_cqr']=curves[i]
        np.savez_compressed(output/f'cdf_{channel}.npz',x=x,count=count,**combined)
        np.savetxt(output/f'cdf_{channel}.csv',np.column_stack([x]+list(combined.values())),delimiter=',',
                   header=','.join(['x']+list(combined)),comments='')
        plot(output,channel,x,combined)
    dump(output/'summary.json',dict(passed=True,method='XP-0.2 + fixed split CQR (Romano et al. 2019)',
         heads=249,probabilities=498,calibration_samples_per_channel=cal['n'],calibration_trajectories=cal['calibration_trajectories'],
         quantile_unit='correlated voxel-time pooled per channel; no formal conformal guarantee',
         conformal_score='max(lower-y,y-upper)',rank='ceil((n+1)*(1-2*lower_probability))',
         projected_empty_intervals=projected,total_intervals=total_pairs,crossing_adjacent_fraction=total_cross/(total_locations*497),
         observations_per_channel=count,starts=list(range(20,821,20)),windows=rows,
         cdf_rearrangement='sort 498 corrected quantiles per position; piecewise-linear interpolation; endpoint atoms',
         comparability='XP uses fixed split-CQR, existing XPF/noise use online calibration; test set previously explored',
         source_sha256={str(p.relative_to(ROOT)):digest(p) for p in [Path(__file__).resolve(),ROOT/'xp02_cqr/xp_head.py',ROOT/'xp02_cqr/train_xp.py']}))
    print('XP_CQR_EVALUATION_COMPLETE',flush=True)

if __name__=='__main__':main()
