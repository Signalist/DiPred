"""Compare raw and causally calibrated X-only quantile forecasts on aligned tests."""
import csv,json,os,sys,time
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from model import XOnlyIQN
ROOT=Path(__file__).resolve().parent.parent
for name in ('xpf16_02grid_2048','xpf16_cdf_fulltest','github/DiPred'):
    sys.path.insert(0,str(ROOT/name))
sys.path.insert(0,str(ROOT))
from clean_reliability_common import configure,manifest,digest
from experiment_reliability_transport_e12 import normalization,history,target
from export_aligned_test_distribution import from_hdf
from calibration_grid import GridCalibrator,per_level_quantiles
from distribution_grid import rearrange_quantiles,quantile_distribution_scores
from cdf_stream import quantile_cdf_sum
OUT=ROOT/'runs/xonly_iqn_20261112'
REF=ROOT/'github/DiPred/results/reliability/xp02_cqr'
ALPHA=torch.arange(1,250,dtype=torch.float64)/500
LEVELS=torch.tensor([k/500 for k in range(1,250)]+[1-k/500 for k in range(249,0,-1)],dtype=torch.float64)


def fields(split,start,records,future=False):
    if start<=760:return target(split,start) if future else history(split,start)
    return from_hdf(records,start+(20 if future else 0))


@torch.no_grad()
def predict(model,h,chunk=4):
    """All 498 requested marginal quantiles from a single shared network."""
    features=model.encode(h);b=len(h);outputs=[]
    for start in range(0,498,chunk):
        levels=LEVELS[start:start+chunk].to(device=h.device,dtype=h.dtype)
        k=len(levels)
        inp=h.unsqueeze(0).expand(k,*h.shape).reshape(k*b,*h.shape[1:])
        feat=features.unsqueeze(0).expand(k,*features.shape).reshape(k*b,*features.shape[1:])
        tau=levels[:,None].expand(k,b).reshape(-1)
        outputs.append(model.decode(feat,inp,tau).reshape(k,b,*h.shape[1:]))
    return torch.cat(outputs)


def pooled_offsets(blocks,levels):
    values=torch.cat(blocks,dim=1).permute(0,2,1,3,4,5).flatten(2)
    result=[]
    for i in range(0,249,8):
        result.append(per_level_quantiles(values[i:i+8].cuda(),levels[i:i+8].cuda()).cpu())
    return torch.cat(result)


@torch.no_grad()
def calibrate(model,norm):
    path=OUT/'calibration.pt'
    if path.exists():return torch.load(path,map_location='cuda',weights_only=False)
    left=[];right=[];records=manifest()['files']['calibration']
    for start in range(0,161,20):
        x=(history('calibration',start)-norm['mean'])/norm['std']
        q=predict(model,x)
        y=(target('calibration',start)-norm['mean'])/norm['std']
        left.append((y.unsqueeze(0)-q[:249])[...,::4,::4].cpu())
        right.append((y.unsqueeze(0)-q[249:].flip(0))[...,::4,::4].cpu())
        print('XONLY_IQN_CALIBRATION',start,flush=True)
    init=torch.stack((pooled_offsets(left,ALPHA),pooled_offsets(right,1-ALPHA)),1)
    tmp=path.with_suffix('.tmp');torch.save(dict(online=init,calibration_trajectories=len(records),starts=list(range(0,161,20))),tmp);os.replace(tmp,path)
    return torch.load(path,map_location='cuda',weights_only=False)


def draw(output,channel,grid,curves):
    labels={'xonly_iqn_raw':'X-only IQN (raw)','xonly_iqn_online':'X-only IQN + online',
            'xp02_cqr':'XP + CQR','xpf16_1':'XPF16-1 + online','xpf16_02':'XPF16-0.2 + online',
            'noise8':'8-noise + online','noise32':'32-noise + online',
            'noise256':'256-noise + online','noise2048':'2048-noise + online'}
    fig,ax=plt.subplots(figsize=(7.4,5))
    for key,label in labels.items():ax.plot(grid,curves[key],label=label,lw=2 if key.startswith('xonly') else 1.4)
    ax.axvline(0,color='gray',ls=':',lw=.7);ax.grid(alpha=.2);ax.legend(fontsize=8)
    ax.set(title=f'{channel}: X-only quantile forecast',xlabel='Forecast − truth (dataset original scale)',ylabel='Full-test mean CDF')
    fig.tight_layout();fig.savefig(output/f'cdf_{channel}.png',dpi=220);fig.savefig(output/f'cdf_{channel}.pdf');plt.close(fig)


@torch.no_grad()
def main():
    configure();norm=normalization();packed=torch.load(OUT/'model.pt',map_location='cuda',weights_only=False)
    model=XOnlyIQN().cuda().eval();model.load_state_dict(packed['model']);model.requires_grad_(False)
    cal=calibrate(model,norm);records=manifest()['files']['test']
    controller=GridCalibrator(ALPHA.cuda(),cal['online'],len(records))
    grid=torch.linspace(-.35,.35,701,dtype=torch.float64,device='cuda')
    sums=torch.zeros((2,2,701),device='cuda',dtype=torch.float64)
    starts=[];events=[];crps=[];cross=[];coverage=[];count=0
    for start in range(0,821,20):
        h=(fields('test',start,records)-norm['mean'])/norm['std']
        q=predict(model,h);lo=q[:249];hi=q[249:].flip(0)
        raw_bad=lo>hi;mid=.5*(lo+hi)
        rawlo=torch.where(raw_bad,mid,lo);rawhi=torch.where(raw_bad,mid,hi)
        online_lo,online_hi=controller.issue(lo,hi)
        events.append(dict(event='issued_both',start=start,time=time.time()))
        truth=(fields('test',start,records,True)-norm['mean'])/norm['std']
        if start:
            crows=[];xrows=[];vrows=[]
            for method,(a,b) in enumerate(((rawlo,rawhi),(online_lo,online_hi))):
                ordered,before,_=rearrange_quantiles(torch.cat((a,b.flip(0))))
                xrows.append(float(before.mean()))
                for c in range(2):
                    diff=((ordered[:,:,c].double()-truth[:,c].double().unsqueeze(0))*norm['std'][0,c,0,0,0].double()).flatten(1)
                    sums[method,c]+=quantile_cdf_sum(diff,LEVELS,grid)
                median=.5*(ordered[248]+ordered[249]);q499=torch.cat((ordered[:249],median[None],ordered[249:]))
                score,_=quantile_distribution_scores(q499,truth)
                crows.append(score.mean((2,3,4)).cpu().numpy())
                vrows.append(((truth>=a[24])&(truth<=b[24])).float().mean((2,3,4)).cpu().numpy())
                del ordered,q499,score
            crps.append(np.stack(crows));cross.append(xrows);coverage.append(np.stack(vrows))
            starts.append(start);count+=truth[:,0].numel()
        controller.reveal(truth)
        events.append(dict(event='revealed_complete_20',start=start,time=time.time()))
        print('XONLY_IQN_TEST',start,flush=True)
    assert starts==list(range(20,821,20)) and count==len(records)*41*20*32*64
    cdf=(sums/count).cpu().numpy();x=grid.cpu().numpy();crps=np.stack(crps);coverage=np.stack(coverage);cross=np.array(cross)
    if not np.isfinite(cdf).all() or np.diff(cdf,axis=-1).min()<-1e-7:raise RuntimeError('Invalid CDF')
    np.savez_compressed(OUT/'metrics.npz',cdf=cdf,x=x,crps=crps,coverage90=coverage,crossing=cross,starts=starts)
    for channel,c in (('u',0),('v',1)):
        ref=np.load(REF/f'cdf_{channel}.npz')
        assert np.array_equal(ref['x'],x) and ref['count']==count
        curves={key:ref[key] for key in ('xp02_cqr','xpf16_1','xpf16_02','noise8','noise32','noise256','noise2048')}
        curves.update(xonly_iqn_raw=cdf[0,c],xonly_iqn_online=cdf[1,c])
        np.savez_compressed(OUT/f'cdf_{channel}.npz',x=x,count=count,**curves)
        with (OUT/f'cdf_{channel}.csv').open('w',newline='') as f:
            w=csv.writer(f);w.writerow(['x',*curves]);w.writerows(zip(x,*curves.values()))
        draw(OUT,channel,x,curves)
    (OUT/'events.json').write_text(json.dumps(events,indent=2)+'\n')
    summary=dict(passed=True,method='X-only IQN-inspired PDE model',inputs='20 observed frames; no P, F or diffusion',
         observations_per_channel=count,starts=starts,records=records,calibration_trajectories=cal['calibration_trajectories'],
         mean_crps=crps.mean((0,2)).tolist(),mean_coverage90=coverage.mean((0,2)).tolist(),
         crossing_fraction=cross.mean(0).tolist(),projection_counts=controller.projected.cpu().tolist(),projection_total=controller.total,
         protocol='same 41 post-cold-start test windows; online per-tail per-trajectory updates only after full 20 frames',
         cdf='498 quantiles after pointwise monotone rearrangement; linear interpolation; endpoint atoms',
         crps='normalized by train channel std, 0.2%-grid integration with interpolated median and endpoint atoms',
         caveat='finite calibration data, dependent pixels and reused test trajectories; no formal distribution-free guarantee',
         source_sha256={str(p.relative_to(ROOT)):digest(p) for p in (Path(__file__).resolve(),ROOT/'xonly_iqn/model.py',ROOT/'xonly_iqn/train.py',OUT/'model.pt')})
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('XONLY_IQN_EVALUATION_COMPLETE',flush=True)

if __name__=='__main__':main()
