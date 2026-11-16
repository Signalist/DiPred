"""Full-trajectory causal X/XP/XPF evaluation using common 0.2% grid."""
import argparse,hashlib,json,math,sys,time
import numpy as np
import torch
from pathlib import Path
from common import ROOT,OUT,Scenario,configure,predict,sample_predict,records,save_atomic
for name in ('fair_x_xp_xpf','xpf16_02grid_2048','xpf16_fullpercent_study','xpf16_cdf_fulltest'):
    sys.path.insert(0,str(ROOT/name))
from fair_x_xp_xpf.model import FairQuantileHead
from fair_x_xp_xpf.evaluate import LEVELS,ALPHA,quantile_crps,sorted_cdf_sum,bounds,pair_quantiles
from calibration_grid import GridCalibrator,MedianCalibrator
from distribution_grid import rearrange_quantiles,quantiles_and_pair

ARMS=('X','XP','XPF');NOISE_ARMS=('8-noise','32-noise','256-noise');MASKS={'X':(0.,0.),'XP':(1.,0.),'XPF':(1.,1.)}
MODES=('raw','fixed_cqr','online')


def load_head(name,arm,seed):
    state=torch.load(OUT/'heads'/name/f'seed_{seed}'/(arm+'.pt'),map_location='cuda',weights_only=False)
    model=FairQuantileHead().cuda().eval().requires_grad_(False);model.load_state_dict(state['model'])
    mean=torch.tensor(state['f_mean'],device='cuda')[None,:,None,None]
    std=torch.tensor(state['f_std'],device='cuda')[None,:,None,None]
    return model,state,mean,std


def issued(scenario,backbone,head,rs,starts,arm,fmean,fstd,return_samples=False):
    x=scenario.batch(rs,starts)
    if arm in NOISE_ARMS:
        samples=sample_predict(backbone,x,scenario.mask,int(arm.split('-')[0]))
        quantiles,_=quantiles_and_pair(samples,torch.zeros_like(x),LEVELS.cuda().float())
        return (quantiles,samples) if return_samples else quantiles
    if arm=='X':
        p=torch.zeros_like(x)
        f=torch.zeros(len(x),48,32,64,device=x.device,dtype=x.dtype)
    elif arm=='XP':
        p=predict(backbone,x,scenario.mask)
        f=torch.zeros(len(x),48,32,64,device=x.device,dtype=x.dtype)
    else:
        p,f=predict(backbone,x,scenario.mask,with_features=True)
        f=((f-fmean)/fstd).clamp(-10,10)
    mask=torch.tensor(MASKS[arm],device='cuda').expand(len(x),2)
    return head.quantiles(x,p,f,mask,LEVELS.cuda().float())


@torch.no_grad()
def calibrate(name,scenario,backbone,head,arm,seed,fmean,fstd):
    path=OUT/'calibration'/name/f'{arm}_{seed}.pt'
    if path.exists():return torch.load(path,map_location='cuda',weights_only=False)
    rs=[r for r in records(name) if r['split']=='calibration']
    end=(min(r['frames'] for r in rs)-40)//20
    starts=sorted({int(round(v))*20 for v in np.linspace(0,end,9)})
    left=[];right=[];scores=[];medians=[]
    for start in starts:
        q=issued(scenario,backbone,head,rs,[start]*len(rs),arm,fmean,fstd)
        y=scenario.batch(rs,[start]*len(rs),True)
        lo,hi=q[:249],q[250:].flip(0)
        left.append((y[None]-lo)[...,::4,::4].cpu());right.append((y[None]-hi)[...,::4,::4].cpu())
        scores.append(torch.maximum(lo-y[None],y[None]-hi)[...,::4,::4].cpu())
        medians.append((y-q[249])[...,::4,::4].cpu())
    initial=torch.stack((pair_quantiles(left,ALPHA),pair_quantiles(right,1-ALPHA)),1)
    value=torch.cat(scores,dim=1).permute(0,2,1,3,4,5).flatten(2);n=value.shape[-1];fixed=[]
    for i in range(249):
        rank=math.ceil((n+1)*(1-2*float(ALPHA[i])))
        if rank>n:raise ValueError('Insufficient calibration samples')
        fixed.append(value[i].cuda().sort(dim=-1).values[:,rank-1].cpu())
    median=torch.cat(medians,0).permute(1,0,2,3,4).flatten(1).median(dim=1).values
    save_atomic(dict(initial=initial,fixed=torch.stack(fixed),median=median,n=n,starts=starts,
        ids=[r['id'] for r in rs],note='pooled spatial-temporal dependent scores; no trajectory coverage guarantee'),path)
    print('CALIBRATION',name,arm,seed,n,flush=True)
    return torch.load(path,map_location='cuda',weights_only=False)


@torch.no_grad()
def run(name,arm,seed):
    path=OUT/'evaluated'/name/f'{arm}_{seed}.pt'
    if path.exists():print('ALREADY_DONE',name,arm,seed,flush=True);return
    scenario=Scenario(name);backbone=scenario.network().eval().requires_grad_(False)
    backbone.load_state_dict(torch.load(OUT/'backbones'/name/'model.pt',map_location='cuda',weights_only=False)['model'])
    if arm in NOISE_ARMS:
        head=fmean=fstd=None
        state=dict(head_parameters=0,group_wall_seconds=0.)
    else:
        head,state,fmean,fstd=load_head(name,arm,seed)
    cal=calibrate(name,scenario,backbone,head,arm,seed,fmean,fstd)
    rs=[r for r in records(name) if r['split']=='test'];batch=len(rs)
    ends={(r['frames']-40)//20 for r in rs}
    if len(ends)!=1:raise ValueError('Variable test window counts require active-trajectory support')
    starts=list(range(0,(next(iter(ends))+1)*20,20))
    controller=GridCalibrator(ALPHA.cuda(),cal['initial'],batch);median=MedianCalibrator(cal['median'],batch)
    levels=LEVELS.cuda().float();radius=3*max(scenario.info['std'])
    grid=torch.linspace(-radius,radius,701,device='cuda',dtype=torch.float64)
    cdfs=torch.zeros(3,scenario.channels,len(grid),device='cuda',dtype=torch.float64)
    coverage=torch.zeros(3,499,scenario.channels,device='cuda',dtype=torch.float64)
    pinball=torch.zeros_like(coverage);window=[];frame=[];snapshots=[];events=[];observations=0
    progress=path.with_name(path.stem+'.progress.pt')
    completed=-1
    code_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if progress.exists():
        saved=torch.load(progress,map_location='cuda',weights_only=False)
        if saved['ids']!=[r['id'] for r in rs] or saved['arm']!=arm or saved['seed']!=seed or saved['code_sha']!=code_sha:
            raise ValueError('Evaluation checkpoint does not match this run')
        completed=saved['start'];window=saved['windows'];frame=saved['frames']
        snapshots=saved['snapshots'];events=saved['events'];observations=saved['observations']
        cdfs=saved['cdfs'];coverage=saved['coverage'];pinball=saved['pinball']
        controller.offsets=saved['offsets'];controller.projected=saved['projected']
        controller.total=saved['total'];controller.updates=saved['updates']
        median.offsets=saved['median_offsets'];median.updates=saved['median_updates']
        print('EVALUATION_RESUMED',name,arm,seed,completed,flush=True)
    for start in starts:
        if start<=completed:continue
        raw=issued(scenario,backbone,head,rs,[start]*batch,arm,fmean,fstd,return_samples=arm in NOISE_ARMS)
        if arm in NOISE_ARMS:q,samples=raw
        else:q,samples=raw,None
        predictions,projected=bounds(q,cal,controller,median)
        events.append(dict(event='issued',start=start,time=time.time()))
        y=scenario.batch(rs,[start]*batch,True)
        events.append(dict(event='revealed',start=start,time=time.time()))
        if start:
            for mode,original in enumerate(predictions):
                if mode==0:
                    if (original[1:]<original[:-1]).any():raise ValueError('Raw quantiles cross')
                    ordered=original;crossing=torch.zeros_like(y);any_crossing=torch.zeros_like(y)
                else:ordered,crossing,any_crossing=rearrange_quantiles(original)
                score=quantile_crps(ordered,y,levels)*scenario.std
                low,high=ordered[24],ordered[474]
                covered=((y>=low)&(y<=high)).float();width=(high-low)*scenario.std
                interval=width+20*((low-y).clamp_min(0)+(y-high).clamp_min(0))*scenario.std
                values=torch.stack((score.mean((2,3,4)),covered.mean((2,3,4)),width.mean((2,3,4)),
                    interval.mean((2,3,4)),crossing.mean((2,3,4)),any_crossing.mean((2,3,4))),-1).cpu().numpy()
                window.append(dict(mode=mode,start=start,values=values,projected=projected[mode]))
                frame.append(dict(mode=mode,start=start,values=score.mean((-1,-2)).cpu().numpy()))
                for i in range(0,499,8):
                    z=ordered[i:i+8];e=y[None]-z;t=levels[i:i+8,None,None,None,None,None]
                    coverage[mode,i:i+8]+=(y[None,:,:scenario.channels]<=z[:,:,:scenario.channels]).double().mean((1,3,4,5))
                    pinball[mode,i:i+8]+=(torch.maximum(t*e,(t-1)*e)*scenario.std)[:,:,:scenario.channels].double().mean((1,3,4,5))
                for c in range(scenario.channels):
                    if mode==0 and samples is not None:
                        # Raw noise CDF is the actual empirical ensemble CDF.
                        histogram=torch.zeros(len(grid)+1,device='cuda',dtype=torch.float64)
                        for sample in samples:
                            errors=((sample[:,c]-y[:,c])*scenario.std[0,c,0,0,0]).flatten().double()
                            bins=torch.bucketize(errors,grid)
                            histogram+=torch.bincount(bins,minlength=len(grid)+1).double()
                        cdfs[mode,c]+=histogram[:-1].cumsum(0)/len(samples)
                    else:
                        errors=((ordered[:,:,c]-y[:,c][None])*scenario.std[0,c,0,0,0]).flatten(1)
                        cdfs[mode,c]+=sorted_cdf_sum(errors,LEVELS,grid)
                if start==20:
                    snapshots.append(dict(mode=mode,q=ordered[:,0,:scenario.channels,::19,4::12,8::24].cpu(),
                        y=y[0,:scenario.channels,::19,4::12,8::24].cpu(),crps=score[0,:scenario.channels,::19,4::12,8::24].cpu()))
            observations+=y[:,0].numel()
        controller.reveal(y);median.reveal(y)
        if start and (start//20)%5==0:
            # Save only after the full forecast was issued, revealed and scored.
            save_atomic(dict(ids=[r['id'] for r in rs],arm=arm,seed=seed,start=start,code_sha=code_sha,
                windows=window,frames=frame,snapshots=snapshots,events=events,observations=observations,
                cdfs=cdfs,coverage=coverage,pinball=pinball,offsets=controller.offsets,
                projected=controller.projected,total=controller.total,updates=controller.updates,
                median_offsets=median.offsets,median_updates=median.updates),progress)
        print('EVALUATED',name,arm,seed,start,flush=True)
    save_atomic(dict(name=name,arm=arm,seed=seed,windows=window,frames=frame,snapshots=snapshots,
        events=events,grid=grid.cpu(),cdf=(cdfs/observations).cpu(),observations=observations,
        qcoverage=(coverage/(len(starts)-1)).cpu(),qpinball=(pinball/(len(starts)-1)).cpu(),
        records=rs,channels=scenario.info['channels'],parameters=state['head_parameters'],
        train_seconds_group=state['group_wall_seconds']),path)
    progress.unlink(missing_ok=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('scenario');parser.add_argument('--arm',choices=ARMS+NOISE_ARMS);parser.add_argument('--seed',type=int);args=parser.parse_args()
    configure()
    if not (OUT/'validation'/f'{args.scenario}.json').exists():raise RuntimeError('Run fixed development evaluation before testing')
    for arm in ((args.arm,) if args.arm else ARMS):
        seeds=(500000,) if arm in NOISE_ARMS else ((args.seed,) if args.seed else (42,123,2026))
        for seed in seeds:run(args.scenario,arm,seed)
    print('EVALUATION_COMPLETE',args.scenario,flush=True)

if __name__=='__main__':main()
