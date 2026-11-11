"""Evaluate all 49 independent heads and actual 8/32 paths causally.

The 8/32 calibrated paths use the established 90% proxy-scale controller;
all other reported noise quantiles are empirical order statistics of paths.
XPF uses a separate alpha-specific two-tail controller per head. Evaluation
streams are restartable at split level; existing window files are overwritten
only after a fresh causal replay from each split's beginning.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from functorch import combine_state_for_ensemble, vmap

from calibration import TwoTailCalibrator
from batched_noise import sample_batched
from clean_reliability_common import ROOT, configure, manifest, dump
from experiment_reliability_transport_e12 import geometry, normalization, history, target, issued
from experiment_fullblock_memory_risk import OUT as MEMORY, load
from experiment_selected16_quantile import new_head
from export_aligned_test_distribution import from_hdf
from train49 import OUT, configs
from scores import interval_metrics, distribution
from xpf_crps_metrics import ensemble_crps
from causal_shape_calibration import initial_state

sys.path.insert(0, str(ROOT/'github/DiPred'))
from reliability.selected16_predict import RequestedThreePredictor
from reliability.features import forecast_features, base_features, risk_features
from reliability.calibration import MemoryQuantileCalibrator

METRICS = ('lower_coverage','upper_coverage','coverage','width','interval_score',
           'lower_pinball','upper_pinball','lower_miss','upper_miss')


class HeadForward(nn.Module):
    def __init__(self,head):
        super().__init__()
        self.head=head
    def forward(self,base,last):
        return self.head.bounds(base,last,None,None,False)


def make_head_batches(heads):
    batches=[]
    for start in range(0,len(heads),7):
        models=[HeadForward(h) for h in heads[start:start+7]]
        f,params,buffers=combine_state_for_ensemble(models)
        batches.append((f,params,buffers,len(models)))
    return batches


def all_bounds(batches,base,last):
    output=[]
    for f,params,buffers,count in batches:
        def call(p,b):
            return f(p,b,base,last)
        lo,hi=vmap(call)(params,buffers)
        output.extend((lo[i],hi[i]) for i in range(count))
    return output


def prepare(p, observed):
    h=(observed-p.norm['mean'])/p.norm['std']
    point, last=forecast_features(p.model,h)
    physical=point*p.norm['std']+p.norm['mean']
    base=base_features(observed,physical,p.norm['mean'],p.norm['std'],p.geometry.expand(len(point),3))
    base=((base-p.stats['base_mean'])/p.stats['base_std']).clamp(-10,10)
    last=((last[:,-48:].half().float()-p.stats['last_mean'])/p.stats['last_std']).clamp(-10,10)
    return h,point,base,last


def get_observed(split, start, records, future=False):
    if start<=760:
        return target(split,start) if future else history(split,start)
    return from_hdf(records,start+(20 if future else 0))


def empirical(samples, probability):
    ordered=samples.sort(dim=0).values
    index=probability*(len(samples)-1)
    a=int(index)
    b=min(a+1,len(samples)-1)
    return ordered[a]+(index-a)*(ordered[b]-ordered[a])


@torch.no_grad()
def main():
    configure()
    smoke = '--smoke' in sys.argv
    assert smoke or (OUT/'training_complete.json').exists(), 'All 49 heads must finish first'
    cfgs=configs()
    norm=normalization()
    heads=[]
    for cfg in cfgs:
        head=new_head('compact16',load(MEMORY/'statistics.pt')['memory_scale'])
        model_path=OUT/'models'/cfg['id']/'model.pt'
        if smoke and not model_path.exists():
            model_path=OUT/'models/XPF16-Q01/model.pt'
        head.load_state_dict(load(model_path)['model'])
        heads.append(head.eval().requires_grad_(False))
    head_batches=make_head_batches(heads)
    initial_file=OUT/'initial_offsets.pt'
    if smoke:
        initial=torch.zeros((49,2,2),device='cuda')
    elif initial_file.exists():
        initial=torch.load(initial_file,map_location='cuda',weights_only=False)
    else:
        records=manifest()['files']['calibration']
        p=RequestedThreePredictor(geometry(records),'selected16','cuda')
        samples=[[] for _ in cfgs]
        for start in range(0,161,20):
            observed=history('calibration',start)
            h,point,base,last=prepare(p,observed)
            raw=all_bounds(head_batches,base,last)
            truth=(target('calibration',start)-norm['mean'])/norm['std']-point
            for i,(lo,hi) in enumerate(raw):
                samples[i].append(torch.stack(((truth-lo)[...,::4,::4],(truth-hi)[...,::4,::4])).cpu())
            print('CALIBRATION',start,flush=True)
        values=[]
        for cfg,blocks in zip(cfgs,samples):
            value=torch.cat(blocks,dim=1).flatten(3).permute(0,2,1,3).reshape(2,2,-1).cuda()
            values.append(torch.stack((torch.quantile(value[0],cfg['lower'],dim=1),
                                       torch.quantile(value[1],cfg['upper'],dim=1))))
        initial=torch.stack(values)
        torch.save(initial.cpu(),initial_file)
    noise_initial=torch.load(ROOT/'runs/xpf16_noise8_noise32_20261111/calibration.pt',map_location='cuda',weights_only=False)
    noise256_file=OUT/'noise256_calibration.pt'
    if noise256_file.exists():
        noise_initial['256-noise']=torch.load(noise256_file,map_location='cuda',weights_only=False)
    else:
        records=manifest()['files']['calibration']
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
        noise_initial['256-noise']=initial_state(torch.cat(ratios).cuda(),'separable_integral')
        torch.save({k:v.cpu() for k,v in noise_initial['256-noise'].items()},noise256_file)
    for split in (('development',) if smoke else ('development','test')):
        folder=OUT/('smoke_evaluation_v4' if smoke else 'evaluation')/split
        if (folder/'run.json').exists():
            print('SKIP_COMPLETE_SPLIT',split,flush=True)
            continue
        records=manifest()['files'][split]
        p=RequestedThreePredictor(geometry(records),'selected16','cuda')
        controls=[TwoTailCalibrator(c['lower'],init,len(records)) for c,init in zip(cfgs,initial)]
        noise={name:MemoryQuantileCalibrator('memory_dynamic',noise_initial[name],len(records)) for name in ('8-noise','32-noise','256-noise')}
        folder=OUT/('smoke_evaluation_v4' if smoke else 'evaluation')/split
        folder.mkdir(parents=True,exist_ok=True)
        times=[]
        events=folder/'events.jsonl'
        events.write_text('')
        for start in ([0] if smoke else range(0,821,20)):
            tick=time.perf_counter()
            observed=get_observed(split,start,records)
            torch.cuda.synchronize(); point_begin=time.perf_counter()
            h,point,base,last=prepare(p,observed)
            torch.cuda.synchronize(); point_seconds=time.perf_counter()-point_begin
            begin=time.perf_counter()
            single=heads[0].bounds(base,last,None,None,False)
            torch.cuda.synchronize(); head_single=time.perf_counter()-begin
            begin=time.perf_counter()
            raw=all_bounds(head_batches,base,last)
            torch.cuda.synchronize(); head_seconds=time.perf_counter()-begin
            assert all(torch.allclose(a,b,rtol=1e-5,atol=2e-6) for a,b in zip(single,raw[0]))
            online=[control.issue(*pair) for control,pair in zip(controls,raw)]
            sample_begin=time.perf_counter()
            sample_times={}
            pathlist=[]
            for seed in range(500000,500256,2):
                pathlist.append(sample_batched(p.model,h,(seed,seed+1)))
                count=seed-500000+2
                if count in (8,32,256):
                    torch.cuda.synchronize()
                    sample_times[str(count)]=time.perf_counter()-sample_begin
            paths=torch.cat(pathlist)
            assert torch.isfinite(paths).all()
            if start <= 760:
                original=issued('diffusion',split,start)
                physical=paths[:32]*norm['std']+norm['mean']
                assert torch.allclose(physical.mean(0),original['mean'],rtol=1e-5,atol=2e-6)
                assert torch.allclose(physical.std(0),original['std32'],rtol=1e-5,atol=2e-6)
            rawsets={'8-noise':paths[:8], '32-noise':paths[:32], '256-noise':paths}
            onsets={}
            for name,vals in rawsets.items():
                radius=1.6448536269514722*vals.std(0).clamp_min(1e-6)
                lo,hi=noise[name].issue(-radius,radius,risk_features(h,point,radius))
                onsets[name]=point.unsqueeze(0)+(vals-point.unsqueeze(0))*((hi-lo)/(2*radius).clamp_min(1e-6)).unsqueeze(0)
            # All fields above are finalized before the full future is loaded.
            with events.open('a') as stream:
                stream.write(json.dumps(dict(event='issued',start=start,time=time.time()))+'\n')
            truth=(get_observed(split,start,records,True)-norm['mean'])/norm['std']
            residual=truth-point
            batch=len(records)
            result=np.empty((2,4,49,batch,2,20,len(METRICS)),dtype=np.float32)
            qpred=[]
            diststats={}
            for mode,(pairs,sets) in enumerate(((raw,rawsets),(online,onsets))):
                for method in range(4):
                    if method==0:
                        bounds=pairs
                    else:
                        vals=sets[('8-noise','32-noise','256-noise')[method-1]]-point.unsqueeze(0)
                        sorted_vals=vals.sort(dim=0).values
                        def order(tau):
                            index=tau*(len(vals)-1); a=int(index); b=min(a+1,len(vals)-1)
                            return sorted_vals[a]+(index-a)*(sorted_vals[b]-sorted_vals[a])
                        bounds=[(order(c['lower']),order(c['upper'])) for c in cfgs]
                    # Vectorize levels for scoring, while keeping the 49
                    # parameter sets and interval predictions independent.
                    lower=torch.stack([b[0] for b in bounds])
                    upper=torch.stack([b[1] for b in bounds])
                    assert torch.isfinite(lower).all() and torch.isfinite(upper).all() and (lower<=upper).all()
                    levels=torch.arange(1,50,device=lower.device,dtype=lower.dtype).reshape(49,1,1,1,1,1)/100
                    m=interval_metrics(lower,upper,residual.unsqueeze(0),levels)
                    result[mode,method]=torch.stack([m[key].mean((-1,-2)) for key in METRICS],dim=-1).cpu().numpy()
                    if method==0:
                        q98=torch.cat((lower,upper.flip(0)))
                        for rearrange in (False,True):
                            d=distribution(q98,residual,rearrange)
                            key=('raw','online')[mode]+('_rearranged' if rearrange else '_unrearranged')
                            diststats[key]={n:v.mean((2,3,4)).cpu().numpy() for n,v in d.items() if n!='pit'}
                            hist=torch.histc(d['pit'].float(),bins=20,min=0,max=1).cpu().numpy()
                            diststats[key]['pit_histogram']=hist
                        if split=='test' and start==0:
                            pixels=torch.linspace(0,40959,32,device='cuda').long()
                            qpred.append(q98[:,[0,-1]].flatten(3)[:,:,:,pixels].cpu().numpy())
                    else:
                        metrics=ensemble_crps(sets[('8-noise','32-noise','256-noise')[method-1]],truth)
                        qnoise=torch.cat((lower,upper.flip(0)))
                        dg=distribution(qnoise,residual,True)
                        values={k:v.mean((2,3,4)).cpu().numpy() for k,v in metrics.items()}
                        values['wis']=dg['wis'].mean((2,3,4)).cpu().numpy()
                        values['pit_histogram']=torch.histc(dg['pit'].float(),bins=20,min=0,max=1).cpu().numpy()
                        diststats[('raw','online')[mode]+'_'+('8-noise','32-noise','256-noise')[method-1]]=values
            for control in controls:
                control.reveal(residual)
            for control in noise.values():
                control.reveal(residual)
            with events.open('a') as stream:
                stream.write(json.dumps(dict(event='revealed_full20',start=start,time=time.time()))+'\n')
            packed={key+'_'+name:value for key,vals in diststats.items() for name,value in vals.items()}
            np.savez_compressed(folder/f'{start:04d}.npz',metrics=result,**packed)
            if qpred:
                np.savez_compressed(folder/'snapshot.npz',raw=qpred[0],online=qpred[1],truth=residual[[0,-1]].flatten(2)[:,:,pixels].cpu().numpy(),pixels=pixels.cpu().numpy())
            times.append(dict(start=start,point_seconds=point_seconds,head49_seconds=head_seconds,head_single_seconds=head_single,noise_sample_seconds=sample_times,total_seconds=time.perf_counter()-tick))
            if start%100==0:
                print('EVALUATE',split,start,'head_seconds',head_seconds,flush=True)
        dump(folder/'run.json',dict(split=split,records=records,starts=([0] if smoke else list(range(0,821,20))),
             metrics=METRICS,timings=times,head_parameters=34004,
             projection=[dict(experiment=c['id'],points=pj.projected_points,total=pj.total_points) for c,pj in zip(cfgs,controls)],
             monotone_calibration='per-tail empirical offsets, midpoint projection on within-head crossings',
             noise_calibration='per-method 90% proxy based scale; reported other quantiles are empirical from transformed actual paths',
             causality='current complete Y loaded only after all forecasts issued; feedback once per block'))
        print('SPLIT_COMPLETE',split,flush=True)
    if not smoke:
        dump(OUT/'evaluation_complete.json',dict(passed=True,models=49,starts=42,splits=['development','test']))
    else:
        print('SMOKE_ONLY: missing models were substituted; no study results produced',flush=True)

if __name__=='__main__':
    main()
