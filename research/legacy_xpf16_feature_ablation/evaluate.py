"""Causal 49-head full-test replay using exact legacy CRPS/calibration."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
import torch
from functorch import combine_state_for_ensemble,vmap
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT/'xpf16_fullpercent_study'))
sys.path.insert(0,str(ROOT/'github/DiPred'))
from clean_reliability_common import configure,manifest
from experiment_reliability_transport_e12 import geometry,normalization,history,target
from experiment_fullblock_memory_risk import OUT as MEMORY,load
from experiment_selected16_quantile import new_head
from reliability.selected16_predict import RequestedThreePredictor
from evaluate49 import prepare,get_observed
from train49 import configs
from calibration import TwoTailCalibrator
from scores import interval_metrics,distribution
from features import ARMS,AblatedHead,mask_base
from selected16_quantile_head import BASE_INDICES
OUT=ROOT/'runs/legacy_xpf16_feature_ablation_20261116'
OLD=ROOT/'runs/xpf16_fullpercent_20261111'


def models_for(arm):
    heads=[]
    for cfg in configs():
        path=(OLD/'models'/cfg['id']/'model.pt') if arm=='XPF' else (OUT/'models'/arm/f"Q{cfg['k']:02d}.pt")
        ck=torch.load(path,map_location='cpu',weights_only=False)
        assert ck['parameters']==34004 and ck['config']['k']==cfg['k']
        h=new_head('compact16',None);h.load_state_dict(ck['model'],strict=True)
        heads.append(AblatedHead(h.eval().requires_grad_(False),arm))
    groups=[]
    for pos in range(0,49,7):
        fun,params,buffers=combine_state_for_ensemble(heads[pos:pos+7]);groups.append((fun,params,buffers))
    return heads,groups


def all_bounds(groups,base,last):
    result=[]
    for fun,params,buffers in groups:
        def call(p,b):return fun(p,b,base,last)
        lo,hi=vmap(call)(params,buffers);result.extend((lo[i],hi[i]) for i in range(len(lo)))
    return result


def feature_stats(head,arm,base,last):
    b=mask_base(base,arm);batch,_,_,h,w=b.shape
    f=head.head.feature_projection(last).reshape(batch,8,20,h,w)
    if 'F' not in arm:f=torch.zeros_like(f)
    yy=torch.linspace(-1,1,h,device=b.device,dtype=b.dtype);xx=torch.linspace(-1,1,w,device=b.device,dtype=b.dtype)
    y,x=torch.meshgrid(yy,xx,indexing='ij');coords=torch.stack((x,y))[None,:,None].expand(batch,2,20,h,w)
    value=torch.cat((b[:,list(BASE_INDICES)],f,coords),1).double()
    return value.sum((0,2,3,4)).cpu(),value.square().sum((0,2,3,4)).cpu(),value.shape[0]*value.shape[2]*value.shape[3]*value.shape[4]

@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('arm',choices=ARMS);args=parser.parse_args();arm=args.arm
    configure();torch.cuda.reset_peak_memory_stats();folder=OUT/'evaluation'/arm;folder.mkdir(parents=True,exist_ok=True)
    if (folder/'run.json').exists():print('SKIP_COMPLETE',arm,flush=True);return
    if arm!='XPF':
        assert all((OUT/'models'/arm/f'Q{k:02d}.pt').exists() for k in range(1,50)), 'All 49 heads for this arm must finish first'
    heads,groups=models_for(arm);norm=normalization();cfgs=configs();initial_path=OUT/'calibration'/f'{arm}.pt'
    if arm=='XPF':
        initial=torch.load(OLD/'initial_offsets.pt',map_location='cuda',weights_only=False)
    elif initial_path.exists():initial=torch.load(initial_path,map_location='cuda',weights_only=False)
    else:
        records=manifest()['files']['calibration'];predictor=RequestedThreePredictor(geometry(records),'selected16','cuda');samples=[[] for _ in cfgs]
        for start in range(0,161,20):
            observed=history('calibration',start);_,point,base,last=prepare(predictor,observed)
            raw=all_bounds(groups,base,last)
            truth=(target('calibration',start)-norm['mean'])/norm['std']-point
            for i,(lo,hi) in enumerate(raw):samples[i].append(torch.stack(((truth-lo)[...,::4,::4],(truth-hi)[...,::4,::4])).cpu())
            print('CALIBRATION',arm,start,flush=True)
        offsets=[]
        for cfg,blocks in zip(cfgs,samples):
            v=torch.cat(blocks,dim=1).flatten(3).permute(0,2,1,3).reshape(2,2,-1).cuda()
            offsets.append(torch.stack((torch.quantile(v[0],cfg['lower'],dim=1),torch.quantile(v[1],cfg['upper'],dim=1))))
        initial=torch.stack(offsets);initial_path.parent.mkdir(exist_ok=True,parents=True);torch.save(initial.cpu(),initial_path)
    records=manifest()['files']['test'];predictor=RequestedThreePredictor(geometry(records),'selected16','cuda')
    controls=[TwoTailCalibrator(c['lower'],init,len(records)) for c,init in zip(cfgs,initial)]
    events=folder/'events.jsonl';events.write_text('');times=[];total=None;squares=None;count=0
    for start in range(0,821,20):
        begin=time.perf_counter();observed=get_observed('test',start,records)
        torch.cuda.synchronize();predict_begin=time.perf_counter()
        _,point,base,last=prepare(predictor,observed)
        torch.cuda.synchronize();head_begin=time.perf_counter();raw=all_bounds(groups,base,last);torch.cuda.synchronize();head_seconds=time.perf_counter()-head_begin
        online=[control.issue(*bounds) for control,bounds in zip(controls,raw)]
        torch.cuda.synchronize();predict_seconds=time.perf_counter()-predict_begin
        fs,fq,fc=feature_stats(heads[0],arm,base,last);total=fs if total is None else total+fs;squares=fq if squares is None else squares+fq;count+=fc
        with events.open('a') as f:f.write(json.dumps(dict(event='issued',start=start,time=time.time()))+'\n')
        truth=(get_observed('test',start,records,True)-norm['mean'])/norm['std'];residual=truth-point
        output={}
        for mode,pairs in (('raw',raw),('online',online)):
            lower=torch.stack([v[0] for v in pairs]);upper=torch.stack([v[1] for v in pairs]);q=torch.cat((lower,upper.flip(0)))
            crossing_any_before=(q[1:]<q[:-1]).any(0).float()
            output[f'{mode}_within_pair_crossing_lead']=(lower>upper).float().mean(0).mean((-1,-2)).cpu().numpy()
            d=distribution(q,residual,True);m=interval_metrics(lower[4],upper[4],residual,.05)
            for key in ('crps','wis','crossing_before','crossing_any'):
                output[f'{mode}_{key}_lead']=d[key].mean((-1,-2)).cpu().numpy()
            output[f'{mode}_crossing_any_before_lead']=crossing_any_before.mean((-1,-2)).cpu().numpy()
            for key in ('coverage','width','interval_score','lower_miss','upper_miss'):
                output[f'{mode}_{key}_lead']=m[key].mean((-1,-2)).cpu().numpy()
            del lower,upper,q,d,m
        for control in controls:control.reveal(residual)
        with events.open('a') as f:f.write(json.dumps(dict(event='revealed_full20',start=start,time=time.time()))+'\n')
        np.savez_compressed(folder/f'{start:04d}.npz',**output)
        times.append(dict(start=start,head49_seconds=head_seconds,predict_seconds=predict_seconds,total_seconds=time.perf_counter()-begin))
        if start%100==0:print('EVALUATE',arm,start,flush=True)
    mean=total/count;std=(squares/count-mean.square()).clamp_min(0).sqrt()
    report=dict(arm=arm,records=records,starts=list(range(0,821,20)),head_parameters=34004,timings=times,
        peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(), timing_batch_size=len(records),
        timing_scope='four-step DiPred, all 49 heads and online issue; excludes I/O, truth, scoring and feature audit',
        online_projection=[dict(k=i+1, projected=c.projected_points, total=c.total_points, fraction=c.projected_points/max(c.total_points,1)) for i,c in enumerate(controls)],
        feature_order=['X_std_u','X_std_v','P_gradient_u','P_gradient_v','P_change_u','P_change_v']+[f'F_projection_{i}' for i in range(8)]+['coord_x','coord_y'],
        feature_stats=dict(mean=mean.tolist(),std=std.tolist(),count=count,projection_source='Q01'),
        causality='features and all bounds issued before future truth; feedback after full 20 frames',
        anchor='P for every arm',calibration='legacy per-tail controller gain .5; calibration IDs independent')
    (folder/'run.json').write_text(json.dumps(report,indent=2)+'\n');print('EVALUATION_COMPLETE',arm,flush=True)
if __name__=='__main__':main()
