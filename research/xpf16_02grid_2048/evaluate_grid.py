"""Evaluate independent XPF16 0.2%-grid heads and actual 2048-path noise.

There are 249 symmetric heads plus one independently trained median head.
Each interval uses its own causal per-tail calibration state. The noise
ensemble also retains a distinct 90%-proxy-scaled full sample distribution.
The full 499-level quantile table and 249-level interval metrics are saved.
"""
import json,sys,time,os
from pathlib import Path
import numpy as np
import torch
from clean_reliability_common import ROOT,configure,dump,manifest
from experiment_reliability_transport_e12 import geometry,history,target,issued,normalization
from experiment_selected16_quantile import new_head
from export_aligned_test_distribution import from_hdf
from train_grid import OUT,configs,EXTREME
from calibration_grid import GridCalibrator,MedianCalibrator,per_level_quantiles
from distribution_grid import quantiles_and_pair,sample_crps,rearrange_quantiles,quantile_distribution_scores,quantile_pit

sys.path.insert(0,str(ROOT/'xpf16_fullpercent_study'))
from evaluate49 import prepare,make_head_batches,all_bounds
from scores import interval_metrics,pinball
sys.path.insert(0,str(ROOT/'xpf16_extreme_002_2048'))
from evaluate_extreme import sample_paths
sys.path.insert(0,str(ROOT/'github/DiPred'))
from reliability.selected16_predict import RequestedThreePredictor
from reliability.calibration import MemoryQuantileCalibrator
from reliability.features import risk_features
sys.path.insert(0, str(ROOT/'xpf16_cdf_fulltest'))
sys.path.insert(0, str(Path(__file__).parent/'cdf'))
from cdf_stream import CDFAccumulator

ALPHA=torch.arange(1,250,dtype=torch.float64)/500
LEVELS=torch.arange(1,500,dtype=torch.float64)/500
METRICS=('lower_coverage','upper_coverage','coverage','width','interval_score','lower_pinball','upper_pinball','lower_miss','upper_miss')
INTERVAL_ARMS=('xpf_raw','xpf_online','noise_raw','noise_proxy','noise_online_two_tail')
QUANTILE_ARMS=('xpf_raw','xpf_online','xpf_online_rearranged','noise_raw','noise_proxy','noise_online_two_tail','noise_online_two_tail_rearranged')


def fields(split,start,records,future=False):
    if start<=760:
        value=target(split,start) if future else history(split,start)
        return value[:len(records)]
    return from_hdf(records,start+(20 if future else 0))


def endpoints(pairs):
    return torch.stack([p[0] for p in pairs[:249]]),torch.stack([p[1] for p in pairs[:249]]),.5*(pairs[249][0]+pairs[249][1])


def assemble(lo,mid,hi):
    return torch.cat((lo,mid.unsqueeze(0),hi.flip(0)),dim=0)


def interval_summary(lo,hi,residual,alpha,chunk=16):
    result=[]
    for i in range(0,len(alpha),chunk):
        j=min(i+chunk,len(alpha))
        levels=alpha[i:j].to(lo.device,dtype=lo.dtype).reshape(j-i,1,1,1,1,1)
        m=interval_metrics(lo[i:j],hi[i:j],residual.unsqueeze(0),levels)
        result.append(torch.stack([m[key].mean((-1,-2)) for key in METRICS],dim=-1).cpu().numpy())
    return np.concatenate(result,axis=0)


def marginal_summary(q,residual,chunk=16):
    result=[]
    for i in range(0,len(LEVELS),chunk):
        j=min(i+chunk,len(LEVELS))
        tau=LEVELS[i:j].to(q.device,dtype=q.dtype).reshape(j-i,1,1,1,1,1)
        e=residual.unsqueeze(0)-q[i:j]
        result.append(torch.stack(((e<=0).float().mean((-1,-2)),pinball(e,tau).mean((-1,-2))),dim=-1).cpu().numpy())
    return np.concatenate(result,axis=0)


def score_summary(q,residual):
    crps,wis=quantile_distribution_scores(q,residual)
    return {'crps':crps.mean((-1,-2)).cpu().numpy(),'wis':wis.mean((-1,-2)).cpu().numpy()}


def offset_from_blocks(blocks,probs):
    # blocks: each [heads,batch,channel,20,8,16], train/test disjoint.
    values=torch.cat(blocks,dim=1).permute(0,2,1,3,4,5).flatten(2).cuda()
    return per_level_quantiles(values,probs.to(values.device)).cpu()


def median_offset(blocks):
    values=torch.cat(blocks,dim=0).permute(1,0,2,3,4).flatten(1).cuda()
    return torch.quantile(values,.5,dim=1).cpu()


@torch.no_grad()
def calibrate(batches,norm,smoke=False):
    out=OUT/'calibration.pt'
    if not smoke and out.exists():return torch.load(out,map_location='cuda',weights_only=False)
    if smoke:
        zero=torch.zeros(249,2,2,device='cuda');med=torch.zeros(2,device='cuda')
        proxy=torch.load(EXTREME/'calibration.pt',map_location='cuda',weights_only=False)['noise_proxy_initial']
        return dict(xpf=zero,noise=zero.clone(),xpf_median=med,noise_median=med.clone(),proxy=proxy)
    records=manifest()['files']['calibration']
    p=RequestedThreePredictor(geometry(records),'selected16','cuda')
    xlo=[];xhi=[];nlo=[];nhi=[];xmed=[];nmed=[]
    for start in range(0,161,20):
        observed=history('calibration',start)
        h,point,base,last=prepare(p,observed)
        lo,hi,mid=endpoints(all_bounds(batches,base,last))
        paths,_=sample_paths(p.model,h)
        noise,_=quantiles_and_pair(paths,point,LEVELS)
        a,b,median=noise[:249],noise[250:].flip(0),noise[249]
        # Future is loaded only after all predictions at every level exist.
        residual=(target('calibration',start)-norm['mean'])/norm['std']-point
        for dest,value in ((xlo,lo),(xhi,hi),(nlo,a),(nhi,b)):
            dest.append((residual.unsqueeze(0)-value)[...,::4,::4].cpu())
        for dest,value in ((xmed,mid),(nmed,median)):
            dest.append((residual-value)[...,::4,::4].cpu())
        del paths,noise,lo,hi,mid,a,b,median
        print('GRID_CALIBRATION',start,flush=True)
    result=dict(xpf=torch.stack((offset_from_blocks(xlo,ALPHA),offset_from_blocks(xhi,1-ALPHA)),1),
                noise=torch.stack((offset_from_blocks(nlo,ALPHA),offset_from_blocks(nhi,1-ALPHA)),1),
                xpf_median=median_offset(xmed),noise_median=median_offset(nmed),
                proxy=torch.load(EXTREME/'calibration.pt',map_location='cpu',weights_only=False)['noise_proxy_initial'])
    temporary=out.with_suffix('.tmp')
    torch.save(result,temporary)
    os.replace(temporary,out)
    return {k:({n:t.cuda() for n,t in v.items()} if isinstance(v,dict) else v.cuda()) for k,v in result.items()}


@torch.no_grad()
def main():
    configure();smoke='--smoke' in sys.argv or '--smoke-full' in sys.argv
    full_smoke='--smoke-full' in sys.argv
    assert smoke or (OUT/'training_complete.json').exists()
    cfgs=configs();heads=[];norm=normalization()
    for cfg in cfgs:
        path=OUT/'models'/cfg['id']/'model.pt'
        if smoke and not path.exists():path=OUT/'models/XPF16-P005/model.pt'
        model=new_head('compact16',None)
        model.load_state_dict(torch.load(path,map_location='cuda',weights_only=False)['model'])
        heads.append(model.eval().requires_grad_(False))
    batches=make_head_batches(heads)
    initial=calibrate(batches,norm,smoke)
    alpha=ALPHA.cuda()
    for split in (('development',) if smoke else ('development','test')):
        folder=OUT/('smoke_full_v3' if full_smoke else 'smoke_v3' if smoke else 'evaluation')/split
        if (folder/'complete.json').exists():
            print('SKIP_GRID_SPLIT',split,flush=True);continue
        folder.mkdir(parents=True,exist_ok=True)
        records=manifest()['files'][split][:2] if smoke and not full_smoke else manifest()['files'][split]
        p=RequestedThreePredictor(geometry(records),'selected16','cuda')
        xctrl=GridCalibrator(alpha,initial['xpf'],len(records))
        nctrl=GridCalibrator(alpha,initial['noise'],len(records))
        xm=MedianCalibrator(initial['xpf_median'],len(records));nm=MedianCalibrator(initial['noise_median'],len(records))
        proxy=MemoryQuantileCalibrator('memory_dynamic',initial['proxy'],len(records))
        # The full CDF is accumulated only on test; all sizes maintain their own
        # calibrated state, including the cold-start window used for feedback.
        cdf = CDFAccumulator('cuda') if split == 'test' and not smoke else None
        if cdf is not None:
            previous = ROOT/'runs/xpf16_noise8_noise32_20261111/calibration.pt'
            noise_initial = torch.load(previous, map_location='cuda', weights_only=False)
            noise_initial['256-noise'] = torch.load(ROOT/'runs/xpf16_fullpercent_20261111/noise256_calibration.pt', map_location='cuda', weights_only=False)
            extra_proxy = {n: MemoryQuantileCalibrator('memory_dynamic', noise_initial[f'{n}-noise'], len(records)) for n in (8, 32, 256)}
        times=[]
        with (folder/'events.jsonl').open('w') as events:
            for start in ([0] if smoke else range(0,821,20)):
                begin=time.perf_counter()
                obs=fields(split,start,records)
                h,point,base,last=prepare(p,obs)
                lo,hi,mid=endpoints(all_bounds(batches,base,last))
                xlo,xhi=xctrl.issue(lo,hi);xmid=xm.issue(mid)
                paths,timing=sample_paths(p.model,h)
                if start<=760:
                    old=issued('diffusion',split,start)
                    physical=paths[:32]*norm['std']+norm['mean']
                    assert torch.allclose(physical.mean(0),old['mean'][:len(records)],rtol=1e-5,atol=2e-6)
                    assert torch.allclose(physical.std(0),old['std32'][:len(records)],rtol=1e-5,atol=2e-6)
                noise,pair=quantiles_and_pair(paths,point,LEVELS)
                nlo,nhi,nmid=noise[:249],noise[250:].flip(0),noise[249]
                nclo,nchi=nctrl.issue(nlo,nhi);ncmid=nm.issue(nmid)
                radius=1.6448536269514722*paths.std(0).clamp_min(1e-6)
                plo,phi=proxy.issue(-radius,radius,risk_features(h,point,radius))
                scale=(phi-plo)/(2*radius).clamp_min(1e-6)
                if cdf is not None:
                    scales = {2048: scale}
                    for count, controller in extra_proxy.items():
                        rad = 1.6448536269514722 * paths[:count].std(0).clamp_min(1e-6)
                        low, high = controller.issue(-rad, rad, risk_features(h, point, rad))
                        scales[count] = (high-low)/(2*rad).clamp_min(1e-6)
                events.write(json.dumps(dict(event='issued_all_499',start=start,time=time.time()))+'\n');events.flush()
                truth=(fields(split,start,records,True)-norm['mean'])/norm['std']
                residual=truth-point
                bounds=((lo,hi),(xlo,xhi),(nlo,nhi),(nlo*scale,nhi*scale),(nclo,nchi))
                intervals=np.stack([interval_summary(a,b,residual,ALPHA) for a,b in bounds])
                qxraw=assemble(lo,mid,hi);qx=assemble(xlo,xmid,xhi)
                if cdf is not None:
                    cdf.add(start, qx, paths, point, residual, scales, norm['std'])
                qxsort,xcross,xany=rearrange_quantiles(qx)
                nqsort,ncross,nany=rearrange_quantiles(assemble(nclo,ncmid,nchi))
                qarms=(qxraw,qx,qxsort,noise,noise*scale,assemble(nclo,ncmid,nchi),nqsort)
                marginals=np.stack([marginal_summary(q,residual) for q in qarms])
                packed={}
                for key,q in (('xpf_raw',qxraw),('xpf_online',qx),('xpf_online_rearranged',qxsort),
                              ('noise_raw_quantiles',noise),('noise_proxy_quantiles',qarms[4]),('noise_two_tail_rearranged',nqsort)):
                    for name,val in score_summary(q,residual).items():packed[key+'_'+name]=val
                for key,q in (('xpf_online',qxsort),('noise_raw',noise),('noise_proxy',qarms[4]),('noise_two_tail',nqsort)):
                    pit=quantile_pit(q,residual)
                    packed[key+'_pit_histogram']=torch.histc(pit.float(),bins=20,min=0,max=1).cpu().numpy()
                for key,value in sample_crps(paths,point,residual,scale,pair).items():packed['noise_'+key]=value.mean((-1,-2)).cpu().numpy()
                for key,value in (('xpf_crossing_fraction',xcross),('xpf_any_crossing',xany),
                                  ('noise_crossing_fraction',ncross),('noise_any_crossing',nany)):
                    packed[key]=value.mean((-1,-2)).cpu().numpy()
                if not smoke and split=='test' and start==0:
                    pixels=torch.linspace(0,40959,16,device='cuda').long()
                    packed['pixels']=pixels.cpu().numpy()
                    packed['truth_snapshot']=residual[[0,-1]].flatten(2)[:,:,pixels].cpu().numpy()
                    packed['point_snapshot']=point[[0,-1]].flatten(2)[:,:,pixels].cpu().numpy()
                    packed['sample_snapshot']=paths[:,[0,-1]].flatten(3)[:,:,:,pixels].cpu().numpy()
                    packed['noise_snapshot']=noise[:,[0,-1]].flatten(3)[:,:,:,pixels].cpu().numpy()
                    packed['xpf_snapshot']=qx[:,[0,-1]].flatten(3)[:,:,:,pixels].cpu().numpy()
                    packed['xpf_rearranged_snapshot']=qxsort[:,[0,-1]].flatten(3)[:,:,:,pixels].cpu().numpy()
                xctrl.reveal(residual);nctrl.reveal(residual);xm.reveal(residual);nm.reveal(residual);proxy.reveal(residual)
                if cdf is not None:
                    for controller in extra_proxy.values():
                        controller.reveal(residual)
                events.write(json.dumps(dict(event='revealed_full20',start=start,time=time.time()))+'\n');events.flush()
                np.savez_compressed(folder/f'{start:04d}.npz',intervals=intervals,marginals=marginals,**packed)
                times.append(dict(start=start,sampling=timing,total_seconds=time.perf_counter()-begin))
                print('GRID_EVALUATION',split,start,flush=True)
                del paths,noise,pair,intervals,marginals,qxraw,qx,qxsort,nqsort,qarms,bounds,packed
        if cdf is not None:
            cdf.save(OUT/'full_test_cdf', records, norm)
        dump(folder/'complete.json',dict(passed=True,records=records,starts=[0] if smoke else list(range(0,821,20)),
             interval_arms=INTERVAL_ARMS,quantile_arms=QUANTILE_ARMS,metrics=METRICS,samples=2048,
             xpf_projection=xctrl.projected.cpu().tolist(),noise_projection=nctrl.projected.cpu().tolist(),
             projection_total=xctrl.total,timings=times,median='separately trained 50% XPF head',
             caveat='Test trajectories historically reused; sample proxy and two-tail calibrated endpoints are distinct.'))
    if not smoke:dump(OUT/'evaluation_complete.json',dict(passed=True,heads=250,quantiles=499,samples=2048))
    else:print('GRID_SMOKE_ONLY: missing trained heads substituted, no formal result',flush=True)

if __name__=='__main__':main()
