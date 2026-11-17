"""Replay all 249 independently fitted pair heads and the separate median."""
from followcommon import *
from evaluate49 import make_head_batches,all_bounds
sys.path.insert(0,str(W/'xpf16_02grid_2048'))
from experiment_selected16_quantile import new_head
from calibration_grid import GridCalibrator,MedianCalibrator
from evaluate_risk import groups
import csv
GRID=W/'runs/xpf16_02grid_2048_20261112'
@torch.no_grad()
def main():
 setup();d=cache('development');predictor=model();threshold=load(PRIOR/'gradient_threshold.pt','cuda');heads=[]
 for k in range(1,251):
  ck=load(GRID/'models'/f'XPF16-P{k:03d}'/'model.pt');assert ck['config']['seed']==42 and ck['parameters']==34004
  m=new_head('compact16',None);m.load_state_dict(ck['model']);heads.append(m.eval().requires_grad_(False))
 batches=make_head_batches(heads);initial=load(GRID/'calibration.pt','cuda');alpha=torch.arange(1,250,device='cuda',dtype=torch.float64)/500
 ctrl=GridCalibrator(alpha,initial['xpf'],12);mctrl=MedianCalibrator(initial['xpf_median'],12);rows=[]
 for w,start in enumerate(range(0,161,20)):
  ids=slice(w*12,(w+1)*12);x=d['x'][ids];p,f=feature(predictor,x,'diffusion');base,last=features(x,p,f);pairs=all_bounds(batches,base,last)
  lo=torch.stack([v[0] for v in pairs[:249]]);hi=torch.stack([v[1] for v in pairs[:249]]);mid=(pairs[249][0]+pairs[249][1])/2
  oa,ob=ctrl.issue(lo,hi);om=mctrl.issue(mid);res=d['y'][ids]-p;masks=groups(p,threshold)
  for mode,lower,upper,median in (('raw',lo,hi,mid),('online',oa,ob,om)):
   q=torch.cat((lower,median[None],upper.flip(0))).sort(dim=0).values;med=q[249];t=torch.arange(1,500,device='cuda',dtype=q.dtype)/500
   for cent in ('original','P_aligned'):
    shift=0 if cent=='original' else med;score=torch.zeros_like(res)
    for j in range(0,499,8):
     error=res[None]-(q[j:j+8]-shift);tt=t[j:j+8,None,None,None,None,None];score+=torch.maximum(tt*error,(tt-1)*error).sum(0)*.004
    score+=((q[0]-shift-res).clamp_min(0)+(res-(q[-1]-shift)).clamp_min(0))*.002
    metrics={'crps':score,'median_mse':((med if cent=='original' else 0)-res).square()}
    if cent=='original':
     m=interval_metrics(lower[24],upper[24],res,.05);metrics.update({k:m[k] for k in ('coverage','width','interval_score')})
    for group,sel in masks.items():
     count=sel.sum((2,3,4));v={k:((val*sel).sum((2,3,4))/count.clamp_min(1)).cpu().numpy() for k,val in metrics.items()};cnt=count.cpu().numpy()
     for i in range(12):
      for c in range(2):
       if cnt[i,c]:rows.append(dict(arm='XPF_grid250',mode=mode,center=cent,region=group,channel=c,**d['meta'][w*12+i],**{k:float(val[i,c]) for k,val in v.items()}))
  ctrl.reveal(res);mctrl.reveal(res);print('GRID250',start,flush=True)
 assert ctrl.updates==9 and mctrl.updates==9
 dest=OUT/'audit/grid250_per_group.csv';dest.parent.mkdir(parents=True,exist_ok=True)
 with dest.open('w',newline='') as f:
  wr=csv.DictWriter(f,fieldnames=sorted(set().union(*(o.keys() for o in rows))));wr.writeheader();wr.writerows(rows)
 dump(OUT/'audit/grid250_config.json',dict(heads=250,pairs=249,median=1,levels=499,seed=42,updates=9,calibration_sha256=digest(GRID/'calibration.pt'),weights={f'P{k:03d}':digest(GRID/'models'/f'XPF16-P{k:03d}'/'model.pt') for k in range(1,251)},target_leakage='full prediction issued before reveal',cold_start_included=True))
if __name__=='__main__':main()
