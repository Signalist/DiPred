"""Exact legacy 49-head XPF, F and XP development replay, causal online."""
from followcommon import *
from evaluate_risk import groups
import csv
@torch.no_grad()
def main():
 setup();d=cache('development');predictor=model();threshold=load(PRIOR/'gradient_threshold.pt','cuda');stats=norm();cfg=configs();outputs=[];events=[]
 for arm in ('XPF','XP','F'):
  heads,batches=models_for(arm)
  initial=load(OLD/'initial_offsets.pt','cuda') if arm=='XPF' else load(ABL/'calibration'/f'{arm}.pt','cuda')
  controllers=[TwoTailCalibrator(c['lower'],init,12) for c,init in zip(cfg,initial)]
  for w,start in enumerate(range(0,161,20)):
   ids=slice(w*12,(w+1)*12);x=d['x'][ids];p,f=feature(predictor,x,'diffusion');assert torch.allclose(p,d['p'][ids],atol=2e-5,rtol=1e-5)
   base,last=features(x,p,f);raw=all_bounds(batches,base,last);online=[c.issue(*pair) for c,pair in zip(controllers,raw)]
   events.append(dict(arm=arm,start=start,event='issue_all_49'));truth=d['y'][ids];residual=truth-p
   masks=groups(p,threshold)
   for mode,pairs in (('raw',raw),('online',online)):
    low=torch.stack([a for a,b in pairs]);high=torch.stack([b for a,b in pairs]);q=torch.cat((low,high.flip(0))).sort(dim=0).values
    median=(q[48]+q[49])/2
    for center in ('original','P_aligned'):
     values=q if center=='original' else q-median[None]
     # Streaming CRPS avoids auxiliary tensors of 101 complete fields.
     q99=torch.cat((values[:49],(values[48:49]+values[49:50])/2,values[49:]));tau=torch.arange(1,100,device='cuda',dtype=q.dtype)/100
     score=torch.zeros_like(residual)
     for j in range(0,99,8):
      e=residual[None]-q99[j:j+8];tt=tau[j:j+8,None,None,None,None,None];score+=torch.maximum(tt*e,(tt-1)*e).sum(0)*.02
     score+=(q99[0]-residual).clamp_min(0)*.01+(residual-q99[-1]).clamp_min(0)*.01
     metrics={'crps':score,'median_mse':(median if center=='original' else torch.zeros_like(median)-residual).square() if center=='P_aligned' else (median-residual).square()}
     # The 90% interval uses original Q05, not the across-head sorted mixture.
     if center=='original':
      lo,hi=pairs[4];m=interval_metrics(lo,hi,residual,.05);metrics.update({k:m[k] for k in ('coverage','width','interval_score')})
     for group,sel in masks.items():
      count=sel.sum((2,3,4));vals={k:((v*sel).sum((2,3,4))/count.clamp_min(1)).cpu().numpy() for k,v in metrics.items()};counts=count.cpu().numpy()
      for j in range(12):
       for c in range(2):
        if counts[j,c]:outputs.append(dict(arm=arm,mode=mode,center=center,region=group,channel=c,**d['meta'][w*12+j],**{k:float(v[j,c]) for k,v in vals.items()}))
   for c in controllers:c.reveal(residual)
   events.append(dict(arm=arm,start=start,event='reveal_full20'));print('LEGACY_AUDIT',arm,start,flush=True)
  assert all(c.revealed_windows==9 for c in controllers)
 folder=OUT/'audit';folder.mkdir(parents=True,exist_ok=True)
 with (folder/'per_group.csv').open('w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=sorted(set().union(*(o.keys() for o in outputs))));writer.writeheader();writer.writerows(outputs)
 dump(folder/'events.json',events);dump(folder/'config.json',dict(arms=['XPF','XP','F'],split='development',starts=list(range(0,161,20)),cold_start_included=True,levels='98 independent raw quantiles sorted; median interpolated; constant 1% tails',coverage='Q05 as issued',seed=42))
if __name__=='__main__':main()
