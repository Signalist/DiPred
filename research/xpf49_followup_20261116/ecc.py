"""Actual XPF49 quantiles plus actual 8/32 diffusion trajectories, ECC ranks."""
from followcommon import *
from query_ensemble import empirical_crps
import csv
@torch.no_grad()
def couple(q,paths,p):
 n=len(paths);rank=paths.argsort(0).argsort(0).float();z=((rank+.5)/n*100-1).clamp(0,98);i=z.floor().long();frac=z-i
 v=torch.gather(q,0,i)*(1-frac)+torch.gather(q,0,(i+1).clamp(max=98))*frac
 return p[None]+v
@torch.no_grad()
def main():
 setup();data=cache('development');pred=model();heads,batches=models_for('XPF');cfg=configs();initial=load(OLD/'initial_offsets.pt','cuda');controls=[TwoTailCalibrator(c['lower'],s,12) for c,s in zip(cfg,initial)]
 qs=queries();thresholds=load(PRIOR/'query/statistics.pt','cuda')['thresholds'];std=norm()['std'].flatten().repeat(2);samples={f'XPF49_ECC{n}_{mode}':[] for n in (8,32) for mode in ('raw','online')};timing=[]
 for w,start in enumerate(range(0,161,20)):
  ids=slice(12*w,12*(w+1));x=data['x'][ids];p,f=feature(pred,x,'diffusion');base,last=features(x,p,f);raw=all_bounds(batches,base,last);on=[c.issue(*v) for c,v in zip(controls,raw)]
  torch.cuda.synchronize();begin=time.perf_counter();paths=torch.stack([pred.rollout(x,500000+k,sign=1) for k in range(32)]);torch.cuda.synchronize();sample_seconds=time.perf_counter()-begin
  for mode,pairs in (('raw',raw),('online',on)):
   q=torch.cat((torch.stack([v[0] for v in pairs]),torch.stack([v[1] for v in reversed(pairs)]))).sort(dim=0).values
   q=torch.cat((q[:49],(q[48:49]+q[49:50])/2,q[49:]))
   for n in (8,32):
    ts=time.perf_counter();z=couple(q,paths[:n],p);value=target_queries(z.flatten(0,1),qs).reshape(n,12,len(qs),4).cpu();samples[f'XPF49_ECC{n}_{mode}'].append(value)
    torch.cuda.synchronize();timing.append(dict(start=start,mode=mode,paths=n,all_queries_seconds=time.perf_counter()-ts,paths32_seconds=sample_seconds))
  # All raw/online couplings have been issued before complete residual is used.
  residual=data['y'][ids]-p
  for c in controls:c.reveal(residual)
  print('ECC',start,flush=True)
 y=target_queries(data['y'],qs);rows=[];predictions={}
 for method,parts in samples.items():
  sample=torch.cat(parts,1).cuda();score=empirical_crps(sample,y)*std;lo,hi=torch.quantile(sample,torch.tensor([.05,.95],device='cuda'),dim=0);coverage=((y>=lo)&(y<=hi)).float();width=(hi-lo)*std
  prob=(sample[:,:,:,None,2:]>thresholds[None,None,None]).float().mean(0);truth=y[:,:,None,2:]>thresholds[None,None];brier=(prob-truth.float()).square();predictions[method]=dict(samples=sample.cpu(),event=prob.cpu())
  score,coverage,width,prob,truth,brier=[v.cpu() for v in (score,coverage,width,prob,truth,brier)]
  for i,meta in enumerate(data['meta']):
   for j,q in enumerate(qs):
    for c in range(2):
     for stat in range(2):
      k=c+2*stat;row=dict(method=method,**meta,query=j,category=q['category'],channel=c,statistic=('mean','max')[stat],crps=float(score[i,j,k]),coverage=float(coverage[i,j,k]),width=float(width[i,j,k]))
      if stat:
       for t in range(3):row.update({f'brier{t}':float(brier[i,j,t,c]),f'prob{t}':float(prob[i,j,t,c]),f'event{t}':int(truth[i,j,t,c])})
      rows.append(row)
 folder=OUT/'query';keys=sorted(set().union(*(r.keys() for r in rows)))
 with (folder/'ecc_per_query.csv').open('w',newline='') as f:
  wr=csv.DictWriter(f,fieldnames=keys);wr.writeheader();wr.writerows(rows)
 save(predictions,folder/'ecc_predictions.pt');dump(folder/'ecc_timing.json',timing)
if __name__=='__main__':main()
