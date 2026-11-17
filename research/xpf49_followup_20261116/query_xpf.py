"""Matched query heads using exact legacy XPF quantile outputs."""
from followcommon import *
from torch import nn
from torch.nn import functional as F
from functorch import combine_state_for_ensemble,vmap
from common import bases,crps
import math,csv,argparse
ARMS=('XP','XPF','XPQ','XPFQ','Q')
class Head(nn.Module):
 def __init__(self):
  super().__init__();self.net=nn.Sequential(nn.Linear(1112,128),nn.SiLU(),nn.Linear(128,128),nn.SiLU(),nn.Linear(128,28))
 def components(self,x,m):
  v=torch.cat((x[:,:120]*m[0],x[:,120:240]*m[1],x[:,240:384]*m[2],x[:,384:392],x[:,392:]*m[3]),1)
  o=self.net(v);return o[:,:4],F.softplus(o[:,4:].reshape(-1,4,6))/6
 def forward(self,x,m,tau):
  c,w=self.components(x,m);return c+(w*bases(tau)[:,None]).sum(2)
 def quantiles(self,x,m,tau):
  c,w=self.components(x,m);return c[None]+torch.einsum('bck,qk->qbc',w,bases(tau))
def mask_for(a):return torch.tensor((a!='Q',a!='Q','F' in a,'Q' in a),device='cuda',dtype=torch.float32)
def train():
 setup();folder=OUT/'query';d=load(folder/'train.pt','cuda');x=d['z'].flatten(0,1);y=d['y'].flatten(0,1)
 if all((folder/f'{a}.pt').exists() for a in ARMS):return
 torch.manual_seed(42);template=Head().cuda();initial={k:v.detach().clone() for k,v in template.state_dict().items()};models=[]
 for _ in ARMS:
  h=Head().cuda();h.load_state_dict(initial);models.append(h)
 fun,params,buf=combine_state_for_ensemble(models);params=tuple(p.requires_grad_() for p in params);names=[k for k,_ in template.named_parameters()];masks=torch.stack([mask_for(a) for a in ARMS])
 optimizer=torch.optim.AdamW(params,lr=3e-4,weight_decay=1e-5);rng=torch.Generator(device='cuda').manual_seed(42);logs=[];begin=time.monotonic();steps=4000
 for step in range(1,steps+1):
  ids=torch.randint(len(x),(256,),device='cuda',generator=rng);tau=torch.rand(256,device='cuda',generator=rng).clamp(.002,.998)
  optimizer.zero_grad(set_to_none=True);q=vmap(fun,in_dims=(0,0,None,0,None))(params,buf,x[ids],masks,tau)
  error=y[ids][None]-q;t=tau[None,:,None];loss=torch.maximum(t*error,(t-1)*error).mean((1,2));assert torch.isfinite(loss).all();loss.sum().backward()
  size=sum(p.grad.flatten(1).square().sum(1) for p in params).sqrt();factor=(1/(size+1e-6)).clamp(max=1)
  for p in params:p.grad.mul_(factor.reshape(len(ARMS),*([1]*(p.ndim-1))))
  optimizer.param_groups[0]['lr']=1e-6+(3e-4-1e-6)*.5*(1+math.cos(math.pi*(step-1)/steps));optimizer.step()
  if step==1 or step%200==0:
   logs.append(dict(step=step,loss=loss.tolist(),elapsed=time.monotonic()-begin));print('QUERY_TRAIN',logs[-1],flush=True)
 init=hashlib.sha256(b''.join(initial[k].cpu().numpy().tobytes() for k in sorted(initial))).hexdigest()
 for i,a in enumerate(ARMS):save(dict(model={k:p[i].detach().cpu() for k,p in zip(names,params)},seed=42,arm=a,steps=steps,parameters=sum(p.numel() for p in template.parameters()),initial_sha256=init,logs=logs),folder/f'{a}.pt')
@torch.no_grad()
def evaluate():
 setup();folder=OUT/'query';levels=torch.arange(1,100,device='cuda')/100;thresholds=load(PRIOR/'query/statistics.pt','cuda')['thresholds'];std=norm()['std'].flatten().repeat(2);qs=load(folder/'development.pt','cuda');cal=load(folder/'calibration.pt','cuda');rows=[];predictions={};cal_predictions={}
 for arm in ARMS:
  h=Head().cuda().eval();h.load_state_dict(load(folder/f'{arm}.pt')['model'])
  for split,d in (('development',qs),('calibration',cal)):
   x=d['z'].flatten(0,1);y=d['y'].flatten(0,1);k=len(d['queries']);n=len(d['meta'])
   q=torch.cat([h.quantiles(x[j:j+1024],mask_for(arm),levels) for j in range(0,len(x),1024)],1);assert (q[1:]>=q[:-1]-1e-6).all()
   event=cdf(q[:,:,2:],thresholds,levels);truth=y[:,None,2:]>thresholds[None];assert (event[:,1:]<=event[:,:-1]+1e-6).all()
   entry=dict(q=q.reshape(99,n,k,4).cpu(),event=event.reshape(n,k,3,2).cpu())
   if split=='calibration':cal_predictions[arm]=entry['event'];continue
   score=crps(q,y,levels)*std;coverage=((y>=q[4])&(y<=q[94])).float();width=(q[94]-q[4])*std;brier=(event-truth.float()).square()
   score,coverage,width,event,truth,brier=[v.cpu() for v in (score,coverage,width,event,truth,brier)]
   for i,meta in enumerate(d['meta']):
    for j,query in enumerate(d['queries']):
     ix=i*k+j
     for c in range(2):
      for stat in range(2):
       hix=c+2*stat;row=dict(method=arm,**meta,query=j,category=query['category'],channel=c,statistic=('mean','max')[stat],crps=float(score[ix,hix]),coverage=float(coverage[ix,hix]),width=float(width[ix,hix]))
       if stat:
        for t in range(3):row.update({f'brier{t}':float(brier[ix,t,c]),f'prob{t}':float(event[ix,t,c]),f'event{t}':int(truth[ix,t,c])})
       rows.append(row)
   predictions[arm]=entry;print('QUERY_EVAL',arm,flush=True)
 keys=sorted(set().union(*(r.keys() for r in rows)))
 with (folder/'per_query.csv').open('w',newline='') as f:
  wr=csv.DictWriter(f,fieldnames=keys);wr.writeheader();wr.writerows(rows)
 aa,bb=zip(*contain_pairs(qs['queries']));viol={}
 for arm,v in predictions.items():
  diff=v['event'][:,aa]-v['event'][:,bb];viol[arm]=dict(pairs=len(aa),fraction=float((diff>1e-6).float().mean()),mean_excess=float(diff.clamp_min(0).mean()),max_excess=float(diff.max()))
 save(dict(predictions=predictions,calibration=cal_predictions,levels=levels.cpu(),thresholds=thresholds.cpu(),targets=qs['y'].cpu(),calibration_targets=cal['y'].cpu(),queries=qs['queries'],meta=qs['meta']),folder/'predictions.pt');dump(folder/'inclusion_violations.json',viol)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['train','evaluate']);a=ap.parse_args();globals()[a.stage]()
