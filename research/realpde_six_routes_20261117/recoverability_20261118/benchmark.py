"""Actual common-backbone batch service timing, one GPU, frozen weights."""
from pathlib import Path
import sys,time
import numpy as np
import pandas as pd
import torch
W=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(W/'recoverability_20261118'))
import evaluation_recoverability as ev
ctl=ev.ctl;local=ev.local;alloc=ev.alloc;O=ev.OUT

def main():
 ctl.setup();d=ev.get_data('development');n=ctl.st.norm();physical=(d['x'][:12]*n['std']+n['mean']).cpu();net=ctl.st.model();warp=local.Head('warp').cuda().eval();ck=ctl.load(O/'warp_42.pt','cuda');warp.load_state_dict(ck['model']);res=ctl.factory('rescrct_original').cuda().eval();rc=ctl.load(O/'rescrct_original_42.pt','cuda');res.load_state_dict(rc['model']);router=ctl.load(O/'router_42.pt');rows=[]
 @torch.no_grad()
 def once(policy):
  x=(physical.cuda()-n['mean'])/n['std'];p=[];f=[]
  for i in range(len(x)):
   a,b=ctl.st.capture(net,x[i:i+1]);p.append(a);f.append(b['F_last'])
  p=torch.cat(p);f=torch.cat(f)
  if policy=='benefit50':
   desc=torch.cat([local.pooled(x),local.pooled(p),f],1).cpu().double();z=alloc.project(desc,router['projection']);priority=(torch.cat([torch.ones_like(z[:,:1]),z],1)@router['models']['benefit']['coef']).numpy();mask=alloc.choose([dict(file=str(i),start=0) for i in range(len(x))],priority,.5)
  else:mask=np.ones(len(x),dtype=bool)
  out=[]
  for i in range(len(x)):
   if policy=='base' or not mask[i]:q=p[i:i+1]
   elif policy=='rescrct_original':q=ctl.mix(res,x[i:i+1],p[i:i+1],None,rc['gain'])
   else:q=warp(ctl.project_feature(f[i:i+1],ck['projection']),p[i:i+1],ck['gain'])
   out.append((q*n['std']+n['mean']).cpu())
  return out
 for p in ['base','warp','rescrct_original','benefit50']:once(p)
 for rep in range(5):
  for p in ['base','warp','rescrct_original','benefit50']:
   torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();t=time.perf_counter();once(p);torch.cuda.synchronize();rows.append(dict(method=p,repeat=rep,requests=12,batch_ms=(time.perf_counter()-t)*1000,peak_allocated_mb=torch.cuda.max_memory_allocated()/1024**2))
 df=pd.DataFrame(rows);df.to_csv(O/'timing_samples.csv',index=False);a=df.groupby('method').batch_ms.agg(['median','min','max']);a['per_request_ms']=a['median']/12;a['scope']='CPU physical histories -> serial 4-call GPU P/F -> policy/correction -> CPU physical outputs; 12 simultaneous requests; warmed; excludes setup';a.to_csv(O/'timing.csv');print('TIMING',a.to_string(),flush=True)
if __name__=='__main__':main()
