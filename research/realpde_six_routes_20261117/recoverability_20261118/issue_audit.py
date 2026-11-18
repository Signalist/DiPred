"""Independent replay: issuing code receives X/P/F and metadata only, never Y."""
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
import torch
W=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(W/'recoverability_20261118'))
import evaluation_recoverability as ev
ctl=ev.ctl;local=ev.local;alloc=ev.alloc;O=ev.OUT

def issue(x,p,f,meta,seed):
 assert len(meta)==len(x)==len(p)==len(f)
 ck=ctl.load(O/f'warp_{seed}.pt','cuda');router=ctl.load(O/f'router_{seed}.pt');head=local.Head('warp').cuda().eval();head.load_state_dict(ck['model']);desc=torch.cat([local.pooled(x),local.pooled(p),f],1).cpu().double();z=alloc.project(desc,router['projection']);r={}
 for name,m in router['models'].items():
  priority=(torch.cat([torch.ones_like(z[:,:1]),z],1)@m['coef']).numpy();r[name]={'priority':priority,'mask':alloc.choose(meta,priority,.5)}
 with torch.no_grad():q=torch.cat([head(ctl.project_feature(f[i:i+8],ck['projection']),p[i:i+8],ck['gain']).cpu() for i in range(0,len(x),8)])
 return r,q

def main():
 ctl.setup();ev.verify();prior=pd.read_csv(O/'policies.csv');errors=pd.read_csv(O/'windows.csv');out=[]
 for split in ['development','late','future','test']:
  d=ev.get_data(split);x,p,f,meta=d['x'],d['p'],d['f'],d['meta'];y=d.pop('y');del d
  # issue() has no truth argument; replay before any subsequent score here.
  for seed in ev.SEEDS:
   policies,q=issue(x,p,f,meta,seed);expected=prior[(prior.split==split)&(prior.seed==seed)&(prior.quota==.5)];saved=errors[(errors.split==split)&(errors.seed==seed)];err=ctl.mse(q.cuda(),y).cpu().numpy();maxerr=float(np.max(np.abs(err-saved.warp_mse.to_numpy())));assert maxerr<1e-10
   for name in ['benefit','risk']:
    savedmask=expected[expected.policy==name].set_index(['file','start']).loc[pd.MultiIndex.from_frame(pd.DataFrame(meta)[['file','start']])]['mask'].to_numpy().astype(bool);assert np.array_equal(savedmask,policies[name]['mask']);out.append(dict(split=split,seed=seed,policy=name,mask_exact=True,mse_replay_max_error=maxerr,quota=int(savedmask.sum())))
   # Input intervention in Y cannot affect issue(): no Y is accepted by its signature.
 pd.DataFrame(out).to_csv(O/'issue_audit.csv',index=False);ctl.dump(O/'issue_audit.json',dict(passed=True,replay_max_mse_error=max(r['mse_replay_max_error'] for r in out),rows=len(out),scope='replay issue code has no Y argument; original retrospective test results were already read before this replay, so this is causal implementation verification not a new blind evaluation'))
 print('ISSUE_AUDIT_OK',len(out),flush=True)
if __name__=='__main__':main()
