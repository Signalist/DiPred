"""Rebuild primary held-out-Re table from raw HDF5, frozen backbone and released heads."""
from pathlib import Path
import sys,json,argparse
import numpy as np
import pandas as pd
import torch
W=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(W/'recoverability_20261118'))
import evaluation_recoverability as ev
ctl=ev.ctl;local=ev.local;alloc=ev.alloc;O=ev.OUT
from common import MANIFEST,read

def main():
 ctl.setup();assets=O/'primary_assets';manifest=json.loads((assets/'asset_manifest.json').read_text());assert ctl.digest(ctl.SHARED)==manifest['backbone_sha256']
 for name,sha in manifest['files'].items():assert ctl.digest(assets/name)==sha,name
 n=ctl.st.norm('cpu');model=ctl.st.model();records=json.loads((assets/'manifest.json').read_text())['files']['test'];shared=ctl.load(assets/'routers_shared.pt');xx=[];pp=[];ff=[];yy=[];meta=[]
 with torch.no_grad():
  for rec in records:
   for start in range(0,161,20):
    raw=read(rec,start);x=(raw[:,:20]-n['mean'][0])/n['std'][0];p,f=ctl.st.capture(model,x[None].cuda());xx.append(x);pp.append(p[0].cpu());ff.append(f['F_last'][0].cpu());yy.append((raw[:,20:]-n['mean'][0])/n['std'][0]);meta.append(dict(file=rec['file'],group=str(rec['group']),start=start))
  x=torch.stack(xx).cuda();p=torch.stack(pp).cuda();f=torch.stack(ff).cuda();desc=torch.cat([local.pooled(x),local.pooled(p),f],1).cpu().double();issued=[]
  for seed in ev.SEEDS:
   ck=ctl.load(assets/f'warp_{seed}.pt','cuda');head=local.Head('warp').cuda().eval();head.load_state_dict(ck['model']);z=alloc.project(desc,shared['projection']);priority=(torch.cat([torch.ones_like(z[:,:1]),z],1)@shared['routers'][str(seed)]['benefit']['coef']).numpy();mask=alloc.choose(meta,priority,.5);q=torch.cat([head(ctl.project_feature(f[i:i+8],ck['projection']),p[i:i+8],ck['gain']).cpu() for i in range(0,len(x),8)]);issued.append(dict(seed=seed,mask=mask,q=q))
  # Truth participates only after all candidate predictions and action masks are finalized.
  y=torch.stack(yy).cuda();base=ctl.mse(p,y).cpu().numpy();rows=[]
  for z in issued:
   correct=ctl.mse(z['q'].cuda(),y).cpu().numpy();m=z['mask'];
   for i,info in enumerate(meta):rows.append(dict(seed=z['seed'],**info,benefit=float(correct[i] if m[i] else base[i]),random_expected=float((base[i]+correct[i])/2),action=int(m[i])))
 df=pd.DataFrame(rows);df.to_csv(O/'reproduced_raw_windows.csv',index=False);tr=df.groupby(['seed','file','group'])[['benefit','random_expected']].mean().reset_index();summary=tr.groupby('seed')[['benefit','random_expected']].mean();summary['gain_percent']=(1-summary.benefit/summary.random_expected)*100;summary.to_csv(O/'reproduced_raw_summary.csv')
 original=pd.read_csv(O/'primary.csv');maximum=max(abs(float(summary.loc[s,'gain_percent'])-float(original[(original.split=='test')&(original.seed==s)&(original.quota==.5)].gain_percent.iloc[0])) for s in ev.SEEDS);ctl.dump(O/'reproduce_raw_audit.json',dict(passed=maximum<1e-5,absolute_gain_percent_max_error=maximum,raw_files=len(records),windows=len(meta),backbone_sha256=ctl.digest(ctl.SHARED),protocol_sha256=ctl.digest(W/'recoverability_20261118/PROTOCOL.md'),note='No training or calibration rerun. Raw-to-table replay for frozen seed models; three seed runs share one pretrained backbone.'));assert maximum<1e-5,maximum;print('RAW_TO_PRIMARY_REPRODUCED',maximum,flush=True)
if __name__=='__main__':main()
