"""Evaluate only after train/cal selection is frozen; no hyperparameter updates."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
import pandas as pd
import torch
W=Path(__file__).resolve().parent.parent
for n in ['direct_correction_controls_20261117','error_structure_20261117','compute_allocation_20261117']:sys.path.insert(0,str(W/n))
import controls as ctl
import local_learn as local
import allocation as alloc
from common import DATA,MANIFEST,read
OUT=W/'runs/recoverability_20261118'
SEEDS=[42,123,2025]

def verify():
 sel=json.loads((OUT/'selection.json').read_text());assert sel['test_access_before_freeze'] is False;assert sel['protocol_sha256']==ctl.digest(W/'recoverability_20261118/PROTOCOL.md')
 for name,sha in sel['weights'].items():assert ctl.digest(OUT/name)==sha,(name,'changed')
 return sel

def prepare_test():
 verify();dest=OUT/'test.pt'
 if dest.exists():return
 m=ctl.st.model();n=ctl.st.norm('cpu');manifest=json.loads(MANIFEST.read_text());test=manifest['files']['test'];other={r['file'] for split in ('train','calibration','development') for r in manifest['files'][split]};assert other.isdisjoint(r['file'] for r in test)
 xs=[];ys=[];ps=[];features=[];meta=[]
 with torch.no_grad():
  for rec in test:
   for start in range(0,161,20):
    raw=read(rec,start);x=(raw[:,:20]-n['mean'][0])/n['std'][0];y=(raw[:,20:]-n['mean'][0])/n['std'][0];p,f=ctl.st.capture(m,x[None].cuda());xs.append(x);ys.append(y);ps.append(p[0].cpu());features.append(f['F_last'][0].cpu());meta.append(dict(file=rec['file'],group=str(rec['group']),start=start))
  ctl.save({'x':torch.stack(xs),'y':torch.stack(ys),'p':torch.stack(ps),'f':torch.stack(features),'meta':meta},dest)
 print('PREPARED_TEST',len(meta),len(test),flush=True)

def get_data(split):
 if split=='test':
  a=ctl.load(OUT/'test.pt');return {k:(v.cuda() if k in ('x','y','p','f') else v) for k,v in a.items()}
 d=ctl.data(split);features=ctl.load(ctl.FEATURES/f'{split}.pt')['features']['F_last'].cuda();return dict(x=d['x'],y=d['y'],p=d['p'],f=features,meta=d['meta'])

def paired(a,b):
 rng=np.random.default_rng(42);ix=rng.integers(0,len(a),(10000,len(a)));v=(b[ix]-a[ix]).mean(1);return np.quantile(v,[.025,.975])

def run():
 verify();prepare_test();rows=[];policies=[]
 with torch.no_grad():
  for split in ['development','late','future','test']:
   d=get_data(split);x,y,p,f=d['x'],d['y'],d['p'],d['f'];meta=d['meta'];desc=torch.cat([local.pooled(x),local.pooled(p),f],1).cpu().double();base=ctl.mse(p,y).cpu().numpy()
   for seed in SEEDS:
    router=ctl.load(OUT/f'router_{seed}.pt');z=alloc.project(desc,router['projection']);priority={k:(torch.cat([torch.ones_like(z[:,:1]),z],1)@v['coef']).numpy() for k,v in router['models'].items()};models={}
    for method in ['warp','rescrct_original']:
     ck=ctl.load(OUT/f'{method}_{seed}.pt','cuda');h=(local.Head('warp') if method=='warp' else ctl.factory(method)).cuda().eval();h.load_state_dict(ck['model']);pro=ck['projection'] if method=='warp' else None;parts=[]
     for i in range(0,len(p),8):
      q=h(ctl.project_feature(f[i:i+8],pro),p[i:i+8],ck['gain']) if method=='warp' else ctl.mix(h,x[i:i+8],p[i:i+8],None,ck['gain']);parts.append(ctl.mse(q,y[i:i+8]).cpu())
     models[method]=torch.cat(parts).numpy();del h
    for i,m in enumerate(meta):rows.append(dict(split=split,seed=seed,**m,base_mse=base[i],warp_mse=models['warp'][i],rescrct_mse=models['rescrct_original'][i],benefit_prediction=priority['benefit'][i],risk_prediction=priority['risk'][i],warp_realized_gain=base[i]-models['warp'][i],rescrct_realized_gain=base[i]-models['rescrct_original'][i]))
    for quota in [0.,.25,.5,.75,1.]:
     for name,values in [('benefit',priority['benefit']),('risk',priority['risk']),('oracle',base-models['warp'])]:
      mask=alloc.choose(meta,values,quota)
      for i,m in enumerate(meta):policies.append(dict(split=split,seed=seed,**m,policy=name,quota=quota,mask=int(mask[i]),priority=float(values[i]),mse=float(models['warp'][i] if mask[i] else base[i])))
     ix=pd.DataFrame(meta)
     for _,g in ix.groupby('start'):
      ratio=np.floor(quota*len(g)+.5)/len(g)
      for i in g.index:
       m=meta[i];policies.append(dict(split=split,seed=seed,**m,policy='random_expected',quota=quota,mask=ratio,priority=np.nan,mse=float((1-ratio)*base[i]+ratio*models['warp'][i])))
   print('EVALUATED',split,len(meta),flush=True)
 df=pd.DataFrame(rows);df.to_csv(OUT/'windows.csv',index=False);pol=pd.DataFrame(policies);pol.to_csv(OUT/'policies.csv',index=False)
 g=pol.groupby(['split','seed','policy','quota','file','group'],as_index=False).agg(mse=('mse','mean'),fraction=('mask','mean'));g.to_csv(OUT/'policy_trajectories.csv',index=False)
 tables=[]
 for (split,seed,quota),v in g.groupby(['split','seed','quota']):
  wide=v.pivot(index='file',columns='policy',values='mse');a=wide['benefit'].to_numpy();b=wide['random_expected'].to_numpy();lo,hi=paired(a,b);groups=v.drop_duplicates('file').set_index('file').loc[wide.index,'group'];per=pd.DataFrame({'group':groups,'diff':b-a}).groupby('group').diff.mean() if False else pd.DataFrame({'group':groups,'delta':b-a}).groupby('group')['delta'].mean();tables.append(dict(split=split,seed=seed,quota=quota,delta=float((b-a).mean()),gain_percent=float((1-a.mean()/b.mean())*100),ci_low=float(lo),ci_high=float(hi),wins=int((a<b).sum()),n=len(a),worst_group_delta=float(per.min())))
 pd.DataFrame(tables).to_csv(OUT/'primary.csv',index=False);print('DONE_EVALUATE',flush=True)
if __name__=='__main__':ctl.setup();run()
