"""Frozen zero-shot cross-scene stress test; see CROSS_SCENE_PROTOCOL.md."""
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import pyarrow as pa
W=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(W/'recoverability_20261118'))
import evaluation_recoverability as ev
ctl=ev.ctl;local=ev.local;alloc=ev.alloc;OUT=ev.OUT
DATA=W.parent.parent/'OLD/DATASET-5PDE'
SCENES={'cylinder':[0,10,20,30,40,50,60,70],'fsi':[0,7,14,21,28,35,42,49]}

def freeze_test_data():
 ev.verify();dest=OUT/'cross_scene.pt'
 if dest.exists():return
 norm=ctl.st.norm('cpu');core=ctl.st.model();xs=[];ys=[];ps=[];fs=[];meta=[];ids=set()
 for scene,indices in SCENES.items():
  files=sorted((DATA/scene/'hf_dataset/real').glob('*.arrow'))
  for index in indices:
   path=files[index]
   with pa.memory_map(str(path),'r') as source:
    table=pa.ipc.open_stream(source).read_all();assert len(table)>=1,(path,len(table));sim=table['sim_id'][0].as_py();key=(scene,sim);assert key not in ids,key;ids.add(key)
    shape=tuple(int(table[k][0].as_py()) for k in ['shape_t','shape_h','shape_w']);assert shape[0]>=340
    uv=[]
    for key in ['u','v']:
     a=np.frombuffer(table[key][0].as_buffer(),dtype=np.float32).reshape(shape)
     uv.append(np.stack([a[start:start+40].copy() for start in [0,100,200,300]]))
    blocks=torch.from_numpy(np.stack(uv,axis=1));assert torch.isfinite(blocks).all(),(scene,sim,'nonfinite')
    blocks=F.interpolate(blocks.reshape(-1,1,*shape[1:]),size=(32,64),mode='area').reshape(4,2,40,32,64)
    for i,start in enumerate([0,100,200,300]):
     x=(blocks[i,:,:20]-norm['mean'][0])/norm['std'][0];y=(blocks[i,:,20:]-norm['mean'][0])/norm['std'][0]
     with torch.no_grad():p,f=ctl.st.capture(core,x[None].cuda())
     assert torch.isfinite(p).all();xs.append(x);ys.append(y);ps.append(p[0].cpu());fs.append(f['F_last'][0].cpu());meta.append(dict(scene=scene,file=f'{scene}:{sim}',group=scene,start=start,arrow=path.name,shape=shape))
    print('CAPTURE',scene,index,sim,flush=True)
 ctl.save(dict(x=torch.stack(xs),y=torch.stack(ys),p=torch.stack(ps),f=torch.stack(fs),meta=meta),dest)
 ctl.dump(OUT/'cross_scene_provenance.json',dict(protocol_sha256=ctl.digest(W/'recoverability_20261118/CROSS_SCENE_PROTOCOL.md'),records=meta,independent_confirmation=False))

def run():
 freeze_test_data();d=ctl.load(OUT/'cross_scene.pt');x,y,p,f=[d[k].cuda() for k in ('x','y','p','f')];meta=d['meta'];desc=torch.cat([local.pooled(x),local.pooled(p),f],1).cpu().double();prediction=[]
 # Store finalized outputs and assignments before using y.
 with torch.no_grad():
  for seed in ev.SEEDS:
   ck=ctl.load(OUT/f'warp_{seed}.pt','cuda');h=local.Head('warp').cuda().eval();h.load_state_dict(ck['model']);router=ctl.load(OUT/f'router_{seed}.pt');z=alloc.project(desc,router['projection']);pri={k:(torch.cat([torch.ones_like(z[:,:1]),z],1)@v['coef']).numpy() for k,v in router['models'].items()}
   mask={}
   for policy,values in pri.items():
    result=np.zeros(len(meta),dtype=bool)
    for scene in SCENES:
     idx=[i for i,m in enumerate(meta) if m['scene']==scene];mm=[meta[i] for i in idx];result[idx]=alloc.choose(mm,values[idx],.5)
    mask[policy]=result
   q=[]
   for i in range(0,len(x),8):q.append(h(ctl.project_feature(f[i:i+8],ck['projection']),p[i:i+8],ck['gain']).cpu())
   prediction.append(dict(seed=seed,q=torch.cat(q),priorities=pri,masks=mask))
 ctl.save(prediction,OUT/'cross_scene_issued.pt')
 rows=[]
 with torch.no_grad():
  base=ctl.mse(p,y).cpu().numpy()
  for pred in prediction:
   error=ctl.mse(pred['q'].cuda(),y).cpu().numpy()
   for i,m in enumerate(meta):rows.append(dict(**m,seed=pred['seed'],base=base[i],warp=error[i],benefit=error[i] if pred['masks']['benefit'][i] else base[i],risk=error[i] if pred['masks']['risk'][i] else base[i],random_expected=(base[i]+error[i])/2))
 df=pd.DataFrame(rows);df.to_csv(OUT/'cross_scene_windows.csv',index=False);tr=df.groupby(['scene','seed','file'],as_index=False)[['base','warp','benefit','risk','random_expected']].mean();tr.to_csv(OUT/'cross_scene_trajectories.csv',index=False);summary=[]
 for (scene,seed),v in tr.groupby(['scene','seed']):
  lo,hi=ev.paired(v.benefit.to_numpy(),v.random_expected.to_numpy());summary.append(dict(scene=scene,seed=seed,n=len(v),base=float(v.base.mean()),warp_gain_pct=float((1-v.warp.mean()/v.base.mean())*100),benefit_vs_random_pct=float((1-v.benefit.mean()/v.random_expected.mean())*100),delta=float((v.random_expected-v.benefit).mean()),ci_low=float(lo),ci_high=float(hi)))
 pd.DataFrame(summary).to_csv(OUT/'cross_scene_summary.csv',index=False);print('CROSS_SCENE_DONE',flush=True)
if __name__=='__main__':ctl.setup();run()
