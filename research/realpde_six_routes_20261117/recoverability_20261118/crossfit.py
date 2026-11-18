"""Post-hoc: 5-fold trajectory-held-out warp gains for a benefit readout."""
from pathlib import Path
import sys,time,hashlib
import numpy as np
import pandas as pd
import torch
W=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(W/'recoverability_20261118'))
import evaluation_recoverability as ev
ctl=ev.ctl;loc=ev.local;alloc=ev.alloc;O=ev.OUT

def main():
 ctl.setup();d=ctl.data('train');names=sorted({m['file'] for m in d['meta']});assert len(names)==46
 # Fixed SHA ordering before fitting; trajectory is the indivisible fold unit.
 ordered=sorted(names,key=lambda n:hashlib.sha256(('42|'+n).encode()).hexdigest());fold={n:i%5 for i,n in enumerate(ordered)};fold_ids=np.array([fold[m['file']] for m in d['meta']]);y=d['y'];p=d['p'];z=d['z'];base=ctl.mse(p,y).cpu().numpy();errors=np.zeros(len(base));records=[];unit=ctl.st.norm()['std'].square().mean()
 for f in range(5):
  torch.manual_seed(42);head=loc.Head('warp').cuda();opt=torch.optim.AdamW(head.parameters(),lr=3e-4,weight_decay=1e-5);rng=torch.Generator(device='cuda').manual_seed(42);train=torch.tensor(np.where(fold_ids!=f)[0],device='cuda');held=np.where(fold_ids==f)[0]
  for step in range(2000):
   ix=train[torch.randint(len(train),(8,),device='cuda',generator=rng)];opt.param_groups[0]['lr']=1e-6+(3e-4-1e-6)*.5*(1+np.cos(np.pi*step/2000));opt.zero_grad();loss=ctl.mse(head(z[ix],p[ix]),y[ix]).mean()/unit;loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),1.);opt.step()
  head.eval()
  with torch.no_grad():
   for j in range(0,len(held),8):
    ix=held[j:j+8];errors[ix]=ctl.mse(head(z[ix],p[ix],.5),y[ix]).cpu().numpy()
  ctl.save(dict(model=head.state_dict(),fold=f,heldout_files=sorted(n for n in names if fold[n]==f)),O/f'crossfit_warp_fold{f}.pt');records.append(dict(fold=f,train_trajectories=len(set(names)-set(n for n in names if fold[n]==f)),heldout_trajectories=len(set(n for n in names if fold[n]==f)),heldout_windows=len(held)));print('CROSSFIT_FOLD',f,len(held),flush=True)
 # Representation PCA fitted without labels on all training inputs, not fold-isolated.
 x=d['x'];features=ctl.load(ctl.FEATURES/'train.pt')['features']['F_last'].cuda();des=torch.cat([loc.pooled(x),loc.pooled(p),features],1).cpu().double();router=ctl.load(O/'router_42_compact.pt');basis=router['projection'];zz=alloc.project(des,basis);target=torch.tensor(base-errors);coef=alloc.ridge(zz,target,.01);ctl.save(dict(projection=basis,coef=coef,alpha=.01,folds=records),O/'crossfit_router.pt')
 for i,m in enumerate(d['meta']):records.append(dict(**m,fold=int(fold_ids[i]),base_mse=base[i],crossfit_warp_mse=errors[i],crossfit_gain=base[i]-errors[i]))
 pd.DataFrame([r for r in records if 'file' in r]).to_csv(O/'crossfit_train_gains.csv',index=False)
 rows=[]
 for split in ['development','late','future','test']:
  a=ev.get_data(split);x,p,f=a['x'],a['p'],a['f'];meta=a['meta'];des=torch.cat([loc.pooled(x),loc.pooled(p),f],1).cpu().double();z=alloc.project(des,basis);pred=(torch.cat([torch.ones_like(z[:,:1]),z],1)@coef).numpy();mask=alloc.choose(meta,pred,.5);sc=pd.read_csv(O/'windows.csv');s=sc[(sc.split==split)&(sc.seed==42)].reset_index(drop=True);assert len(s)==len(mask);saved=np.where(mask,s.warp_mse,s.base_mse);random=(s.warp_mse+s.base_mse)/2;values=pd.DataFrame(dict(file=s.file,group=s.group,benefit=saved,random=random,priority=pred,gain=s.warp_realized_gain)).groupby(['file','group']).mean();rows.append(dict(split=split,router='crossfit',n=len(values),mse=float(values.benefit.mean()),random_mse=float(values.random.mean()),gain_pct=float((1-values.benefit.mean()/values.random.mean())*100)))
 pd.DataFrame(rows).to_csv(O/'crossfit_router_summary.csv',index=False);ctl.dump(O/'crossfit_manifest.json',dict(folds=5,seed=42,fold_unit='trajectory',unsupervised_projection_uses_all_training_inputs=True,head_labels_out_of_fold=True,head_gain=.5,alpha=.01,scope='post-hoc diagnostic after primary held-out results; not independent confirmation',training_trajectories=len(names)));print('CROSSFIT_DONE',rows,flush=True)
if __name__=='__main__':main()
