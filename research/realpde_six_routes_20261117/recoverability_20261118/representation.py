"""Exploratory feature readouts of the same frozen warp benefit labels."""
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import torch
W=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(W/'compute_allocation_20261117'))
import allocation as a
O=W/'runs/recoverability_20261118';S=W/'runs/error_structure_20261117'
KINDS=['X','P','X+P','F_last','F_mid0','F_mid99','F_det4_1500','F_det4_6000','X+P+F_last']

def run():
 labels=a.s.load(O/'train_cal_errors.pt')['42']['warp'];gain=labels['gain'];data={k:a.s.load(S/f'{k}.pt') for k in ['train','calibration','development','late','future']};calbase=labels['base']['calibration'];calcorrected=labels['errors']['calibration'][str(gain)];result=[];fitted={}
 for kind in KINDS:
  # All representations are extracted by the existing four-step capture code;
  # deterministic control training budgets did not exactly match diffusion.
  def raw(split):return torch.cat([data[split]['features'][part] for part in kind.split('+')],1).double()
  x=raw('train');b=a.basis_fit(x,16);z=a.project(x,b);zc=a.project(raw('calibration'),b);y=torch.tensor(labels['base']['train']-labels['errors']['train'][str(gain)],dtype=torch.float64);opts=[]
  for alpha in [.01,.1,1.,10.]:
   coef=a.ridge(z,y,alpha);pr=(torch.cat([torch.ones_like(zc[:,:1]),zc],1)@coef).numpy();m=a.choose(data['calibration']['meta'],pr,.5);loss=a.mean_trajectory(np.where(m,calcorrected,calbase),data['calibration']['meta']);opts.append((loss,alpha,coef))
  loss,alpha,coef=min(opts,key=lambda q:q[0]);fitted[kind]={'basis':b,'coef':coef,'alpha':alpha,'calibration_mse':loss}
  for split in ['development','late','future']:
   zz=a.project(raw(split),b);pr=(torch.cat([torch.ones_like(zz[:,:1]),zz],1)@coef).numpy();d=data[split];df=pd.read_csv(O/'windows.csv');r=df[(df.split==split)&(df.seed==42)];assert len(r)==len(pr);truth=r.warp_realized_gain.to_numpy();base=r.base_mse.to_numpy();corr=r.warp_mse.to_numpy();mask=a.choose(d['meta'],pr,.5);rmean=a.mean_trajectory((base+corr)/2,d['meta']);pmean=a.mean_trajectory(np.where(mask,corr,base),d['meta']);result.append(dict(kind=kind,split=split,alpha=alpha,calibration_mse=loss,spearman=float(spearmanr(pr,truth).correlation),rmse=float(np.sqrt(np.mean((pr-truth)**2))),router_gain_pct=float((1-pmean/rmean)*100)))
  print('REP',kind,flush=True)
 pd.DataFrame(result).to_csv(O/'representation_diagnostics.csv',index=False);a.s.save(fitted,O/'representation_models.pt');print('REP_DONE',flush=True)
if __name__=='__main__':a.s.setup();run()
