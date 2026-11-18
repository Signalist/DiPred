"""Use six archived samples to inspect tail dependence and task consistency."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
W=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(W/'goal_task_error_20261117'))
from task_error import Tasks,TASKS
O=W/'runs/recoverability_20261118';SOURCE=W/'runs/lowcost_joint_20261117/conditional_scores/scorefit'

def main():
 task=Tasks();rows=[];components=[]
 for split in ['development','late','future']:
  readout=torch.load(W/f'runs/goal_task_error_20261117/{split}/predictions.pt',map_location='cpu',weights_only=False)['direct/F_det4_1500/16']
  for path in sorted((SOURCE/split).glob('audit_*.pt')):
   i=int(path.stem.split('_')[-1]);d=torch.load(path,map_location='cpu',weights_only=False);mu=d['norm']['mean'];sd=d['norm']['std'];y=d['y'][None]*sd+mu;truth=task.values(y.double())[0];direct=readout[i]
   for name in ['smooth','ar__mixed','secc3','ecc32']:
    z=d['samples'][name]*sd+mu;vals=task.values(z.double());speed=z.square().sum(1).sqrt();target=y.square().sum(1).sqrt();smax=speed.flatten(1).amax(1);tmax=float(target.max());loc=speed.flatten(1).argmax(1);loc_truth=int(target.flatten().argmax());time_mean=speed.mean((-1,-2));a=time_mean[:,:-1].flatten().numpy();b=time_mean[:,1:].flatten().numpy();adjacent=float(np.corrcoef(a,b)[0,1]);rank=float((smax<=tmax).float().mean());q90=float(torch.quantile(smax,.9));location_t=(loc//(32*64)).float();location_y=((loc%(32*64))//64).float();location_x=(loc%64).float();tt=loc_truth//(32*64);yy=loc_truth%(32*64)//64;xx=loc_truth%64
    rows.append(dict(split=split,index=i,method=name,truth_max=tmax,sample_mean_max=float(smax.mean()),sample_q90_max=q90,truth_rank=rank,adjacent_time_corr=adjacent,mean_peak_space_distance=float(((location_y-yy).square()+(location_x-xx).square()).sqrt().mean()),mean_peak_time_distance=float((location_t-tt).abs().mean()),max_event_prob=float((smax>tmax).float().mean())))
    for k,label in enumerate(TASKS):components.append(dict(split=split,index=i,method=name,task=label,truth=float(truth[k]),direct_readout=float(direct[k]),field_task_mean=float(vals[:,k].mean()),field_task_q05=float(torch.quantile(vals[:,k],.05)),field_task_q95=float(torch.quantile(vals[:,k],.95)),direct_in_field90=bool((direct[k]>=torch.quantile(vals[:,k],.05)) and (direct[k]<=torch.quantile(vals[:,k],.95))),truth_in_field90=bool((truth[k]>=torch.quantile(vals[:,k],.05)) and (truth[k]<=torch.quantile(vals[:,k],.95)))))
 pd.DataFrame(rows).to_csv(O/'tail_snapshots.csv',index=False);pd.DataFrame(components).to_csv(O/'task_field_consistency.csv',index=False);print('TAIL_TASK_SNAPSHOTS',len(rows),len(components),flush=True)
if __name__=='__main__':main()
