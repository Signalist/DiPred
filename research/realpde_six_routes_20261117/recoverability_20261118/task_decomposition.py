"""Task bias decomposition: exact algebra, not causal attribution of learned heads."""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
W=Path(__file__).resolve().parent.parent;O=W/'runs/recoverability_20261118';S=W/'runs/goal_task_error_20261117';TASKS=['kinetic','fluctuation','enstrophy','regional_u'];rows=[]
for split in ['train','calibration','development','late','future']:
 d=torch.load(S/f'{split}.pt',map_location='cpu',weights_only=False);pred=None if split=='train' else torch.load(S/split/'predictions.pt',map_location='cpu',weights_only=False)
 for i,m in enumerate(d['meta']):
  for j,name in enumerate(TASKS):
   v=dict(split=split,**m,task=name,truth=float(d['jy'][i,j]),base=float(d['jp'][i,j]),signed_error=float(d['jy'][i,j]-d['jp'][i,j]),linear=float(d['linear'][i,j]),quadratic=float(d['quad'][i,j]))
   if pred is not None:
    for k in ['direct/F_det4_1500/16','direct/X+P/16','task_affine','direct/task_stats/8']:v[k]=float(pred[k][i,j])
   rows.append(v)
df=pd.DataFrame(rows);assert np.max(np.abs(df.signed_error-df.linear-df.quadratic))<1e-9;df.to_csv(O/'task_decomposition_windows.csv',index=False)
tr=df.groupby(['split','file','group','task'],as_index=False).mean(numeric_only=True);tr.to_csv(O/'task_decomposition_trajectories.csv',index=False);summary=tr.groupby(['split','task']).mean(numeric_only=True);summary.to_csv(O/'task_decomposition_summary.csv');print('TASK_DECOMPOSITION',len(df),flush=True)
