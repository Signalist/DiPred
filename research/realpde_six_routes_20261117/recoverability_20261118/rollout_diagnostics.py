"""Analyze archived 200-frame free runs; no autoregressive retraining."""
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
import torch
W=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(W/'f_risk_query_20261116'))
from common import load,norm,read,records
O=W/'runs/recoverability_20261118'

def diagnostics(q,y,dx,dy):
 err=q-y;s=torch.fft.rfft2(err);h,w=y.shape[-2:];ky=torch.fft.fftfreq(h).reshape(h,1);kx=torch.fft.rfftfreq(w).reshape(1,-1);radius=torch.sqrt(kx*kx+ky*ky);power=s.abs().square();hi=float(power[...,radius>.25].sum()/power.sum().clamp_min(1e-20));curl=lambda a:(a[1,:,1:-1,2:]-a[1,:,1:-1,:-2])/(2*dx)-(a[0,:,2:,1:-1]-a[0,:,:-2,1:-1])/(2*dy);cq=curl(q);cy=curl(y);iq=int(cq.abs().reshape(-1).argmax());iy=int(cy.abs().reshape(-1).argmax());hw=(h-2)*(w-2);dq=(iq%hw)//(w-2);dy0=(iy%hw)//(w-2);eq=(iq%hw)%(w-2);ey=(iy%hw)%(w-2)
 return dict(mse=float(err.square().mean()),high_frequency_error_power_fraction=hi,truth_enstrophy=float(.5*cy.square().mean()),pred_enstrophy=float(.5*cq.square().mean()),vorticity_peak_space_distance_grid=float(np.hypot(dq-dy0,eq-ey)),vorticity_peak_time_distance_frames=abs(iq//hw-iy//hw))

def main():
 arch=load(W/'runs/direct_correction_controls_20261117/rollout_issued.pt');n=norm('cpu');coords=json.loads((W/'runs/goal_task_error_20261117/coordinates.json').read_text());rec={v['file']:v for v in records('development')};rows=[]
 for method in ['base','warp_F_last','rescrct_original']:
  for item in arch[method]:
   truth=torch.cat([read(rec[item['file']],s)[:,:20] for s in range(20,201,20)],1).reshape(2,10,20,32,64).permute(1,0,2,3,4)
   p=item['free']*n['std']+n['mean']
   for i in range(10):rows.append(dict(method=method,file=item['file'],group=rec[item['file']]['group'],block=i+1,**diagnostics(p[i],truth[i],coords['dx'],coords['dy'])))
 df=pd.DataFrame(rows);df.to_csv(O/'rollout_spectrum_vorticity.csv',index=False);summary=df.groupby(['method','block'])[['mse','high_frequency_error_power_fraction','pred_enstrophy','truth_enstrophy','vorticity_peak_space_distance_grid','vorticity_peak_time_distance_frames']].mean();summary.to_csv(O/'rollout_spectrum_summary.csv');print('ROLLOUT_DIAGNOSTICS',len(rows),flush=True)
if __name__=='__main__':main()
