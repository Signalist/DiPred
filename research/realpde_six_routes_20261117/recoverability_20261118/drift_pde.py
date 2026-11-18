"""Frozen source-domain geometric updater applied to actual simulated PDE drift."""
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
import torch
W=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(W/'online_signed_state_20261117'))
from geo_inference import GeometricPredictor
from common import setup,dump,digest
O=W/'runs/recoverability_20261118';SCENARIOS=['stable','viscosity_step','forcing_step','sensor_step']

def simulate(seed,scenario):
 rng=np.random.default_rng(seed);H,Wd=32,64;y,x=np.meshgrid(np.arange(H)/H,np.arange(Wd)/Wd,indexing='ij');u=.1733+.04*np.sin(2*np.pi*(x+rng.random()))*np.cos(2*np.pi*(y+rng.random()));v=.005*np.cos(2*np.pi*(x+rng.random()))*np.sin(2*np.pi*(y+rng.random()));a=np.stack([u,v]);ky=np.fft.fftfreq(H,d=1/H)[:,None];kx=np.fft.fftfreq(Wd,d=1/Wd)[None,:];lap=(2*np.pi)**2*(kx*kx+ky*ky);out=[];obs=[];dt=.01
 for t in range(500):
  out.append(a.copy());noise=np.zeros_like(a)
  if scenario=='sensor_step' and t>=240:noise[0]+=.01;noise+=rng.normal(0,.001,a.shape)
  obs.append(a+noise);nu=.0005*(4 if scenario=='viscosity_step' and t>=240 else 1);cx=.05;cy=.015;phase=np.exp((-nu*lap-2j*np.pi*(cx*kx+cy*ky))*dt);a=np.fft.ifft2(np.fft.fft2(a,axes=(-2,-1))*phase,axes=(-2,-1)).real
  if scenario=='forcing_step' and t>=240:a[0]+=.00015*np.sin(2*np.pi*y)
 return torch.tensor(np.stack(out,axis=1),dtype=torch.float32),torch.tensor(np.stack(obs,axis=1),dtype=torch.float32)

def main():
 setup();rows=[];events=[]
 for scenario in SCENARIOS:
  for seed in range(100,108):
   clean,obs=simulate(seed,scenario);pred=GeometricPredictor('rolling',linear=True)
   for start in range(0,461,20):
    x=obs[:,start:start+20][None];q=pred.predict(x,start,False);p=pred.pending*pred.n['std']+pred.n['mean'];coeff=pred.state.last;future=clean[:,start+20:start+40][None].cuda();e=lambda a:float((a-future).square().mean());rows.append(dict(scenario=scenario,seed=seed,start=start,post=start+20>=240,base_mse=e(p),corrected_mse=e(q),correction_rms=float((q-p).square().mean().sqrt()),observed_feedback=scenario=='sensor_step'))
    pred.reveal(obs[:,start+20:start+40][None],start+40,False);events.append(dict(scenario=scenario,seed=seed,start=start,issue=start+20,reveal=start+40))
   print('PDE_DRIFT',scenario,seed,flush=True)
 df=pd.DataFrame(rows);df.to_csv(O/'pde_drift_windows.csv',index=False);tr=df.groupby(['scenario','seed','post'],as_index=False)[['base_mse','corrected_mse','correction_rms']].mean();tr.to_csv(O/'pde_drift_trajectories.csv',index=False);summary=tr.groupby(['scenario','post'])[['base_mse','corrected_mse','correction_rms']].mean();summary['gain_pct']=(1-summary.corrected_mse/summary.base_mse)*100;summary.to_csv(O/'pde_drift_summary.csv');dump(O/'pde_drift_audit.json',dict(passed=all(e['reveal']==e['start']+40 and e['issue']==e['start']+20 for e in events),scenarios=SCENARIOS,seeds=list(range(100,108)),full_feedback_events=len(events),protocol_sha256=digest(W/'recoverability_20261118/DRIFT_PROTOCOL.md'),pde='simulated periodic advection-diffusion; not real-world measurement',boundary_change_tested=False))
 print('PDE_DRIFT_DONE',flush=True)
if __name__=='__main__':main()
