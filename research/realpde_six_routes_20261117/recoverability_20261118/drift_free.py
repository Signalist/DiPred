"""Warm-then-free continuation for controlled PDE dynamics; no feedback in rollout."""
from pathlib import Path
import sys
import pandas as pd
import torch
W=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(W/'recoverability_20261118'))
from drift_pde import simulate,SCENARIOS,GeometricPredictor,setup,O

def main():
 setup();rows=[]
 with torch.no_grad():
  for scenario in SCENARIOS:
   for seed in range(100,108):
    clean,obs=simulate(seed,scenario);p=GeometricPredictor('rolling',linear=True)
    for j in range(3):p.predict(obs[:,j*20:(j+1)*20][None],j*20,False);p.reveal(obs[:,(j+1)*20:(j+2)*20][None],j*20+40,False)
    h=obs[:,60:80][None].cuda();base=h.clone()
    for block in range(10):
     start=60+block*20;q=p.predict(h,start,False);p.skip_feedback();base_norm=(base-p.n['mean'])/p.n['std'];base=(p.predictor.rollout(base_norm,500000,sign=0)*p.n['std']+p.n['mean']).detach();target=clean[:,start+20:start+40][None].cuda();rows.append(dict(scenario=scenario,seed=seed,block=block+1,base_mse=float((base-target).square().mean()),corrected_mse=float((q-target).square().mean())));h=q.detach()
   print('PDE_FREE',scenario,flush=True)
 df=pd.DataFrame(rows);df.to_csv(O/'pde_drift_free_windows.csv',index=False);summary=df.groupby('scenario')[['base_mse','corrected_mse']].mean();summary['gain_pct']=(1-summary.corrected_mse/summary.base_mse)*100;summary.to_csv(O/'pde_drift_free_summary.csv')
if __name__=='__main__':main()
