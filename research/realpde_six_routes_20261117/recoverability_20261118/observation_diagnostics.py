"""One-sensor target-assisted oracle diagnostic, fixed predictive assimilator."""
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import torch
W=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(W/'active_observation_20261117'))
from critic import *
O=W/'runs/recoverability_20261118'

def main():
 setup();assim=PredictiveAssimilator(json.loads((ROOT/'selection.json').read_text())['config']);rows=[]
 for split in ['development','late','future']:
  d=load(BASE/f'{split}.pt','cuda');truth=load(SOURCE/f'{split}.pt','cuda')['y'];req=load(ROOT/split/'requests.pt');critic=load(CRITIC/split/'requests.pt')
  for i,m in enumerate(d['meta']):
   pack=assim.pack(d['scale'][i]);r=(truth[i]-d['p'][i])*norm()['std'][0];base,dot=loss_stats(pack,r);obs=observation_rows(r).double();ix=torch.arange(128,device='cuda')[:,None]*2+torch.arange(2,device='cuda')[None];cov=pack['cov'][ix[:,:,None],ix[:,None,:]];h=pack['h'];z=assim.cfg['gain']*(h.transpose(1,2)@torch.linalg.solve(cov,obs[:,:,None]))[:,:,0];gain=(2*(z*dot).sum(-1)-((z@pack['g'])*z).sum(-1)).cpu().numpy();expected=single_gain(assim,pack).cpu().numpy();cscore=np.array(critic[i]['predicted_single_gains']);entry=dict(**m,split=split,base_mse=base,oracle=float(gain.max()),random=float(gain.mean()),oracle_index=int(gain.argmax()),oracle_negative=bool(gain.max()<0),gain_range=float(np.ptp(gain)),correlation_analytic=float(spearmanr(expected,gain).correlation),correlation_critic=float(spearmanr(cscore,gain).correlation))
   for method in ['risk','variance','voi','uniform']:
    ind=req[i]['actions'][method+'_1'][0];entry[method]=float(gain[ind]);entry[method+'_index']=int(ind)
   entry['critic']=float(gain[critic[i]['order'][0]]);rows.append(entry)
  print('VOI_ORACLE',split,len(d['meta']),flush=True)
 pd.DataFrame(rows).to_csv(O/'observation_single_oracle.csv',index=False)
if __name__=='__main__':main()
