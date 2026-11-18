"""Post-hoc best/worst test examples for failure inspection only."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
W=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(W/'recoverability_20261118'))
import evaluation_recoverability as ev
ctl=ev.ctl;local=ev.local;O=ev.OUT

def main():
 ctl.setup();d=ev.get_data('test');rows=pd.read_csv(O/'windows.csv');rows=rows[(rows.split=='test')&(rows.seed==42)].reset_index(drop=True);indices=[int(rows.warp_realized_gain.idxmax()),int(rows.warp_realized_gain.idxmin())];ck=ctl.load(O/'warp_42.pt','cuda');h=local.Head('warp').cuda().eval();h.load_state_dict(ck['model']);n=ctl.st.norm();fig,axes=plt.subplots(2,5,figsize=(15,6));records=[]
 for row,i in enumerate(indices):
  p=d['p'][i:i+1];y=d['y'][i:i+1];f=d['f'][i:i+1]
  with torch.no_grad():q=h(ctl.project_feature(f,ck['projection']),p,ck['gain']);a=[(t*n['std']+n['mean'])[0,0,9].cpu().numpy() for t in (y,p,q)]
  errp=a[1]-a[0];errq=a[2]-a[0];lim=max(np.max(np.abs(errp)),np.max(np.abs(errq)));lo=min(v.min() for v in a);hi=max(v.max() for v in a)
  for j,v in enumerate(a):axes[row,j].imshow(v,cmap='viridis',vmin=lo,vmax=hi);axes[row,j].set_title(['Truth','P','Warp'][j])
  for j,v in enumerate([errp,errq],3):axes[row,j].imshow(v,cmap='coolwarm',vmin=-lim,vmax=lim);axes[row,j].set_title(['P error','Warp error'][j-3])
  for ax in axes[row]:ax.axis('off')
  record=rows.iloc[i];axes[row,0].set_ylabel('best' if row==0 else 'worst');records.append(dict(rank='best' if row==0 else 'worst',file=record['file'],start=int(record['start']),lead=10,base_mse=float(record['base_mse']),warp_mse=float(record['warp_mse']),realized_gain=float(record['warp_realized_gain'])))
 fig.suptitle('Post-hoc test examples: same colour scale within each row (u, lead 10)',fontsize=13);fig.tight_layout();fig.savefig(O/'case_examples.png',dpi=160);fig.savefig(O/'case_examples.pdf');pd.DataFrame(records).to_csv(O/'case_examples.csv',index=False);print('CASES',records,flush=True)
if __name__=='__main__':main()
