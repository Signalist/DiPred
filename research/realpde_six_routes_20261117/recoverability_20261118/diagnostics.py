"""Exploratory diagnostics of risk, attainable gains, ranking and action transfer."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr,kendalltau
W=Path(__file__).resolve().parent.parent
O=W/'runs/recoverability_20261118'

def corr(x,y):
 x=np.asarray(x);y=np.asarray(y)
 if len(x)<3 or np.std(x)==0 or np.std(y)==0:return float('nan')
 return float(spearmanr(x,y).correlation)

def main():
 d=pd.read_csv(O/'windows.csv');rows=[];bins=[];groups=[];trans=[]
 for (split,seed),g in d.groupby(['split','seed']):
  base=g.base_mse.to_numpy();gain=g.warp_realized_gain.to_numpy();pred=g.benefit_prediction.to_numpy();risk=g.risk_prediction.to_numpy();n=len(g);order=np.argsort(-pred,kind='stable');k=int(np.floor(n*.5+.5));oracle=np.sort(gain)[-k:].sum()/n;selected=gain[order[:k]].sum()/n;random=.5*gain.mean();actual_res=g.rescrct_realized_gain.to_numpy()
  rows.append(dict(split=split,seed=seed,n=n,spearman_true_risk_realized_gain=corr(base,gain),spearman_predrisk_realized_gain=corr(risk,gain),spearman_predgain_realized_gain=corr(pred,gain),gain_rmse=float(np.sqrt(np.mean((pred-gain)**2))),gain_mae=float(np.mean(np.abs(pred-gain))),top50_recall=float(len(set(order[:k])&set(np.argsort(-gain)[:k]))/k),oracle_regret=float(oracle-selected),random_regret=float(oracle-random),action_transfer_spearman=corr(pred,actual_res),action_transfer_realized=float(actual_res[order[:k]].sum()/n),action_transfer_random=float(.5*actual_res.mean()),negative_action_fraction=float((gain<0).mean())))
  g=g.copy();g['risk_quintile']=pd.qcut(g.base_mse.rank(method='first'),5,labels=False);g['gain_quintile']=pd.qcut(g.warp_realized_gain.rank(method='first'),5,labels=False)
  for (r,b),a in g.groupby(['risk_quintile','gain_quintile']):bins.append(dict(split=split,seed=seed,risk_quintile=int(r)+1,gain_quintile=int(b)+1,n=len(a),risk_mse=float(a.base_mse.mean()),warp_gain=float(a.warp_realized_gain.mean()),negative_fraction=float((a.warp_realized_gain<0).mean())))
  for re,a in g.groupby('group'):
   groups.append(dict(split=split,seed=seed,re=re,windows=len(a),files=a.file.nunique(),base_mse=float(a.base_mse.mean()),warp_gain_pct=float((1-a.warp_mse.mean()/a.base_mse.mean())*100),rescrct_gain_pct=float((1-a.rescrct_mse.mean()/a.base_mse.mean())*100),risk_gain_spearman=corr(a.base_mse,a.warp_realized_gain)))
 pd.DataFrame(rows).to_csv(O/'risk_recoverability.csv',index=False);pd.DataFrame(bins).to_csv(O/'risk_gain_joint_bins.csv',index=False);pd.DataFrame(groups).to_csv(O/'re_robustness.csv',index=False)
 # Calibration is the only split with the restricted target-assisted local oracle.
 oracle=pd.read_csv(W/'runs/error_structure_20261117/calibration_local_oracle.csv');piv=oracle.pivot(index=['file','group','start'],columns='method',values='mse').reset_index();existing=pd.read_csv(W/'runs/error_structure_20261117/calibration_oracle.csv');b=existing[existing.method=='base'][['file','group','start','mse']].rename(columns={'mse':'base'});j=piv.merge(b,on=['file','group','start'],validate='one_to_one');learned=pd.read_csv(W/'runs/direct_correction_controls_20261117/calibration/windows.csv');learned=learned[learned.method=='warp_F_last'][['file','group','start','mse']].rename(columns={'mse':'learned'});j=j.merge(learned,on=['file','group','start'],validate='one_to_one');
 for method in ['geometry','amplitude','combined','learned']:j[method+'_gain']=j['base']-j[method]
 j.to_csv(O/'oracle_gap_windows.csv',index=False);print('DIAGNOSTICS',len(d),len(j),flush=True)
if __name__=='__main__':main()
