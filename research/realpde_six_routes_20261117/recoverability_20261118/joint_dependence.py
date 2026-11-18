"""Same-issue, fixed-marginal dependence comparison by reused Re groups."""
from pathlib import Path
import pandas as pd
W=Path(__file__).resolve().parent.parent;S=W/'runs/lowcost_joint_20261117/conditional_scores/scorefit';O=W/'runs/recoverability_20261118';parts=[]
for split in ['development','late','future']:
 d=pd.read_csv(S/split/'windows.csv');d['split']=split;parts.append(d)
d=pd.concat(parts,ignore_index=True);metrics=['es','vs','marginal_crps','functional_crps','crps_tke_peak','crps_kinetic_mean','crps_enstrophy_grid_peak','crps_speed_max','crps_mvpe','brier'];tr=d.groupby(['split','method','file','group'],as_index=False)[metrics].mean();tr.to_csv(O/'dependence_trajectories.csv',index=False);group=tr.groupby(['split','method','group'],as_index=False)[metrics].mean();group.to_csv(O/'dependence_by_re.csv',index=False);print('DEPENDENCE_BY_RE',len(group),flush=True)
