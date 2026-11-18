"""Common raw/fixed/online procedure: marginal vs five-task coverage distinction."""
from pathlib import Path
import pandas as pd
W=Path(__file__).resolve().parent.parent;O=W/'runs/recoverability_20261118';p=pd.read_csv(W/'runs/online_signed_state_20261117/probability/all_trajectories.csv');keys=['split','mode','method'];cols=['es','functional_crps','coverage90','coverage90_u','coverage90_v','width90'];p.groupby(keys)[cols].mean().reset_index().to_csv(O/'common_calibration_summary.csv',index=False)
t=pd.read_csv(W/'runs/online_signed_state_20261117/temporal_intervals/summary.csv');t.to_csv(O/'task_interval_summary.csv',index=False);print('CALIBRATION_SUMMARY',len(p),len(t),flush=True)
