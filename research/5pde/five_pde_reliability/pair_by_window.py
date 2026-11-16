"""Window-resolved 95% bootstrap intervals over independent trajectory IDs."""
import argparse,csv
import numpy as np
import torch
from common import OUT,Scenario
from evaluate import ARMS,NOISE_ARMS,MODES
import importlib.util
from pathlib import Path
_spec=importlib.util.spec_from_file_location('five_pde_summary', Path(__file__).with_name('summarize.py'))
_summary=importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_summary)
DEST,writer=_summary.DEST,_summary.writer
SEEDS=(42,123,2026)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('scenario');args=parser.parse_args();name=args.scenario
    scenario=Scenario(name,device='cpu');data={};starts=None
    for arm,seed in [(arm,seed) for seed in SEEDS for arm in ARMS]+[(arm,500000) for arm in NOISE_ARMS]:
            p=torch.load(OUT/'evaluated'/name/f'{arm}_{seed}.pt',map_location='cpu',weights_only=False)
            for m,mode in enumerate(MODES):
                items=[r for r in p['windows'] if r['mode']==m]
                if starts is None:starts=[r['start'] for r in items]
                assert starts==[r['start'] for r in items]
                data[(arm,seed,mode)]=np.stack([r['values'] for r in items]).astype(np.float64)
    n=data[('X',42,'raw')].shape[1];draw=np.random.default_rng(20261115).integers(0,n,size=(10000,n))
    path=DEST/name/'paired_by_window.csv'
    fields=['calibration','seed','channel','baseline','comparison','metric','window_start','mean_delta','ci95_low','ci95_high','trajectory_win_rate','trajectories','bootstrap_repetitions']
    h,w=writer(path,fields)
    for mode in MODES:
        for seed in (*SEEDS,'pooled'):
            for channel,index in [(c,i) for i,c in enumerate(scenario.info['channels'])]+[('all',slice(0,scenario.channels))]:
                for left,right in (('X','XP'),('XP','XPF'),('X','XPF'),('XPF','8-noise'),('XPF','32-noise'),('XPF','256-noise')):
                    for metric,k in (('crps',0),('interval_score90',3)):
                        def values(arm):
                            choices=(500000,) if arm in NOISE_ARMS else (SEEDS if seed=='pooled' else (seed,))
                            v=np.stack([data[(arm,s,mode)][:,:,index,k] for s in choices]).mean(0)
                            return v.mean(-1) if channel=='all' else v
                        delta=values(left)-values(right)
                        for start in range(0,len(starts),8):
                            chunk=delta[start:start+8];boots=chunk[:,draw].mean(-1)
                            low=np.quantile(boots,.025,axis=1);high=np.quantile(boots,.975,axis=1)
                            for i,row in enumerate(chunk):
                                w.writerow(dict(calibration=mode,seed=seed,channel=channel,baseline=left,comparison=right,
                                    metric=metric,window_start=starts[start+i],mean_delta=float(row.mean()),ci95_low=float(low[i]),
                                    ci95_high=float(high[i]),trajectory_win_rate=float((row>0).mean()),trajectories=n,bootstrap_repetitions=10000))
    h.close();print('PAIRED_WINDOWS_COMPLETE',name,flush=True)

if __name__=='__main__':main()
