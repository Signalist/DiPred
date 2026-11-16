"""Export aligned full-test 0.2% CDFs and auditable per-window metrics."""
import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from clean_reliability_common import manifest
from experiment_reliability_transport_e12 import normalization
OUT=ROOT/'runs/fair_f_p_pf_xf_sharedref_20261116'
DEST=ROOT/'github/DiPred/results/reliability/fair_f_p_pf_xf_sharedref'
ARMS=('F','P','PF','XF');SEEDS=(42,123,2026);MODES=('raw','fixed_cqr','online')
FIELDS=('crps','coverage90','width90','interval_score90','adjacent_crossing','any_crossing')

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()

def main():
    assert (OUT/'evaluation_complete.json').exists()
    DEST.mkdir(parents=True,exist_ok=True)
    parts=[torch.load(OUT/'evaluated'/f'{arm}_{seed}.pt',map_location='cpu',weights_only=False) for seed in SEEDS for arm in ARMS]
    cdfs={'u':[],'v':[]};summary=[];snapshots=[];arrays={}
    with (DEST/'per_window.csv').open('w',newline='') as fw,(DEST/'per_frame.csv').open('w',newline='') as ff,(DEST/'quantile_metrics.csv').open('w',newline='') as fq:
        base=['method','seed','calibration','trajectory_id','Re','window_start','channel']
        ww=csv.DictWriter(fw,base+list(FIELDS));ww.writeheader()
        wf=csv.DictWriter(ff,base+[f'crps_frame_{i:02d}' for i in range(1,21)]);wf.writeheader()
        wq=csv.DictWriter(fq,['method','seed','calibration','channel','quantile','empirical_coverage','coverage_error','pinball']);wq.writeheader()
        for part in parts:
            arm,seed=part['arm'],part['seed'];rs=part['records']
            for mode_id,mode in enumerate(MODES):
                windows=[v for v in part['windows'] if v['mode']==mode_id]
                frames=[v for v in part['frames'] if v['mode']==mode_id]
                assert len(windows)==len(frames)==41
                a=np.stack([v['values'] for v in windows]);arrays[(arm,seed,mode)]=a
                for w,frame in zip(windows,frames):
                    assert w['start']==frame['start']
                    for j,r in enumerate(rs):
                        for c,label in enumerate(('u','v')):
                            common=dict(method=arm,seed=seed,calibration=mode,trajectory_id=r['file'],Re=r['re'],window_start=w['start'],channel=label)
                            ww.writerow(dict(common,**{k:float(v) for k,v in zip(FIELDS,w['values'][j,c])}))
                            wf.writerow(dict(common,**{f'crps_frame_{k+1:02d}':format(float(v),'.9g') for k,v in enumerate(frame['values'][j,c])}))
                for c,label in enumerate(('u','v')):
                    cdfs[label].append((f'{arm}_{seed}_{mode}',part['cdf'][mode_id,c].numpy()))
                    for i in range(499):
                        tau=(i+1)/500;obs=float(part['qcoverage'][mode_id,i,c]);loss=float(part['qpinball'][mode_id,i,c])
                        wq.writerow(dict(method=arm,seed=seed,calibration=mode,channel=label,quantile=tau,empirical_coverage=obs,coverage_error=obs-tau,pinball=loss))
                for label,index in (('u',0),('v',1),('both',slice(None))):
                    row=dict(method=arm,seed=seed,calibration=mode,channel=label,parameters=part['parameters'],training_group_seconds=part['train_seconds_group'])
                    row.update({key:float(np.mean(a[:,:,index,i],dtype=np.float64)) for i,key in enumerate(FIELDS)});summary.append(row)
            snapshots.extend(dict(method=arm,seed=seed,**v) for v in part['snapshots'])
    with (DEST/'summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,list(summary[0]));w.writeheader();w.writerows(summary)
    # Bring in the earlier, matched X/XP/XPF runs for true trajectory-paired comparisons.
    old_out=ROOT/'runs/fair_x_xp_xpf_20261113/evaluated'
    for seed in SEEDS:
        for arm in ('X','XP','XPF'):
            old=torch.load(old_out/f'{arm}_{seed}.pt',map_location='cpu',weights_only=False)
            assert [r['file'] for r in old['records']]==[r['file'] for r in parts[0]['records']]
            for mode_id,mode in enumerate(MODES):
                rows=[v for v in old['windows'] if v['mode']==mode_id]
                assert [v['start'] for v in rows]==list(range(20,821,20))
                arrays[(arm,seed,mode)]=np.stack([v['values'] for v in rows])
    compare=('X','F','P','XP','XF','PF','XPF')
    with (DEST/'paired_comparison.csv').open('w',newline='') as f:
        columns=['calibration','seed','channel','baseline','comparison','metric','mean_delta','ci95_low','ci95_high','trajectory_win_rate','trajectories','bootstrap_repetitions']
        w=csv.DictWriter(f,columns);w.writeheader();rng=np.random.default_rng(20261116)
        for mode in MODES:
            for seed in (*SEEDS,'pooled'):
                for label,index in (('u',0),('v',1),('both',slice(None))):
                    for li,left in enumerate(compare):
                        for right in compare[li+1:]:
                            for metric,k in (('crps',0),('interval_score90',3)):
                                def per_trajectory(arm):
                                    ids=SEEDS if seed=='pooled' else (seed,)
                                    values=[]
                                    for sd in ids:
                                        x=arrays[(arm,sd,mode)][:,:,index,k]
                                        values.append(x.mean(axis=(0,2) if x.ndim==3 else (0,)))
                                    return np.stack(values).mean(0)
                                delta=per_trajectory(left)-per_trajectory(right);assert delta.shape==(12,)
                                boot=delta[rng.integers(0,12,size=(10000,12))].mean(1)
                                w.writerow(dict(calibration=mode,seed=seed,channel=label,baseline=left,comparison=right,metric=metric,mean_delta=float(delta.mean()),ci95_low=float(np.quantile(boot,.025)),ci95_high=float(np.quantile(boot,.975)),trajectory_win_rate=float((delta>0).mean()),trajectories=12,bootstrap_repetitions=10000))
    for channel,entries in cdfs.items():
        with (DEST/f'cdf_{channel}.csv').open('w',newline='') as f:
            w=csv.writer(f);w.writerow(['x']+[label for label,_ in entries])
            for i,x in enumerate(parts[0]['grid'].numpy()):w.writerow([float(x)]+[float(v[i]) for _,v in entries])
    np.savez_compressed(DEST/'prediction_snapshot.npz',methods=np.array([f"{v['method']}_{v['seed']}_{MODES[v['mode']]}" for v in snapshots]),levels=np.arange(1,500)/500,q=np.stack([v['q'].numpy() for v in snapshots]),y=np.stack([v['y'].numpy() for v in snapshots]),crps=np.stack([v['crps'].numpy() for v in snapshots]),normalization_std=normalization()['std'].cpu().numpy())
    weights=DEST/'weights';weights.mkdir(exist_ok=True)
    for seed in SEEDS:
        for arm in ARMS:shutil.copy2(OUT/f'seed_{seed}'/f'{arm}.pt',weights/f'{arm}_{seed}.pt')
    shutil.copy2(OUT/'validation.csv',DEST/'validation.csv')
    code=DEST/'code';code.mkdir(exist_ok=True)
    for p in (ROOT/'fair_f_p_pf_xf_sharedref').glob('*.py'):shutil.copy2(p,code/p.name)
    shutil.copy2(ROOT/'fair_f_p_pf_xf_sharedref/run.sh',code/'run.sh')
    config=dict(methods=ARMS,seeds=SEEDS,steps=2000,training_windows=819,history=20,horizon=20,
        levels=list(np.arange(1,500)/500),cdf_grid='[-0.35,0.35],701 points, forecast minus truth in original data units',
        test_starts=list(range(20,821,20)),excluded_cold_start=0,trajectory_split=manifest()['files'],
        input_policy='F/P/PF: direct X slot zero; XF: X slot active; all arms share the observed last X frame as output reference. P and F are functions of X.',
        warning='Reused exploratory test; all arms require X last frame as reference, P/F are functions of X, and trajectory bootstrap may understate uncertainty under shared Reynolds conditions.',
        calibration='same independent per-trajectory online calibration with feedback only after 20 frames; pooled fixed CQR; spatial/temporal dependence precludes claiming trajectory-level conformal coverage',
        crps='trapezoidal 2x pinball at 0.002 levels with clamped tails')
    (DEST/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    (DEST/'README.md').write_text('# F / P / PF / XF full-test CDF\n\nCSV contains all seeds, raw, fixed-CQR and online calibrated 0.2% quantiles. See config.json for data split, grid and references. F/P/PF do not feed X into the head input, but all arms use X last frame as reference and DiPred-derived inputs depend on X. Test trajectories were previously used; comparisons are exploratory. Archived code assumes the original research workspace, cached inputs and source data; CSV/NPZ results can be read independently.\n')
    with (DEST/'README.md').open('a') as f:
        f.write('\nReplot without the original dataset: `import pandas as pd; import matplotlib.pyplot as plt; d=pd.read_csv("cdf_u_combined_online.csv"); d.plot(x="x"); plt.show()`.\n')
    files={str(p.relative_to(DEST)):sha(p) for p in DEST.rglob('*') if p.is_file() and p.name!='manifest.json'}
    (DEST/'manifest.json').write_text(json.dumps(dict(observations_per_channel=parts[0]['observations'],files_sha256=files),indent=2)+'\n')
    print('EXPORTED',DEST,flush=True)
if __name__=='__main__':main()
