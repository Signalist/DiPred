"""Write reproducible scalar tables and trajectory-unit paired bootstrap."""
import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT/'fair_x_xp_xpf'))
sys.path.insert(0,str(ROOT))
from evaluate import OUT, ARMS, SEEDS, LEVELS
from clean_reliability_common import manifest, digest
from experiment_reliability_transport_e12 import normalization, FIT_RECORDS, FIT_STARTS
from experiment_extended_train_error_adapters import LATE_RECORDS, LATE_STARTS

DEST=ROOT/'github/DiPred/results/reliability/fair_x_xp_xpf'
MODES=('raw','fixed_cqr','online')
FIELDS=('crps','coverage90','width90','interval_score90','adjacent_crossing','any_crossing')


def writer(path, columns):
    handle=path.open('w',newline=''); w=csv.DictWriter(handle,fieldnames=columns);w.writeheader()
    return handle,w


def mean(values):return float(np.mean(values,dtype=np.float64))


def main():
    assert (OUT/'evaluation_complete.json').exists() and (OUT/'validation_complete.txt').exists()
    DEST.mkdir(parents=True,exist_ok=True)
    parts=[torch.load(OUT/'evaluated'/f'{arm}_{seed}.pt',map_location='cpu',weights_only=False)
           for seed in SEEDS for arm in ARMS]
    benchmark={(r['method'],r['seed']):r for r in json.loads((OUT/'benchmark.json').read_text())}
    wh,ww=writer(DEST/'per_window.csv',['method','seed','calibration','trajectory_id','Re','window_start','channel',*FIELDS,'pair_projection_rate'])
    fh,fw=writer(DEST/'per_frame.csv',['method','seed','calibration','trajectory_id','Re','window_start','channel','future_frame','crps'])
    qh,qw=writer(DEST/'quantile_metrics.csv',['method','seed','calibration','channel','quantile','empirical_coverage','coverage_error','pinball'])
    arrays={};snapshots=[];summary=[]
    cdf_tables={c:[] for c in ('u','v')}
    for part in parts:
        arm,seed=part['arm'],part['seed']; records=part['records']; bench=benchmark[(arm,seed)]
        for m,mode in enumerate(MODES):
            rows=[v for v in part['windows'] if v['mode']==m]
            frame=[v for v in part['frames'] if v['mode']==m]
            assert len(rows)==len(frame)==41
            a=np.stack([v['values'] for v in rows])
            arrays[(arm,seed,mode)]=a
            for w,fr in zip(rows,frame):
                assert w['start']==fr['start']
                for t,r in enumerate(records):
                    for channel,c in enumerate(('u','v')):
                        common=dict(method=arm,seed=seed,calibration=mode,trajectory_id=r['file'],Re=r['re'],window_start=w['start'],channel=c)
                        row=dict(common);row.update({k:float(v) for k,v in zip(FIELDS,w['values'][t,channel])});row['pair_projection_rate']=w['projected'];ww.writerow(row)
                        for lead in range(20):
                            fw.writerow(dict(common,future_frame=lead+1,crps=float(fr['values'][t,channel,lead])))
            for channel,c in enumerate(('u','v')):
                for k,tau in enumerate(LEVELS):
                    observed=float(part['qcoverage'][m,k,channel]);loss=float(part['qpinball'][m,k,channel])
                    qw.writerow(dict(method=arm,seed=seed,calibration=mode,channel=c,quantile=float(tau),empirical_coverage=observed,coverage_error=observed-float(tau),pinball=loss))
                cdf_tables[c].append((f'{arm}_{seed}_{mode}',part['cdf'][m,channel].numpy()))
            for c,index in (('u',0),('v',1),('both',slice(None))):
                entry=dict(method=arm,seed=seed,calibration=mode,channel=c,
                           training_group_seconds=part['train_seconds_group'],parameters=part['parameters'],
                           inference_seconds_per_trajectory=bench['seconds'])
                entry.update({name:mean(a[:,:,index,i]) for i,name in enumerate(FIELDS)})
                summary.append(entry)
        snapshots.extend(dict(method=arm,seed=seed,**v) for v in part['snapshots'])
    wh.close();fh.close();qh.close()
    sh,sw=writer(DEST/'summary.csv',list(summary[0]));sw.writerows(summary);sh.close()
    for c in ('u','v'):
        grid=parts[0]['grid'].numpy();path=DEST/f'cdf_{c}.csv'
        with path.open('w',newline='') as f:
            w=csv.writer(f);w.writerow(['x']+[label for label,_ in cdf_tables[c]])
            for i,x in enumerate(grid):w.writerow([float(x)]+[float(a[i]) for _,a in cdf_tables[c]])
    # Resample trajectories, never treat windows or seeds as independent units.
    ph,pw=writer(DEST/'paired_comparison.csv',['calibration','seed','channel','baseline','comparison','metric','mean_delta','ci95_low','ci95_high','trajectory_win_rate','trajectories','bootstrap_repetitions'])
    rng=np.random.default_rng(20261113)
    for mode in MODES:
        for seed in (*SEEDS,'pooled'):
            for c,index in (('u',0),('v',1),('both',slice(None))):
                for left,right in (('X','XP'),('XP','XPF'),('X','XPF')):
                    for metric,k in (('crps',0),('interval_score90',3)):
                        def trajectory(arm):
                            seeds=SEEDS if seed=='pooled' else (seed,)
                            return np.stack([arrays[(arm,s,mode)][:,:,index,k].mean(axis=tuple(i for i in (0,2) if i<arrays[(arm,s,mode)][:,:,index,k].ndim)) for s in seeds]).mean(0)
                        a,b=trajectory(left),trajectory(right);d=a-b
                        assert d.shape==(12,)
                        boot=d[rng.integers(0,12,size=(10000,12))].mean(1)
                        pw.writerow(dict(calibration=mode,seed=seed,channel=c,baseline=left,comparison=right,metric=metric,mean_delta=mean(d),ci95_low=float(np.quantile(boot,.025)),ci95_high=float(np.quantile(boot,.975)),trajectory_win_rate=mean(d>0),trajectories=12,bootstrap_repetitions=10000))
    ph.close()
    # Raw Q/Y and per-point scores allow a genuinely independent numerical check.
    np.savez_compressed(DEST/'prediction_snapshot.npz',
        methods=np.array([f"{v['method']}_{v['seed']}_{MODES[v['mode']]}" for v in snapshots]),
        levels=LEVELS.numpy(),q=np.stack([v['q'].numpy() for v in snapshots]),
        y=np.stack([v['y'].numpy() for v in snapshots]),crps=np.stack([v['crps'].numpy() for v in snapshots]),
        normalization_std=normalization()['std'].cpu().numpy(),trajectory=parts[0]['records'][0]['file'],window_start=20)
    (DEST/'weights').mkdir(exist_ok=True)
    weights={}
    for seed in SEEDS:
        for arm in ARMS:
            source=OUT/f'seed_{seed}'/f'{arm}.pt';target=DEST/'weights'/f'{arm}_{seed}.pt'
            shutil.copy2(source,target);weights[str(target.relative_to(DEST))]=digest(target)
    shutil.copy2(OUT/'validation.csv',DEST/'validation.csv')
    shutil.copy2(OUT/'feature_audit.json',DEST/'feature_audit.json')
    shutil.copy2(OUT/'train_cache_audit.log',DEST/'train_cache_audit.log')
    shutil.copy2(OUT/'late_cache_audit.log',DEST/'late_cache_audit.log')
    code=DEST/'code';code.mkdir(exist_ok=True)
    for source in (ROOT/'fair_x_xp_xpf').glob('*.py'):shutil.copy2(source,code/source.name)
    config=dict(base_commit='ee69f8992df3b61c2082e385140dc57035207609',seeds=SEEDS,arms=ARMS,
        steps=2000,batch_size=4,optimizer='AdamW',learning_rate='cosine 3e-4 to 1e-6',weight_decay=1e-5,
        loss='quantile Huber kappa=0.1, tau uniform [0.002,0.998]',train_windows=819,
        input_slots='X(2), P(2), learned F 48->160->8x20; absent slots zeroed',
        backbone='frozen four-step DiPred, zero-noise point and final actual up2 hidden state',
        calibration='same independent per-trajectory/tail/channel online offset gain .5; pooled split CQR',
        splits=manifest()['files'],actual_fit_records=FIT_RECORDS,actual_fit_starts=FIT_STARTS,actual_late_records=LATE_RECORDS,actual_late_starts=LATE_STARTS,evaluation_starts=list(range(20,821,20)),cold_start=0,
        levels=[float(t) for t in LEVELS],crps='2x trapezoidal pinball integral; spacing .002; Q(0)=Q(.002), Q(1)=Q(.998)',
        normalization='train-only mean/std; scores and CDF in original data units',
        tests='reused exploratory test trajectories; no validated fresh holdout',
        crossing='monotone raw architecture; fixed/online reordered before distribution metrics; pre-reordering crossing reported',
        inference='single trajectory full 499-quantile synchronized GPU latency; XP/XPF include four DiPred calls; excludes feedback updates',
        bootstrap='10000 trajectory resamples; seed-pooled average within trajectory')
    (DEST/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    (DEST/'manifest.json').write_text(json.dumps(dict(weights_sha256=weights,backbone_sha256=digest(ROOT/'github/DiPred/reliability/checkpoints/current.pt'),records=len(parts),observations_per_channel=parts[0]['observations'],benchmark=list(benchmark.values())),indent=2)+'\n')
    print('SUMMARIZED',DEST,flush=True)

if __name__=='__main__':main()
