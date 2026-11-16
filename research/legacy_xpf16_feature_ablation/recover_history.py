"""Recover real 49-grid and 2048-noise per-trajectory/window scores."""
import csv,json,sys,hashlib
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from clean_reliability_common import manifest
from experiment_reliability_transport_e12 import normalization
OUT=ROOT/'runs/legacy_xpf16_feature_ablation_20261116'
OLD49=ROOT/'runs/xpf16_fullpercent_20261111/evaluation/test'
OLD250=ROOT/'runs/xpf16_02grid_2048_20261112/evaluation/test'

def main():
    OUT.mkdir(parents=True,exist_ok=True);records=manifest()['files']['test'];assert len(records)==12
    for archived in ('xpf16_independent49/test_run.json', 'xpf16_02grid_2048/test_protocol_and_timing.json'):
        metadata=json.loads((ROOT/'github/DiPred/results/reliability'/archived).read_text())
        assert [r['file'] for r in metadata['records']]==[r['file'] for r in records], archived
    std=normalization()['std'].detach().cpu().numpy().reshape(2)
    raw=[];online=[];rows=[];grid250=[]
    for start in range(20,821,20):
        f=OLD49/f'{start:04d}.npz';g=OLD250/f'{start:04d}.npz';assert f.exists() and g.exists()
        x=np.load(f);y=np.load(g)
        for method in ('XPF','8-noise','32-noise','256-noise'):
            for mode in ('raw','online'):
                kinds=('quantile_rearranged',) if method=='XPF' else ('empirical','fair')
                for kind in kinds:
                    key=f'{mode}_rearranged_crps' if method=='XPF' else f'{mode}_{method}_{kind}'
                    v=x[key];assert v.shape==(12,2) and np.isfinite(v).all()
                    if method=='XPF' and kind=='quantile_rearranged':(online if mode=='online' else raw).append(float(v.mean()))
                    for t,r in enumerate(records):
                        for c,ch in enumerate(('u','v')):
                            rows.append(dict(protocol='49',method=method,mode=mode,estimator=kind,start=start,window_index=start//20+1,trajectory_id=r['file'],Re=r['re'],channel=ch,standardized_crps=float(v[t,c]),physical_crps=float(v[t,c]*std[c])))
        for method,prefix in (('XPF','xpf'),('2048-noise','noise')):
            for mode,estimator,key in ([('raw','quantile_unrearranged','xpf_raw_crps'),('online','quantile_rearranged','xpf_online_rearranged_crps')] if method=='XPF' else [('raw','empirical','noise_raw_empirical'),('raw','fair','noise_raw_fair'),('online','empirical','noise_proxy_empirical'),('online','fair','noise_proxy_fair')]):
                v=y[key];assert v.shape==(12,2,20) and np.isfinite(v).all()
                for t,r in enumerate(records):
                    for c,ch in enumerate(('u','v')):
                        grid250.append(dict(protocol='250',method=method,mode=mode,estimator=estimator,start=start,window_index=start//20+1,trajectory_id=r['file'],Re=r['re'],channel=ch,standardized_crps=float(v[t,c].mean()),physical_crps=float(v[t,c].mean()*std[c])))
    for name,data in (('historical_49_windows.csv',rows),('historical_250_windows.csv',grid250)):
        with (OUT/name).open('w',newline='') as f:
            w=csv.DictWriter(f,list(data[0]));w.writeheader();w.writerows(data)
    timing=[]
    a=json.loads((ROOT/'github/DiPred/results/reliability/xpf16_independent49/test_run.json').read_text())
    b=json.loads((ROOT/'github/DiPred/results/reliability/xpf16_02grid_2048/test_protocol_and_timing.json').read_text())
    for row in a['timings']:
        for count in ('8','32','256'):
            timing.append(dict(method=count+'-noise',protocol='49',start=row['start'],batch_size=12,
                sampling_seconds=row['noise_sample_seconds'][count],point_seconds=row['point_seconds'],
                note='historical cumulative sampling only; excludes calibration and scoring'))
    for row in b['timings']:
        timing.append(dict(method='2048-noise',protocol='250',start=row['start'],batch_size=12,
            sampling_seconds=row['sampling']['2048'],point_seconds='',
            note='historical cumulative sampling only; excludes calibration and scoring'))
    with (OUT/'historical_noise_timing.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,list(timing[0]));w.writeheader();w.writerows(timing)
    # Compare exact historical scalar anchors without modifying an evaluator.
    report=json.loads((ROOT/'github/DiPred/results/reliability/xpf16_independent49/test_distribution_windows.json').read_text())
    historic=np.array([v['online_rearranged_crps'] for v in report]);actual=np.array(online)
    assert np.allclose(historic,actual,rtol=2e-6,atol=2e-7)
    results=dict(windows=41,trajectories=12,history49_raw_mean=float(np.mean(raw)),history49_online_mean=float(np.mean(online)),
        online_json_max_abs_diff=float(np.max(np.abs(historic-actual))),
        noise_online_empirical_means={method:float(np.mean([row['standardized_crps'] for row in rows if row['method']==method and row['mode']=='online' and row['estimator']=='empirical'])) for method in ('8-noise','32-noise','256-noise')},
        trajectory_order_matches_both_published_protocols=True,protocol250_note='Actual historical per-trajectory, per-lead NPZ; original report weights Re groups, while 49 primary window curves weight trajectories.',
        estimator_note='Noise is the actual stored empirical ensemble CRPS; XPF is quantile-integral CRPS; 2048 uses 250-grid proxy-scale protocol.',
        limitation='Old 49-head noise CRPS is saved per trajectory/channel/window, not by individual lead; do not invent lead values.')
    (OUT/'historical_recovery.json').write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(results),flush=True)
if __name__=='__main__':main()
