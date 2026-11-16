"""Export complete per-window and 20-lead tables with reproducible paired CIs."""
import argparse,csv,hashlib,json,shutil,sys,os
from pathlib import Path
import numpy as np
import torch
from common import ROOT,OUT,Scenario,records
from evaluate import ARMS,NOISE_ARMS,MODES,LEVELS

DEST=ROOT.parents[1]/'results/reliability/5pde'
FIELDS=('crps','coverage90','width90','interval_score90','adjacent_crossing','any_crossing')
SEEDS=(42,123,2026)


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()


def writer(path,cols):
    path.parent.mkdir(parents=True,exist_ok=True);handle=path.open('w',newline='')
    w=csv.DictWriter(handle,fieldnames=cols);w.writeheader();return handle,w


def main():
    parser=argparse.ArgumentParser();parser.add_argument('scenario');args=parser.parse_args();name=args.scenario
    sc=Scenario(name,device='cpu');dest=DEST/name;dest.mkdir(parents=True,exist_ok=True)
    experiments=[(arm,seed) for seed in SEEDS for arm in ARMS]+[(arm,500000) for arm in NOISE_ARMS]
    parts=[torch.load(OUT/'evaluated'/name/f'{arm}_{seed}.pt',map_location='cpu',weights_only=False) for arm,seed in experiments]
    bench=json.loads((OUT/'benchmarks'/f'{name}.json').read_text())
    backbone_parameters=torch.load(OUT/'backbones'/name/'model.pt',map_location='cpu',weights_only=False)['parameters']
    times={(v['arm'],v['seed'],v['calibration']):v['seconds'] for v in bench['results']}
    channels=sc.info['channels'];arrays={};summary=[];snapshots=[];cdf={c:[] for c in channels}
    qh,qw=writer(dest/'quantile_metrics.csv',['method','seed','calibration','channel','quantile','empirical_coverage','coverage_error','pinball'])
    for p in parts:
        arm,seed=p['arm'],p['seed'];rs=p['records'];run=dest/'runs'/f'{arm}_{seed}'
        common=['method','seed','calibration','trajectory_id','official_group','parameters','Re','window_start','channel']
        wh,ww=writer(run/'per_window.csv',common+list(FIELDS)+['pair_projection_rate'])
        fh,fw=writer(run/'per_frame.csv',common+[f'crps_frame_{i:02d}' for i in range(1,21)])
        for m,mode in enumerate(MODES):
            windows=[v for v in p['windows'] if v['mode']==m];frames=[v for v in p['frames'] if v['mode']==m]
            assert len(windows)==len(frames)>0
            a=np.stack([v['values'] for v in windows]);arrays[(arm,seed,mode)]=a
            for w,fr in zip(windows,frames):
                assert w['start']==fr['start']
                for j,r in enumerate(rs):
                    for c,label in enumerate(channels):
                        base=dict(method=arm,seed=seed,calibration=mode,trajectory_id=r['id'],official_group=r['official_group'],
                            parameters=json.dumps(r['parameters'],separators=(',',':')),Re=r.get('Re'),window_start=w['start'],channel=label)
                        row=dict(base);row.update({k:float(v) for k,v in zip(FIELDS,w['values'][j,c])});row['pair_projection_rate']=w['projected'];ww.writerow(row)
                        row=dict(base);row.update({f'crps_frame_{lead+1:02d}':format(float(fr['values'][j,c,lead]),'.9g') for lead in range(20)});fw.writerow(row)
            for c,label in enumerate(channels):
                for i,tau in enumerate(LEVELS):
                    observed=float(p['qcoverage'][m,i,c]);qw.writerow(dict(method=arm,seed=seed,calibration=mode,
                        channel=label,quantile=float(tau),empirical_coverage=observed,coverage_error=observed-float(tau),pinball=float(p['qpinball'][m,i,c])))
                cdf[label].append((f'{arm}_{seed}_{mode}',p['cdf'][m,c].numpy()))
            for label,index in [(c,i) for i,c in enumerate(channels)]+[('all',slice(0,sc.channels))]:
                row=dict(method=arm,seed=seed,calibration=mode,channel=label,head_parameters=p['parameters'],
                    training_group_seconds=p['train_seconds_group'],inference_seconds=times[(arm,seed,mode)],
                    backbone_parameters=0 if arm=='X' else backbone_parameters,
                    total_parameters=p['parameters']+(0 if arm=='X' else backbone_parameters))
                row.update({key:float(np.mean(a[:,:,index,i],dtype=np.float64)) for i,key in enumerate(FIELDS)});summary.append(row)
        wh.close();fh.close();snapshots.extend(dict(method=arm,seed=seed,**v) for v in p['snapshots'])
    qh.close()
    for kind in ('per_window','per_frame'):
        ih,iw=writer(dest/f'{kind}_index.csv',['method','seed','path'])
        for arm,seed in experiments:iw.writerow(dict(method=arm,seed=seed,path=f'runs/{arm}_{seed}/{kind}.csv'))
        ih.close()
    sh,sw=writer(dest/'summary.csv',list(summary[0]));sw.writerows(summary);sh.close()
    for channel in channels:
        with (dest/f'cdf_{channel}.csv').open('w',newline='') as handle:
            w=csv.writer(handle);w.writerow(['x']+[key for key,_ in cdf[channel]])
            for i,value in enumerate(parts[0]['grid'].numpy()):w.writerow([float(value)]+[float(a[i]) for _,a in cdf[channel]])
    ph,pw=writer(dest/'paired_comparison.csv',['calibration','seed','channel','baseline','comparison','metric','mean_delta','ci95_low','ci95_high','trajectory_win_rate','trajectories','bootstrap_repetitions'])
    rng=np.random.default_rng(20261114);n=len(parts[0]['records'])
    for mode in MODES:
        for seed in (*SEEDS,'pooled'):
            for label,index in [(c,i) for i,c in enumerate(channels)]+[('all',slice(0,sc.channels))]:
                for left,right in (('X','XP'),('XP','XPF'),('X','XPF'),('XPF','8-noise'),('XPF','32-noise'),('XPF','256-noise')):
                    for metric,k in (('crps',0),('interval_score90',3)):
                        def values(arm):
                            ids=(500000,) if arm in NOISE_ARMS else (SEEDS if seed=='pooled' else (seed,))
                            output=[]
                            for s in ids:
                                v=arrays[(arm,s,mode)][:,:,index,k]
                                output.append(v.mean(axis=(0,2) if v.ndim==3 else (0,)))
                            return np.stack(output).mean(0)
                        d=values(left)-values(right);assert d.shape==(n,)
                        draw=d[rng.integers(0,n,size=(10000,n))].mean(1)
                        pw.writerow(dict(calibration=mode,seed=seed,channel=label,baseline=left,comparison=right,metric=metric,
                            mean_delta=float(d.mean()),ci95_low=float(np.quantile(draw,.025)),ci95_high=float(np.quantile(draw,.975)),
                            trajectory_win_rate=float((d>0).mean()),trajectories=n,bootstrap_repetitions=10000))
    ph.close()
    np.savez_compressed(dest/'prediction_snapshot.npz',
        methods=np.array([f"{v['method']}_{v['seed']}_{MODES[v['mode']]}" for v in snapshots]),levels=LEVELS.numpy(),
        q=np.stack([v['q'].numpy() for v in snapshots]),y=np.stack([v['y'].numpy() for v in snapshots]),
        crps=np.stack([v['crps'].numpy() for v in snapshots]),normalization_std=np.array(sc.info['std']),
        trajectory=parts[0]['records'][0]['id'],window_start=20)
    weights={};(dest/'weights').mkdir(exist_ok=True)
    sources=[OUT/'backbones'/name/'model.pt']+[OUT/'heads'/name/f'seed_{s}'/(arm+'.pt') for s in SEEDS for arm in ARMS]
    for source in sources:
        label='backbone.pt' if source.parent.name==name else f'{source.stem}_{source.parent.name}.pt'
        target=dest/'weights'/label;shutil.copy2(source,target);weights[str(target.relative_to(dest))]=sha(target)
    (dest/'calibration').mkdir(exist_ok=True)
    for source in (OUT/'calibration'/name).glob('*.pt'):shutil.copy2(source,dest/'calibration'/source.name)
    shutil.copy2(OUT/'risk_data'/name/'complete.json',dest/'risk_training_windows.json')
    shutil.copy2(OUT/'validation'/f'{name}.json',dest/'validation.json')
    shutil.copy2(OUT/'preflight'/f'{name}.json',dest/'preflight.json')
    split=records(name);num_test=len(parts[0]['records']);window_count=len([v for v in parts[0]['windows'] if v['mode']==0])
    config=dict(scenario=name,repository_base_commit='d6f7a40c881370e43b2831370608facc9fca8cac',backbone_source_git_blobs={'reliability/networks.py':'2a622770f8bc3ac9f4f6062b1d193eaec1214150','reliability/backbone.py':'37c55c8a0d81ef36d66bb49093b6e221b746c58a'},domain='real only',source_revision='0ae4426085d250787f96d5456bc0d30ce5f277d2',
        source_split=split,Re_source='first filename token for fluid scenarios; not applicable to combustion',normalization=sc.info,history_frames=20,horizon=20,cold_start=0,
        evaluated_starts=[v['start'] for v in parts[0]['windows'] if v['mode']==0],test_trajectories=num_test,
        full_test_windows_per_trajectory=window_count,head_seeds=SEEDS,noise_seed=500000,noise_sampling='Independent RNG per path, four denoising calls; 8/32 are prefixes of 256',backbone_seed=42,steps=2000,batch_size=4,
        train_risk_windows=2048,head_input_slots='X:2, P:2, final denoising feature projection:8, total 12; absent slots masked',head_loss='quantile Huber kappa=.1, tau Uniform(0,1) clipped .002,.998',
        optimizer='AdamW, lr cosine 3e-4 to 1e-6, weight decay 1e-5, gradient clip 1',
        crps='499 levels .002 to .998, trapezoid 2*pinball, endpoint clamping',
        calibration='same fixed CQR and independent per-trajectory online two-tail offsets, gain .5',
        cdf='forecast-minus-truth; raw noise uses the native sample empirical CDF; all other distributions use piecewise-linear quantiles with endpoint atoms; calibrated noise is a quantile-calibrated distribution, not a calibrated ensemble',
        crossing='raw model monotone by construction; calibrated levels sorted before scoring; pre-sort crossing reported',
        caveat='RealPDEBench trajectories may have been used in earlier project development; do not claim fresh independent confirmation.')
    (dest/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    download=json.loads((Path(os.environ['FIVE_PDE_DATA'])/'download_manifest.json').read_text())
    source_files={item['rfilename']:item.get('lfs',{}).get('sha256') for item in download['files']}
    manifest=dict(weights_sha256=weights,observations_per_channel=parts[0]['observations'],gpu=bench['gpu'],benchmarks=bench['results'],
        source_shard_expected_sha256={r['shard']:source_files.get(r['shard']) for r in split},
        source_hash_note='Expected LFS SHA-256 values from supplied download manifest; this experiment checks sample field values, not a full shard rehash.')
    (dest/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('SUMMARY_COMPLETE',name,window_count,num_test,flush=True)

if __name__=='__main__':main()
