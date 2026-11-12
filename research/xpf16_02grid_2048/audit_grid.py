"""Audit 250 heads, all 499 probabilities, scores and causal online replay."""
import hashlib,json,subprocess,sys
import numpy as np
import torch
from clean_reliability_common import ROOT,digest,manifest
from train_grid import OUT,PREVIOUS,EXTREME,configs


def tensor_hash(state):
    h=hashlib.sha256()
    for key,value in sorted(state.items()):
        h.update(key.encode());h.update(value.contiguous().numpy().tobytes())
    return h.hexdigest()


def main():
    cfgs=configs()
    complete=json.loads((OUT/'training_complete.json').read_text())
    assert complete['passed'] and complete['total_heads']==250 and complete['reused']==50
    weights={};parameter_hashes={};reused=[]
    for c in cfgs:
        path=OUT/'models'/c['id']/'model.pt'
        loaded=torch.load(path,map_location='cpu',weights_only=False)
        assert loaded['config']==c and loaded['parameters']==34004 and loaded['training_pairs']==819
        assert loaded['loss'][-1]['step']==2000
        assert all(torch.isfinite(x).all() for x in loaded['model'].values())
        weights[c['id']]=digest(path);parameter_hashes[c['id']]=tensor_hash(loaded['model'])
        if c['k']==1 or c['k']%5==0 and c['k']<250:
            previous=(EXTREME/'models/XPF16-Q0p2/model.pt' if c['k']==1 else
                      PREVIOUS/'models'/f"XPF16-Q{c['k']//5:02d}"/'model.pt')
            assert loaded['source_checkpoint_sha256']==digest(previous)
            original=torch.load(previous,map_location='cpu',weights_only=False)
            assert all(torch.equal(loaded['model'][k],v) for k,v in original['model'].items())
            reused.append(c['id'])
    assert len(reused)==50 and len(set(parameter_hashes.values()))==250
    latency=json.loads((OUT/'latency.json').read_text())
    assert len(latency['heads'])==250 and all(v['seconds_per_batch']>0 for v in latency['heads'])
    portable=json.loads((OUT/'portable_audit.json').read_text())
    assert portable['passed']
    splits=manifest()['files']
    for a,b in (('train','calibration'),('train','development'),('train','test'),('calibration','development'),('calibration','test'),('development','test')):
        assert {x['file'] for x in splits[a]}.isdisjoint({x['file'] for x in splits[b]})
        assert {x['group'] for x in splits[a]}.isdisjoint({x['group'] for x in splits[b]})
    report_splits={}
    for split in ('development','test'):
        directory=OUT/'evaluation'/split
        meta=json.loads((directory/'complete.json').read_text())
        assert meta['samples']==2048 and meta['starts']==list(range(0,821,20))
        events=[json.loads(x) for x in (directory/'events.jsonl').read_text().splitlines()]
        assert len(events)==84
        for j,start in enumerate(meta['starts']):
            issue,reveal=events[2*j:2*j+2]
            assert issue['event']=='issued_all_499' and reveal['event']=='revealed_full20'
            assert issue['start']==reveal['start']==start and issue['time']<=reveal['time']
        maxerr=0.; max_rearranged=0.
        for start in meta['starts']:
            d=np.load(directory/f'{start:04d}.npz')
            iv=d['intervals'];m=d['marginals']
            assert iv.shape==(5,249,len(meta['records']),2,20,9)
            assert m.shape==(7,499,len(meta['records']),2,20,2)
            assert np.isfinite(iv).all() and np.isfinite(m).all()
            assert (iv[...,3]>=-1e-6).all()
            for k in range(249):
                alpha=(k+1)/500
                maxerr=max(maxerr,float(np.max(np.abs(iv[:,k,...,4]-(iv[:,k,...,5]+iv[:,k,...,6])/alpha))))
            # Empirical CDF of a nondecreasing quantile sequence is nondecreasing.
            for index in (2,3,4,6):
                max_rearranged=max(max_rearranged,float(np.maximum(-np.diff(m[index,...,0],axis=0),0).max()))
        assert maxerr<1e-3 and max_rearranged<1e-6,(maxerr,max_rearranged)
        report_splits[split]=dict(windows=42,score_identity_max_error=maxerr,
          sorted_cdf_max_violation=max_rearranged,xpf_projection=meta['xpf_projection'],
          noise_projection=meta['noise_projection'],projection_total=meta['projection_total'])
    snap=np.load(OUT/'evaluation/test/0000.npz')
    raw=snap['sample_snapshot'][:,0,0,:].astype(np.float64)
    point=snap['point_snapshot'][0,0,:].astype(np.float64)
    actual=snap['noise_snapshot'][:,0,0,:].astype(np.float64)
    probabilities=np.arange(1,500,dtype=np.float64)/500
    reconstructed=np.quantile(raw,probabilities,axis=0,method='linear')-point[None]
    qerror=float(np.max(np.abs(reconstructed-actual)))
    assert qerror<3e-6,qerror
    q=snap['xpf_rearranged_snapshot']
    assert np.diff(q,axis=0).min()>=-1e-7
    # The CDF product must cover every trajectory and every non-cold window,
    # and must use one shared physical-scale grid for all six methods.
    cdf_dir=OUT/'full_test_cdf'
    cdf_config=json.loads((cdf_dir/'config.json').read_text())
    methods=('xpf16_1','xpf16_02','noise8','noise32','noise256','noise2048')
    assert tuple(cdf_config['methods'])==methods
    assert cdf_config['starts']==list(range(20,821,20))
    count=len(manifest()['files']['test'])*41*20*32*64
    assert cdf_config['observations_per_channel']==count
    coordinates=[]
    for channel in ('u','v'):
        result=np.load(cdf_dir/f'cdf_{channel}.npz')
        csv=np.genfromtxt(cdf_dir/f'cdf_{channel}.csv',delimiter=',',names=True)
        assert result['count']==count and len(result['x'])==701
        assert np.allclose(result['x'],-result['x'][::-1],atol=1e-15)
        assert np.allclose(csv['x'],result['x']) and (csv['observations']==count).all()
        coordinates.append(result['x'])
        for method in methods:
            values=result[method]
            assert np.isfinite(values).all() and values.min()>=-1e-7 and values.max()<=1+1e-7
            assert (np.diff(values)>=-1e-7).all()
            assert np.allclose(csv[method],values,atol=1e-12)
    assert np.array_equal(*coordinates)
    for name in ('test_grid_calibration.py','test_distribution_grid.py'):
        subprocess.run([sys.executable,str(Path(__file__).with_name(name))],check=True)
    subprocess.run([sys.executable,str(ROOT/'xpf16_cdf_fulltest/test_cdf_stream.py')],check=True)
    out=dict(passed=True,heads=250,unique_parameter_sha256=parameter_hashes,checkpoint_sha256=weights,
       reused_prior_independent_heads=reused,independent_median_head='XPF16-P250',quantile_levels=499,
       samples=2048,snapshot_sample_quantile_max_error=qerror,splits=report_splits,portable_inference=portable,
       full_test_cdf=dict(observations_per_channel=count,starts=cdf_config['starts'],methods=methods),
       caveat='Historical development/test trajectories have been used in earlier research; no formal coverage guarantee.')
    (OUT/'independent_audit.json').write_text(json.dumps(out,indent=2)+'\n')
    print('GRID_AUDIT_PASSED',flush=True)

if __name__=='__main__':
    from pathlib import Path
    main()
