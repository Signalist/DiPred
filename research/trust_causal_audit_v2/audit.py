"""Read-only real-data preparation and small full-member replay checks.

This command does not generate missing native forecasts. It lists every missing
cache and returns READY=false instead. Test data receive metadata inspection
only, never model comparison or a score.
"""
import copy
import csv
from collections import Counter
import json
import time
import numpy as np
import torch
from .common import ROOT,OUT,OLD_CACHE,OLD_OUT,DATA,MEMBERS,digest,tensor_hash,dump,records,load_backbone,state_hash,code_hashes
from .data import Key,CacheIndex,H5Reader,window_catalog,raw_risk,fit_scale,prepare_inputs,model_inputs,batch_noise
from .model import TrustDiffusion,native_sample


def inspect_data():
    import h5py
    rs=records();file_rows=[];catalog={};lengths={};missing=[]
    archived_hashes=json.loads((ROOT/'results/native_restart_consistency/data_hashes.json').read_text())
    for split in ('train','development','test'):
        lens={}
        for index,r in enumerate(rs[split]):
            p=DATA/r['file']
            if not p.exists():
                missing.append(str(p));continue
            with h5py.File(p,'r') as f:
                n=f['u'].shape[0]
                assert f['v'].shape==f['u'].shape and f['u'].shape[1:]==(64,128)
                assert len(f['t'])==n
                keys=list(f.keys())
                # No documented obstacle/validity-mask field exists. Do not
                # invent masks from zero velocities or curvilinear coordinates.
                masks=[k for k in keys if any(word in k.lower() for word in ('mask','obstacle','valid'))]
                assert not masks,'Explicit geometry needs a documented adapter before READY'
            sha=digest(p)
            assert sha==archived_hashes[r['file']],r['file']
            lens[r['file']]=n
            file_rows.append(dict(split=split,trajectory_id=r['file'],original_index=index,
                 actual_frames=n,declared_frames=r['frames'],sha256=sha,
                 matches_archived_full_file_hash=True,geometry_fields=keys,explicit_masks=masks))
        lengths[split]=lens
        if len(lens)!=len(rs[split]): continue
        files=[r['file'] for r in rs[split]]
        catalog[split]=window_catalog(split,files,lens,(480,) if split=='test' else (420,440,460))
    for i,a in enumerate(('train','development','test')):
        for b in ('train','development','test')[:i]:
            assert not set(lengths[a]) & set(lengths[b])
            assert not {r['group'] for r in rs[a]} & {r['group'] for r in rs[b]}
    return rs,file_rows,lengths,catalog,missing


def load_cache(split,start,rs,lock):
    p=OLD_CACHE/f'{split}_{start}_native.pt'
    if not p.exists(): raise FileNotFoundError(str(p))
    files=[r['file'] for r in rs[split]]
    sha=digest(p)
    # Locked before all audit work; no in-place cache regeneration.
    before=json.loads((OUT/'PROTECTED_BEFORE.json').read_text())
    assert before[str(p.relative_to(ROOT))]==sha
    payload=torch.load(p,map_location='cpu',weights_only=False)
    idx=CacheIndex(payload,split,start,files,str(p.relative_to(ROOT)))
    lock[str(p.relative_to(ROOT))]=dict(sha256=sha,identity=payload['identity'],
       history_sha256=payload['history_sha256'],paths_sha256=tensor_hash(idx.paths),
       checkpoint_binding='old cache did not store checkpoint hash; small real replay audited separately')
    return idx,payload


def run():
    started=time.perf_counter()
    rs,files_info,lengths,catalog,missing_data=inspect_data()
    counts={s:{str(start):sum(row['key'].start==start for row in rows)
               for start in sorted({r['key'].start for r in rows})} for s,rows in catalog.items()}
    # These assertions are in addition to, not substitutes for, actual HDF5
    # shape inspection and independent per-file window eligibility above.
    if not missing_data:
        assert counts=={'train':{'420':46,'440':46,'460':45},'development':{'420':12,'440':12,'460':12},'test':{'480':12}}
    needed=[(split,start) for split in ('train','development') for start in (400,420,440)]
    missing_caches=[str(OLD_CACHE/f'{s}_{t}_native.pt') for s,t in needed if not (OLD_CACHE/f'{s}_{t}_native.pt').exists()]
    base=dict(passed=False,counts_from_actual_hdf5=counts,files=files_info,missing_data=missing_data,
              missing_native_caches=missing_caches,optimizer_updates=0,training_runs=0,
              new_native_cache_generation=0,formal_test_comparisons=0,
              test_scope='metadata/length/hash only; no test predictions, maps or scores',
              missing_cache_recovery_required=[dict(path=p,members=32,denoise_steps=4) for p in missing_caches])
    if missing_data or missing_caches:
        dump(OUT/'DATA_AUDIT.json',base);return base
    native,norm=load_backbone();base_hash=state_hash(native)
    cpu_norm={k:v.cpu() for k,v in norm.items()}
    reader=H5Reader(DATA);lock={};bykey={};raw_maps=[];entry_bykey={}
    # Cache processing is limited to the six already-existing prior shards.
    # Verify all original source histories and every selected member slice.
    for split,start in needed:
        idx,payload=load_cache(split,start,rs,lock)
        full_history=reader_batch=torch.stack([reader(r['file'],start,start+20) for r in rs[split]])
        assert tensor_hash(full_history)==payload['history_sha256'],(split,start)
        for row in catalog[split]:
            key=row['key']
            if key.start-20!=start: continue
            previous=idx.lookup(Key(split,key.trajectory_id,start))
            assert previous.cache_row==row['original_index']
            observed=reader(key.trajectory_id,key.start,key.start+20)
            # Manifest target hash is an audit label only, never an input.
            target=reader(key.trajectory_id,key.start+20,key.start+40)
            raw=raw_risk(observed,previous,cpu_norm['std'])
            explicit=torch.stack([((previous.paths.mean(0)[c]-observed[c])/cpu_norm['std'][c]).square()
                                  for c in range(2)]).sum((0,1))[None]/40
            torch.testing.assert_close(raw,explicit,rtol=1e-6,atol=1e-8)
            item=dict(split=split,trajectory_id=key.trajectory_id,start=key.start,
                original_index=row['original_index'],Re=rs[split][row['original_index']]['group'],
                actual_frames=lengths[split][key.trajectory_id],
                history_interval=[key.start,key.start+20],previous_history_interval=[start,start+20],
                previous_prediction_interval=list(previous.target_interval),target_interval=[key.start+20,key.start+40],
                history_sha256=tensor_hash(observed),target_sha256=tensor_hash(target),
                previous_path_sha256=tensor_hash(previous.paths),previous_cache=previous.source,
                previous_cache_row=previous.cache_row,previous_member_ids=list(MEMBERS),
                risk_source='previous_prediction_minus_observed_history',raw_risk_sha256=tensor_hash(raw))
            k=(split,key.trajectory_id,key.start)
            bykey[k]=raw
            entry_bykey[k]=item
            if split=='train':raw_maps.append(raw)
        del idx,payload,full_history,reader_batch
    scale=fit_scale(raw_maps,'train');assert scale['maps']==137
    dump(OUT/'RISK_NORMALIZATION.json',dict(**scale,formula='mean_c,h((P_prev_mean-H_current)/sigma_c)^2',
         checkpoint_sha256=digest(ROOT/'reliability/checkpoints/current.pt'),normalization={k:v.tolist() for k,v in cpu_norm.items()},
         train_map_keys=[list(k) for k in bykey if k[0]=='train']))
    manifest=[];arrays={}
    # A keyed NPZ index is explicit; values cannot shift when a window loses a
    # trajectory. Store only small maps, retaining issued P in read-only caches.
    for key,raw in bykey.items():
        risk=(torch.log1p(raw)/scale['log1p_p99']).clamp(0,1)
        token=f'{key[0]}__{key[1]}__{key[2]}'
        arrays[token+'__raw']=raw.numpy();arrays[token+'__risk']=risk.numpy()
        entry=dict(entry_bykey[key],map_key=token,risk_sha256=tensor_hash(risk))
        entry_bykey[key]=entry
        manifest.append(entry)
    np.savez_compressed(OUT/'risk_maps.npz',**arrays)
    dump(OUT/'WINDOW_MANIFEST.json',manifest)
    dump(OUT/'TEST_CATALOG.json',[dict(split=r['key'].split,trajectory_id=r['key'].trajectory_id,start=r['key'].start,
         original_index=r['original_index'],actual_frames=r['frames'],history_interval=[480,500],
         previous_prediction_interval=[480,500],target_interval=[500,520],audit_scope='metadata_only') for r in catalog['test']])
    dump(OUT/'CACHE_MANIFEST.json',lock)

    # Quantify the old failures by inspecting the saved rows against HDF5, not
    # by rerunning old training/inference. Their scores remain untouched.
    old_rows_audit=[]
    for split in ('train','development'):
        old=torch.load(OLD_CACHE/f'{split}_windows.pt',map_location='cpu',weights_only=False)['rows']
        expected_keys={(r['key'].split,r['key'].trajectory_id,r['key'].start) for r in catalog[split]}
        present=set()
        for row in old:
            k=(split,row['file'],row['start']);present.add(k)
            expected=entry_bykey[k]
            first_position=next(i for i,v in enumerate(old) if v['file']==row['file'])
            old_rows_audit.append(dict(split=split,trajectory_id=row['file'],start=row['start'],
                saved_history_matches_named_hdf5=tensor_hash(row['history'])==expected['history_sha256'],
                saved_target_matches_named_hdf5=tensor_hash(row['truth'])==expected['target_sha256'],
                legacy_previous_lookup_row=first_position,correct_previous_cache_row=expected['previous_cache_row'],
                legacy_previous_lookup_correct=first_position==expected['previous_cache_row']))
        base.setdefault('old_missing_records',{})[split]=[list(k) for k in sorted(expected_keys-present)]
    dump(OUT/'OLD_ROW_IDENTITY_AUDIT.json',old_rows_audit)
    base['old_rows_with_wrong_history']=[r for r in old_rows_audit if not r['saved_history_matches_named_hdf5']]
    base['old_rows_with_wrong_target']=[r for r in old_rows_audit if not r['saved_target_matches_named_hdf5']]
    base['old_rows_with_wrong_previous_lookup']=[r for r in old_rows_audit if not r['legacy_previous_lookup_correct']]
    base['old_risk_field_reused']=False

    # Four real full32 releases (two train before/after the shortened row,
    # the short trajectory at its last valid start, and one development).
    # Cache parity checks are not scores and never load test paths/truth.
    examples=[('train',rs['train'][0]['file'],420),('train','21600_5.h5',440),
              ('train',rs['train'][34]['file'],460),('development',rs['development'][-1]['file'],460)]
    replay=[];all_noise=[];gpu_seconds=0.
    for split,name,start in examples:
        key=Key(split,name,start);entry=entry_bykey[(split,name,start)]
        idx,_=load_cache(split,start-20,rs,lock)
        previous=idx.lookup(Key(split,name,start-20))
        local_reader=H5Reader(DATA)
        prepared=prepare_inputs(local_reader,key,previous,cpu_norm,scale)
        assert local_reader.reads==[dict(trajectory_id=name,interval=[start,start+20])]
        assert tensor_hash(prepared['raw_risk'])==entry['raw_risk_sha256']
        assert tensor_hash(prepared['risk'])==entry['risk_sha256']
        noise,identities=batch_noise(key,[r['file'] for r in rs[split]])
        again,identities_again=batch_noise(key,[r['file'] for r in rs[split]])
        assert torch.equal(noise,again) and identities==identities_again
        inputs=model_inputs(prepared,noise)
        # Reproduce the old per-trajectory draw solely to document mismatching
        # actual noise; these paths are never used for model evaluation.
        wrong,_=batch_noise(key,[name])
        noise_mismatches=sum(not torch.equal(noise[i],wrong[i]) for i in range(32))
        wrapper=TrustDiffusion(copy.deepcopy(native),True).eval()
        torch.cuda.synchronize();t=time.perf_counter()
        with torch.no_grad():
            baseline=native_sample(native,inputs['history'],noise)
            closed=wrapper.sample(**inputs,enabled=False)
        torch.cuda.synchronize();seconds=time.perf_counter()-t;gpu_seconds+=seconds
        assert torch.equal(baseline,closed)
        current_idx,_=load_cache(split,start,rs,lock)
        saved=current_idx.lookup(key).paths.cuda()
        physical=baseline*norm['std']+norm['mean']
        maxerr=float((physical-saved).abs().max())
        # Preserve the failed FP32 comparison; DO NOT widen its tolerance.
        # Diagnose the historical engine on one member of the ORIGINAL batch.
        full_history=torch.stack([local_reader(r['file'],start,start+20) for r in rs[split]]).cuda()
        hfull=(full_history-norm['mean'])/norm['std']
        full_noise=torch.randn(hfull.shape,device='cuda',generator=torch.Generator(device='cuda').manual_seed(MEMBERS[0]))
        original_index=entry['original_index']
        assert torch.equal(full_noise[original_index],noise[0])
        old_tf32,old_deterministic=torch.backends.cudnn.allow_tf32,torch.backends.cudnn.deterministic
        try:
            torch.backends.cudnn.allow_tf32=True
            torch.backends.cudnn.deterministic=False
            with torch.no_grad():
                historical=native_sample(native,hfull,full_noise)*norm['std']+norm['mean']
            engine_error=float((historical[original_index]-saved[0]).abs().max())
        finally:
            torch.backends.cudnn.allow_tf32=old_tf32
            torch.backends.cudnn.deterministic=old_deterministic
        assert engine_error<=2e-6,(key,engine_error)
        replay.append(dict(split=split,trajectory_id=name,start=start,
            original_index=entry['original_index'],members=32,leads=20,denoising_labels=[99,66,33,0],
            publication_reader_requests_only_observed_history=True,
            model_input_reads=[dict(trajectory_id=name,interval=[start,start+20])],
            disabled_wrapper_maxabs=float((baseline-closed).abs().max()),
            old_native_cache_FP32_physical_maxabs=maxerr,physical_tolerance=2e-6,
            old_cache_is_strict_FP32_output_baseline=maxerr<=2e-6,
            old_cache_original_tf32_batch_member0_error=engine_error,
            strict_noise_pairing=True,new_FP32_baseline_matches_closed_wrapper=True,
            old_single_trajectory_noise_member_mismatches=noise_mismatches,
            output_sha256=tensor_hash(physical),seconds=seconds))
        all_noise.extend(identities)
        del current_idx,idx,wrapper,previous
    dump(OUT/'REAL_REPLAY.json',replay);dump(OUT/'NOISE_IDENTITIES.json',all_noise)
    dump(OUT/'CACHE_MANIFEST.json',lock)
    assert state_hash(native)==base_hash
    base.update(passed=True,train_windows=137,development_windows=36,test_catalog_windows=12,
        raw_maps=173,risks_independently_recomputed=True,scale_source='137 valid training windows only',
        key_alignment='split, trajectory_id, start; cache identity.files.index(trajectory_id)',
        real_replay_groups=4,real_native_paths=128,real_closed_wrapper_paths=128,
        no_test_cache_or_test_future_read=True,old_negative_conclusion_valid=False,
        ready_input_catalog=True,seconds=time.perf_counter()-started,replay_gpu_seconds=gpu_seconds,
        legacy_precision_diagnostic_paths=150,
        pairing_policy='Old TF32 caches are past-feedback sources only. Future comparisons must use native_sample with the identical reconstructed noise and FP32 flags, never the old cached output as the FP32 baseline.',
        code_hashes=code_hashes(),
        backbone_state_sha256_before=base_hash,backbone_state_sha256_after=state_hash(native))
    dump(OUT/'DATA_AUDIT.json',base)
    print('REAL DATA PASS:',counts,'; maps=173; real32-member replay groups=4; updates=0',flush=True)
    return base
