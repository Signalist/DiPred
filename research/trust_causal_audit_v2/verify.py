"""Independent checks of audit evidence, keyed maps, sources, and zero updates."""
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from research.trust_causal_audit_v2.common import ROOT,OUT,DATA,dump,digest,tensor_hash,records,assert_protected,code_hashes
from research.trust_causal_audit_v2.data import H5Reader,CacheIndex,Key,batch_noise


def main():
    protected=assert_protected()
    pre=json.loads((OUT/'PREFLIGHT.json').read_text())
    data=json.loads((OUT/'DATA_AUDIT.json').read_text())
    execution=json.loads((OUT/'EXECUTION.json').read_text())
    assert pre['passed'] and data['passed']
    assert all(pre[x]['passed'] for x in 'ABCDEFG')
    assert all(pre['G'][k] for k in ('future_leak_mutant_rejected','last_valid_mutant_rejected','positional_cache_mutant_rejected'))
    assert pre['parameters']=={'G_context':1269,'G_trust':1269}
    assert pre['identical_parameter_initialization']
    for record in (pre,data,execution):
        assert record['optimizer_updates']==0 and record['training_runs']==0
    assert execution['optimizer_construction_guard_enabled']
    assert execution['formal_test_comparisons']==0 and execution['new_native_cache_generation']==0
    for record in (pre,data):
        assert record['backbone_state_sha256_before']==record['backbone_state_sha256_after']
        for name,sha in record['code_hashes'].items():assert digest(ROOT/name)==sha,name
    for arm in ('G_context','G_trust'):
        assert [r['label'] for r in pre['F']['four_step_direct_gradients'][arm]]==[99,66,33,0]
        assert all(v>0 for row in pre['F']['four_step_direct_gradients'][arm] for v in row['direct_parameter_gradient_norms'].values())
    assert pre['F']['full32_crps_value_error']<1e-12 and pre['F']['full32_crps_gradient_error']<1e-12
    manifest=json.loads((OUT/'WINDOW_MANIFEST.json').read_text())
    keys=[(r['split'],r['trajectory_id'],r['start']) for r in manifest]
    assert len(set(keys))==len(keys)==173
    assert sum(r['split']=='train' for r in manifest)==137
    assert sum(r['split']=='development' for r in manifest)==36
    test=json.loads((OUT/'TEST_CATALOG.json').read_text());assert len(test)==12
    assert all(r['audit_scope']=='metadata_only' and r['target_interval']==[500,520] for r in test)
    assert not any(r['split']=='test' for r in manifest)
    scale=json.loads((OUT/'RISK_NORMALIZATION.json').read_text())
    ck=torch.load(ROOT/'reliability/checkpoints/current.pt',map_location='cpu',weights_only=False)
    std=ck['normalization']['std'].reshape(2,1,1,1)
    assert scale['checkpoint_sha256']==digest(ROOT/'reliability/checkpoints/current.pt')
    maps=np.load(OUT/'risk_maps.npz')
    reader=H5Reader(DATA);rs=records()
    lock=json.loads((OUT/'CACHE_MANIFEST.json').read_text())
    raw_train=[];cache_errors=[]
    for path,meta in lock.items():
        p=ROOT/path;assert digest(p)==meta['sha256']
        # Replay provenance caches at 460 are metadata-checked, not risk sources.
        payload=torch.load(p,map_location='cpu',weights_only=False)
        split=meta['identity']['split'];start=meta['identity']['start']
        index=CacheIndex(payload,split,start,[r['file'] for r in rs[split]],path)
        assert tensor_hash(index.paths)==meta['paths_sha256']
        for row in manifest:
            if row['previous_cache']!=path:continue
            s=row['start'];name=row['trajectory_id'];key=Key(split,name,s-20)
            previous=index.lookup(key)
            assert previous.cache_row==row['previous_cache_row']==row['original_index']
            assert tensor_hash(previous.paths)==row['previous_path_sha256']
            h=reader(name,s,s+20);target=reader(name,s+20,s+40)
            assert tensor_hash(h)==row['history_sha256'] and tensor_hash(target)==row['target_sha256']
            assert row['previous_prediction_interval']==row['history_interval']==[s,s+20]
            assert row['target_interval']==[s+20,s+40]
            # Independent reduction of the complete 40 channel/lead entries.
            error=(previous.paths.mean(0)-h)/std
            raw=(error*error).sum((0,1))[None]/40
            stored=torch.from_numpy(maps[row['map_key']+'__raw'])
            diff=float((raw-stored).abs().max())
            assert torch.allclose(raw,stored,atol=1e-8,rtol=1e-6)
            assert tensor_hash(stored)==row['raw_risk_sha256']
            risk=(torch.log1p(stored)/scale['log1p_p99']).clamp(0,1)
            assert tensor_hash(risk)==row['risk_sha256']
            assert np.array_equal(risk.numpy(),maps[row['map_key']+'__risk'])
            assert risk.min()>=0 and risk.max()<=1
            if split=='train':raw_train.append(stored)
            cache_errors.append(diff)
        del payload,index
    q=float(torch.quantile(torch.log1p(torch.stack(raw_train).flatten()),.99).clamp_min(1e-6))
    assert q==scale['log1p_p99'] and scale['split']=='train' and scale['maps']==137
    noise=json.loads((OUT/'NOISE_IDENTITIES.json').read_text())
    groups={}
    for n in noise:groups.setdefault((n['split'],n['trajectory_id'],n['history_start']),[]).append(n)
    assert len(groups)==4 and len(noise)==128
    for (split,name,start),nn in groups.items():
        _,recomputed=batch_noise(Key(split,name,start),[r['file'] for r in rs[split]])
        assert recomputed==nn
    replay=json.loads((OUT/'REAL_REPLAY.json').read_text())
    assert len(replay)==4
    for r in replay:
        assert r['members']==32 and r['leads']==20 and r['disabled_wrapper_maxabs']==0
        assert r['new_FP32_baseline_matches_closed_wrapper'] and r['strict_noise_pairing']
        assert r['old_cache_original_tf32_batch_member0_error']<=2e-6
        assert not r['old_cache_is_strict_FP32_output_baseline']
        assert r['old_native_cache_FP32_physical_maxabs']>r['physical_tolerance']==2e-6
    result=dict(passed=True,READY_FOR_RETRAINING=True,optimizer_updates=0,training_runs=0,
        formal_test_comparisons=0,protected_files_unchanged=protected,
        source_hashes_checked=True,independent_risk_maps_checked=len(cache_errors),
        independent_raw_risk_maxabs=max(cache_errors),independent_train_scale=q,
        real_noise_hashes_reconstructed=len(noise),old_precision_mismatch_preserved=True,
        ready_scope='v2 inputs, keyed caches, new identical-noise FP32 native/wrapper comparison and untrained routing interface only; no performance claim',
        required_next_run_policy='Use new native_sample with identical actual noise/FP32 flags for baseline; do not compare to archived TF32 outputs as a strict FP32 baseline',
        code_hashes=code_hashes())
    dump(OUT/'VERIFICATION.json',result)
    print('VERIFIED: READY_FOR_RETRAINING=true; maps=173; updates=0; old artifacts unchanged',flush=True)


if __name__=='__main__':main()
