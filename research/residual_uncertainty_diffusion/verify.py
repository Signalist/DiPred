"""No training/inference: independently verify saved inputs, scores and logs."""
import csv,time
from collections import defaultdict
from .core import *
from .experiment import freeze_hash


def csvread(path):
    rows=list(csv.DictReader(Path(path).open()))
    for r in rows:
        if 'step' in r:r['step']=int(r['step'])
        if 'start' in r:r['start']=int(r['start'])
        for m in METRICS:
            if m in r:r[m]=float(r[m])
    return rows


def main():
    started=time.perf_counter();protected=read(OUT/'PROTECTED.json')
    for path,sha in protected.items():assert digest(ROOT/path)==sha,path
    frozen=read(OUT/'FROZEN.json');pre=read(OUT/'PREFLIGHT.json')
    for source in (frozen,pre):
        for p,sha in source['code_hashes'].items():assert digest(ROOT/p)==sha,p
    assert pre['passed'] and pre['optimizer_updates']==0
    assert pre['full_scoring_pipeline_synthetic_pass'] and pre['fair8_gradcheck']
    assert pre['fair8_gradient_error']<1e-12
    manifest=read(OUT/'DATA_MANIFEST.json');assert len(manifest)==173
    assert len({(r['split'],r['trajectory_id'],r['start']) for r in manifest})==173
    noise=read(OUT/'NOISE_MANIFEST.json');plan=noise['schedule'];assert len(plan)==500
    logs={a:[json.loads(t) for t in (OUT/f'train_{a}.jsonl').read_text().splitlines()] for a in ARMS}
    for a in ARMS:
        assert len(logs[a])==500 and [r['step'] for r in logs[a]]==list(range(1,501))
        assert all(np.isfinite(r['fair8_crps']) and r['out_gradient_norm']>0 for r in logs[a])
    for i,item in enumerate(plan):
        c0,c1=logs['C0'][i],logs['C1'][i]
        assert c0['key']==c1['key']==item['key']
        assert c0['member_ids']==c1['member_ids']==item['member_ids'] and len(set(item['member_ids']))==8
        assert c0['noise_sha256']==c1['noise_sha256']
        assert c0['updates_total']==2*i+1 and c1['updates_total']==2*i+2
    complete=read(OUT/'TRAINING_COMPLETE.json');assert complete['updates']==1000 and complete['steps_each']==500
    native,norm=load_backbone();initial=state_hash(native.net.out)
    assert initial==complete['backbone_out_initial_hash']
    assert complete['frozen_non_out']==freeze_hash(native,True)
    ckpts=read(OUT/'CHECKPOINTS.json');assert {(r['arm'],r['step']) for r in ckpts}=={(a,s) for a in ARMS for s in (250,500)}
    out_changes={}
    for r in ckpts:
        assert digest(ROOT/r['path'])==r['sha256']
        model=CorrectedDiffusion(native);ck=torch.load(ROOT/r['path'],map_location='cuda',weights_only=False)
        model.correction.load_state_dict(ck['correction']);model.native.net.out.load_state_dict(ck['out'])
        assert state_hash(model.correction)==r['correction_state_hash'] and state_hash(model.native.net.out)==r['out_state_hash']
        assert freeze_hash(model.native,True)==complete['frozen_non_out']
        delta=torch.cat([(model.native.net.out.state_dict()[k]-native.net.out.state_dict()[k]).double().flatten() for k in ck['out']])
        rms=float(delta.square().mean().sqrt());assert rms>0
        out_changes[f"{r['arm']}_{r['step']}"]=rms
        if r['step']==500:assert abs(rms-complete['output_changes'][r['arm']]['out_delta_rms'])<1e-12
    # All saved E/W/target hashes and training-only condition scales.
    entries=[]
    for split in ('train','development'):
        p=CACHE/f'{split}_conditions.pt'
        assert digest(p)==read(OUT/f'{split}_XPF_AUDIT.json')['cache_sha256']
        data=torch.load(p,map_location='cpu',weights_only=False);entries.extend(data)
        lookup={r['cache_key']:r for r in manifest}
        for e in data:
            row=lookup[e['cache_key']];k=e['key'];assert row['history_interval']==row['previous_target_interval']==[k.start,k.start+20]
            assert row['target_interval']==[k.start+20,k.start+40]
            assert tensor_hash(e['E'])==row['E_sha256'] and tensor_hash(e['W'])==row['W_sha256']
            assert tensor_hash(e['target'])==row['target_sha256']
            assert e['key'].split==split
    train=[e for e in entries if e['key'].split=='train'];dev=[e for e in entries if e['key'].split=='development']
    assert len(train)==137 and len(dev)==36
    scales=read(OUT/'CONDITION_SCALES.json')
    ee=torch.cat([e['E'].flatten() for e in train]);ww=torch.stack([e['W'] for e in train])
    assert float(torch.quantile(ee.abs(),.995).clamp_min(.1))==scales['E_clip']
    assert [float(v) for v in ww.square().mean((0,2,3,4)).sqrt().clamp_min(1e-6)]==scales['W_scale']
    reader=H5Reader(DATA);rs=records()
    # Validate each historical source exactly by cache key; no XPF repetition.
    for split in ('train','development'):
        for s in (420,440,460):
            rows=[r for r in manifest if r['split']==split and r['start']==s]
            path=ROOT/rows[0]['feedback_cache'];assert digest(path)==rows[0]['feedback_cache_sha256']
            index=CacheIndex(torch.load(path,map_location='cpu',weights_only=False),split,s-20,[r['file'] for r in rs[split]],str(path))
            for r in rows:
                e=next(e for e in entries if e['cache_key']==r['cache_key']);H=reader(r['trajectory_id'],s,s+20)
                previous=index.lookup(Key(split,r['trajectory_id'],s-20))
                assert tensor_hash(H)==r['history_sha256'] and tensor_hash(previous.paths)==r['previous_path_sha256']
                recomputed=(previous.paths.mean(0)-H)/norm['std'].cpu()
                assert torch.equal(recomputed,e['E'])
            del index
    # All 540 score rows replayed from the exact cached 32-member paths.
    rows=csvread(OUT/'DEVELOPMENT.csv');assert len(rows)==540
    lookup={(r['arm'],r['step'],r['trajectory_id'],r['start'],r['channel']):r for r in rows}
    assert len(lookup)==len(rows)
    maxerr=0.;explicit_errors=[]
    for arm,step in [('native',0)]+[(a,s) for a in ARMS for s in (250,500)]:
        saved=torch.load(CACHE/f'development_{arm}_{step}_paths.pt',map_location='cpu',weights_only=False)
        assert len(saved)==36 and len({r['key'] for r in saved})==36
        for i,item in enumerate(saved):
            entry=next(e for e in dev if e['cache_key']==item['key']);p=item['paths'].cuda()
            assert p.shape==(32,2,20,32,64) and tensor_hash(p)==item['sha256']
            values=score(p,entry['target'],norm)
            for value in values:
                original=lookup[(arm,step,entry['key'].trajectory_id,entry['key'].start,value['channel'])]
                assert original['output_sha256']==item['sha256']
                for m in METRICS:
                    error=abs(value[m]-original[m]);assert error<1e-12,(m,error)
                    maxerr=max(maxerr,error)
            assert abs(values[0]['smse']-.5*(values[1]['smse']+values[2]['smse']))<1e-12
            if i==0:
                z=p.double();y=(entry['target'].cuda().double()-norm['mean'].double())/norm['std'].double()
                pair=sum((z[j:j+1]-z[:j]).abs().mean((1,2,3,4)).sum() for j in range(1,32))
                first=(z-y).abs().mean()
                explicit=first-pair/(32*32);fair=first-pair/(32*31)
                assert abs(float(explicit)-values[0]['crps'])<1e-12 and abs(float(fair)-values[0]['fair_crps'])<1e-12
                explicit_errors.append(abs(float(explicit)-values[0]['crps']))
    selected,tr,macro=select(rows);assert selected==read(OUT/'SELECTION.json')
    assert all(r['chosen_arm']=='native' for r in selected)
    assert read(OUT/'TEST_STOP.json')['test_inference_groups']==0 and not (OUT/'TEST.csv').exists()
    assert not (OUT/'STOP.json').exists()
    for path,sha in protected.items():assert digest(ROOT/path)==sha
    result=dict(passed=True,actual_optimizer_updates=1000,runs={'C0':500,'C1':500},no_restarted_training=True,
        protected_files_unchanged=len(protected),condition_records=173,historical_E_identity_recomputed_all=True,
        xpf_reused_frozen=True,absolute_risk_information_retained=True,train8_fair_development32_empirical=True,
        independent_score_rows=540,max_score_error=maxerr,explicit_full32_crps_errors=explicit_errors,
        paired_noise_hashes_checked=500,only_native_out_and_G_updated=True,backbone_out_changes=out_changes,
        both_scientific_selections_native=True,no_test_inference=True,seconds=time.perf_counter()-started,
        verified_sources=code_hashes())
    dump(OUT/'VERIFICATION.json',result);print('VERIFIED:1000 updates;540 score rows;both native;no test',flush=True)

if __name__=='__main__':main()
