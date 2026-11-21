"""Audit immutable dependencies, all identities, training/selection and fixed saved-output subset."""
from core import *
from research.provenance_uncertainty_conditioning.metrics import scores,structure
from collections import defaultdict


def main():
    configure();check();assert not (OUT/'VERIFICATION.json').exists()
    assert json.loads((OUT/'EVALUATION_TESTS.json').read_text())['passed']
    frozen=json.loads((OUT/'TRAINING_FROZEN.json').read_text())
    for p,h in frozen['code'].items():assert digest(ROOT/p)==h,p
    for p,h in json.loads((OUT/'EVALUATION_FROZEN.json').read_text())['code'].items():assert digest(ROOT/p)==h,p
    assert digest(OUT/'batch_schedule.json')==frozen['schedule_sha256']
    model,norm=load_model();std=norm['std'].reshape(2,1,1,1)
    complete=json.loads((OUT/'TRAINING_COMPLETE.json').read_text());assert complete['runs']==9 and complete['steps_each']==1000 and complete['total_updates']==9000
    assert state_hash(model)==complete['backbone_state_sha256']
    schedule=json.loads((OUT/'batch_schedule.json').read_text());training=read_csv(OUT/'training.csv');assert len(training)==9000
    assert len({(r['arm'],r['seed'],r['step']) for r in training})==9000
    maxloss=0.
    for r in training:
        s=schedule[int(r['step'])-1];assert int(r['k']) in [2,5,10]
        assert all(str(r[k])==str(s[k]) for k in ['trajectory','start','repeat','k'])
        maxloss=max(maxloss,abs(float(r['crps'])-float(r['member_truth'])+float(r['member_pair_half'])))
        assert float(r['gradient_norm'])>0 and np.isfinite(float(r['gradient_norm']))
    assert maxloss<2e-7
    for arm in CFG['arms']:
        for seed in CFG['seeds']:
            log=[json.loads(s) for s in (OUT/f'train_{arm}_{seed}.jsonl').read_text().splitlines()]
            assert len(log)==1000 and [r['step'] for r in log]==list(range(1,1001))
    weights=json.loads((OUT/'checkpoints.json').read_text());assert len(weights)==45
    initial=defaultdict(list)
    for item in weights:
        a=read_checkpoint(item);assert sum(p.numel() for p in a.parameters())==1760
        if item['step']==0:initial[item['seed']].append(state_hash(a))
        else:assert torch.count_nonzero(a.out.weight)>0
    assert all(len(set(v))==1 and len(v)==3 for v in initial.values())
    # Recalculate every scientific/protected choice from the full development table.
    dev=read_csv(OUT/'development.csv')
    assert len(dev)==5*9*48*4*3
    selections=json.loads((OUT/'selection.json').read_text());assert len(selections)==18
    candidates=read_csv(OUT/'selection_candidates.csv');assert len(candidates)==45
    for chosen in selections:
        arm=chosen['arm'];seed=str(chosen['seed'])
        own=[r for r in candidates if r['arm']==arm and r['seed']==seed]
        computed=[]
        for c in own:
            selected=[r for r in dev if r['arm']==arm and r['seed']==seed and r['step']==c['step'] and r['channel']=='overall']
            base=[r for r in dev if r['arm']==arm and r['seed']==seed and r['step']=='0' and r['channel']=='overall']
            objective=float(np.mean([float(r['crps']) for r in selected if r['k']!='0']))
            assert abs(objective-float(c['development_restart_crps']))<1e-12
            ratios=[]
            for k in CFG['eval_k']:
                for metric in ['mse','smse']:
                    ratio=np.mean([float(r[metric]) for r in selected if r['k']==str(k)])/np.mean([float(r[metric]) for r in base if r['k']==str(k)])
                    assert abs(ratio-float(c[f'{metric}_k{k}']))<1e-12;ratios.append(ratio)
            eligible=all(v<=CFG['noninferiority_ratio'] for v in ratios);assert eligible==(c['eligible']=='True')
            computed.append((objective,int(c['step']),eligible))
        if chosen['policy']=='scientific':expected=min(computed)[:2]
        else:
            zero=next(r for r in computed if r[1]==0);good=[r for r in computed if r[2] and r[0]<zero[0]];expected=(min(good) if good else zero)[:2]
        assert expected[1]==chosen['step']
    sf=json.loads((OUT/'FROZEN_SELECTION.json').read_text());es=json.loads((OUT/'EVALUATION_STARTED.json').read_text())
    assert sf['utc']<es['utc'] and digest(OUT/'selection.json')==sf['selection_sha256']
    manifest=[json.loads(s) for s in (OUT/'evaluation_manifest.jsonl').read_text().splitlines()]
    plan=json.loads((OUT/'evaluation_plan.json').read_text());assert len(manifest)==len(plan)==528
    events=[json.loads(s) for s in (OUT/'events.jsonl').read_text().splitlines() if json.loads(s)['action']!='development_complete'];assert len(events)==528*3
    extensions=[json.loads(s) for s in (OUT/'extension_manifest.jsonl').read_text().splitlines()];assert len(extensions)==480
    sources={e['path']:e for e in entries('test')+extensions}
    rawrows=read_csv(OUT/'scores.csv.gz')
    keys=META+['view','channel','policy','arm','seed']
    ix={tuple(str(r[k]) for k in keys):r for r in rawrows};assert len(ix)==len(rawrows)
    expected_score_rows=528*(19*3+3*2*20*3);assert len(rawrows)==expected_score_rows
    firsts={}
    for r in CFG['records']['test']:firsts.setdefault(r['group'],r['file'])
    audited_scores=0;maxscore=0.;cached_outputs=0;audit_groups=0;matrix_groups=0;allnoise=set();oldquality=read_csv(native.DELIVERY/'quality.csv')
    old_distribution=read_csv(native.DELIVERY/'distribution.csv')
    old_di={(r['trajectory'],r['start'],r['repeat'],r['k']):r for r in old_distribution if r['split']=='test'}
    oldix={(r['trajectory'],r['start'],r['repeat'],r['k'],r['method'],r['channel']):r for r in oldquality if r['split']=='test'}
    uniqueitems={f"{r['arm']}_{r['seed']}_{r['step']}":r['checkpoint'] for r in selections if r['step']}
    adapters={key:read_checkpoint(v) for key,v in uniqueitems.items()}
    selected_output_replays=0;maxreplay=0.
    for i,(record,planned) in enumerate(zip(manifest,plan)):
        assert all(record[k]==planned[k] for k in planned)
        eslice=events[3*i:3*i+3];assert [r['action'] for r in eslice]==['conditions_issued','predictions_issued','truth_revealed']
        assert all(all(e[k]==record[k] for k in planned) for e in eslice)
        assert eslice[0]['utc']<=eslice[1]['utc']<=eslice[2]['utc']
        source=sources[record['source']];b=prior.load_bundle(source)
        assert source['sha256']==record['source_sha256']
        for call in b['calls'].values():
            for h in call['noise_sha256']:assert h not in allnoise;allnoise.add(h)
        prepared={}
        for k in CFG['eval_k']:
            prepared[k]=prepare(b,k,norm);h,mask,spread,z=prepared[k];c=record['conditions'][str(k)]
            assert tensor_hash(h)==c['history_sha256'] and tensor_hash(mask)==c['mask_sha256'] and tensor_hash(spread)==c['spread_sha256']
            assert [tensor_hash(v) for v in z]==c['noise_sha256']
        assert digest(record['temporal_path'])==record['temporal_sha256'];matrix_groups+=1
        assert set(record['outputs'])=={f'{key}/k{k}' for key in uniqueitems for k in CFG['eval_k']}
        if 'audit_path' in record:
            assert digest(record['audit_path'])==record['audit_sha256']
            saved=torch.load(record['audit_path'],map_location='cpu',weights_only=False)
            assert saved['identity']==planned
            for key,p in saved['outputs'].items():assert tensor_hash(p)==record['outputs'][key];cached_outputs+=1
        truth=prior.frames(record['trajectory'],record['start']+20,record['start']+40).cuda();assert tensor_hash(truth)==record['truth_sha256']==eslice[2]['truth_sha256']
        if record['split']=='test':
            for k in CFG['k']:
                for method in ['B','S']:
                    for ch in ['overall','u','v']:
                        key=tuple(str(record[m]) if m!='k' else str(k) for m in META)+('tail',ch,'native',method,'-1')
                        previous=oldix[record['trajectory'],str(record['start']),str(record['repeat']),str(k),method,ch]
                        assert all(abs(float(ix[key][m])-float(previous[m]))<1e-12 for m in QUAL)
                        if method=='S' and ch=='overall':
                            old_dist=old_di[record['trajectory'],str(record['start']),str(record['repeat']),str(k)]
                            assert abs(float(ix[key]['restart_excess_ed_u'])-float(old_dist['excess_ed_u']))<1e-12
        # Fixed score audit: first trajectory of each test Re, all three saved windows, repeat0.
        if record['trajectory'] in firsts.values() and record['start'] in [400,440,820] and record['repeat']==0:
            audit_groups+=1;parent=b['paths']['A'].cuda();direct=b['paths']['B'].cuda();ind=b['paths']['C'].cuda()
            # Every selected nonzero checkpoint is replayed for k2 on this9-group audit subset.
            for name,a in adapters.items():
                with torch.no_grad():replay=decode(sample(model,a,*prepared[2]),norm)
                error=float((replay.cpu()-saved['outputs'][name+'/k2']).abs().max());assert error<=CFG['replay_atol'];maxreplay=max(maxreplay,error);selected_output_replays+=32
            for k in CFG['eval_k']:
                native_s=b['paths']['B' if k==0 else 'S'+str(k)].cuda()
                for view in (['direct'] if k==0 else ['tail','joint']):
                    def transform(p):return p if view=='direct' else p[:,:,:20-k] if view=='tail' else torch.cat([parent[:,:,:k],p[:,:,:20-k]],2)
                    refs=(direct[:,:,k:] if view=='tail' else direct,transform(native_s),ind[:,:,k:] if view=='tail' else ind)
                    structs=tuple(structure(p.double()/std) for p in refs);target=truth[:,k:] if view=='tail' else truth
                    for chosen in selections:
                        item=chosen['checkpoint'];name=f"{item['arm']}_{item['seed']}_{item['step']}"
                        p=transform(saved['outputs'][name+f'/k{k}'].cuda()) if item['step'] else transform(native_s)
                        rr,_,_=scores(p,target,std,view,k,parent,refs,structs)
                        for row in rr:
                            key=tuple(str(record[m]) if m!='k' else str(k) for m in META)+(view,row['channel'],chosen['policy'],item['arm'],str(item['seed']))
                            for metric,v in row.items():
                                if metric!='channel' and v is not None:maxscore=max(maxscore,abs(v-float(ix[key][metric])))
                            audited_scores+=1
        if (i+1)%48==0:print('VERIFIED identities',i+1,'/528',flush=True)
    assert audit_groups==9 and maxscore<1e-9
    # Every lead-table identity is unique, has complete physical indices and reduces to its score.
    leadcount=0
    for split in ['test','extension']:
        for seed in ['native','42','43','44']:
            groups={}
            with gzip.open(OUT/f'leads_{split}_{seed}.csv.gz','rt') as f:
                for row in csv.DictReader(f):
                    key=tuple(row[k] for k in keys);lead=int(row['tail_lead']);k=int(row['k'])
                    assert int(row['physical_frame'])==int(row['start'])+20+(k if row['view']=='tail' else 0)+lead-1
                    if key not in groups:groups[key]=[set(),np.zeros(5)]
                    seen,sums=groups[key];assert lead not in seen;seen.add(lead)
                    sums+=np.array([float(row[m]) for m in ['crps','mse','smse','variance','width90']]);leadcount+=1
            for key,(seen,sums) in groups.items():
                row=ix[key];length=20-int(row['k']) if row['view']=='tail' else 20
                assert seen==set(range(1,length+1))
                assert np.max(np.abs(sums/length-np.array([float(row[m]) for m in ['crps','mse','smse','variance','width90']])))<1e-10
    cost=json.loads((OUT/'EVALUATION_COST.json').read_text());assert cost['groups']==528 and cost['cache_bytes']<CFG['max_cache_bytes']
    actual_cache_bytes=sum(p.stat().st_size for p in CACHE.rglob('*') if p.is_file())
    assert actual_cache_bytes==cost['cache_bytes']
    assert cost['adapter_full20_paths']==len(uniqueitems)*528*4*32 and cost['native_extension_full20_paths']+cost['reused_native_extension_full20_paths']==480*192
    assert sum(cost['lead_rows'].values())==leadcount
    assert cost['wall_seconds']<=CFG['max_evaluation_seconds'] and complete['seconds']<=CFG['max_training_seconds']
    # Independently reconstruct critical CRPS paired trajectory intervals including seed pooling.
    paired=read_csv(OUT/'paired_intervals.csv');ptr=read_csv(OUT/'paired_by_trajectory.csv');bootstrap_checked=0
    for r in paired:
        if r['metric']!='crps':continue
        fields=['split','k','view','channel','policy','seed','contrast']
        subset=sorted([p for p in ptr if all(p[f]==r[f] for f in fields)],key=lambda p:p['trajectory'])
        assert len(subset)==12
        a=np.array([float(p['crps']) for p in subset]);rng=np.random.default_rng(123);v=a[rng.integers(12,size=(10000,12))].mean(1);ci=np.quantile(v,[.025,.975,.05/6,1-.05/6])
        assert abs(a.mean()-float(r['mean']))<1e-12
        assert np.max(np.abs(ci-np.array([float(r[k]) for k in ['ci95_low','ci95_high','ci98_333_low','ci98_333_high']])))<1e-12;bootstrap_checked+=1
    check()
    dump(OUT/'VERIFICATION.json',dict(passed=True,updates=9000,checkpoints=45,selection_choices=18,groups=528,extension_sources=480,all_noise_hashes_unique=len(allnoise),audit_score_groups=audit_groups,audit_score_rows=audited_scores,audit_score_maxabs=maxscore,cached_outputs_checked=cached_outputs,temporal_files_checked=matrix_groups,selected_output_replay_paths=selected_output_replays,replay_maxabs=maxreplay,all_lead_rows_checked=leadcount,crps_bootstrap_intervals_checked=bootstrap_checked,loss_decomposition_maxabs=maxloss,old_files_unchanged=True,backbone_state_sha256=state_hash(model),utc=time.time()))
    print('VERIFICATION PASS',flush=True)

if __name__=='__main__':main()
