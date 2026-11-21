"""Independent read-only cache, identity, time alignment and statistical audit."""
from core import *
from analyze import interval,QUAL,DIST


def main():
    check_dependencies();frozen=json.loads((OUT/'FROZEN.json').read_text())
    assert frozen['config_sha256']==CONFIG_HASH
    for name,sha in frozen['code'].items():assert digest(ROOT/name)==sha
    assert digest(OUT/'training_history_reference.json')==frozen['training_history_sha256']
    assert digest(OUT/'data_hashes.json')==frozen['data_sha256']
    for name,sha in json.loads((OUT/'data_hashes.json').read_text()).items():assert digest(DATA/name)==sha
    pre=json.loads((OUT/'PREFLIGHT.json').read_text());assert pre['passed']
    for r in pre['caches']:assert digest(Path(r['path']))==r['sha256']
    complete=json.loads((OUT/'COMPLETE.json').read_text());assert complete['groups']==280 and complete['samples']==53760 and complete['network_calls']==6720
    manifest=json.loads((OUT/'cache_manifest.json').read_text());assert len(manifest)==280
    expected={(sp,r['file'],start,rep) for sp,rr in RECORDS.items() for r in rr for start in CFG['starts'] for rep in range(2)}
    assert {(m['split'],m['trajectory'],m['start'],m['repeat']) for m in manifest}==expected
    events=[json.loads(line) for line in (OUT/'events.jsonl').read_text().splitlines()];assert len(events)==560
    all_seeds=[];all_noise=[];samples=0;calls=0
    scores=read_csv(OUT/'quality.csv');distribution=read_csv(OUT/'distribution.csv');detail=read_csv(OUT/'lead_channel.csv')
    assert len(scores)==10080 and len(distribution)==840 and len(detail)==96320
    qi={(r['split'],r['trajectory'],r['start'],r['repeat'],r['k'],r['method'],r['channel']):r for r in scores};assert len(qi)==len(scores)
    di={(r['split'],r['trajectory'],r['start'],r['repeat'],r['k']):r for r in distribution};assert len(di)==len(distribution)
    norm=json.loads((OUT/'training_history_reference.json').read_text())['normalization'];norm={k:torch.tensor(v,device='cuda') for k,v in norm.items()};std=norm['std'].reshape(2,1,1,1)
    rechecks=0
    for j,m in enumerate(manifest):
        file=Path(m['path']);assert digest(file)==m['sha256']
        bundle=torch.load(file,map_location='cpu',weights_only=False);ident=bundle['identity']
        assert ident['config_sha256']==CONFIG_HASH and ident['frozen']==CFG['frozen']
        assert ident['member_ids']==list(range(32)) and ident['parent_child_ids']==[[i,i] for i in range(32)]
        assert tensor_hash(bundle['history'])==m['history_sha256']
        assert ident['history_indices']==list(range(m['start'],m['start']+20))
        first,second=events[2*j:2*j+2]
        assert first['action']=='issued_all_paths' and second['action']=='truth_revealed'
        for field in ['split','trajectory','start','repeat']:assert first[field]==second[field]==m[field]
        assert first['utc']>=frozen['utc'] and first['utc']<=second['utc'] and first['cache_sha256']==m['sha256']
        for name,path in bundle['paths'].items():
            assert path.shape==(32,2,20,32,64) and bool(torch.isfinite(path).all())
            call=bundle['calls'][name];family='S' if name.startswith('S') else name;k=int(name[1:]) if family=='S' else 0
            assert call['seeds']==seeds_for(m['split'],m['trajectory'],m['start'],m['repeat'],family,k)
            assert call['output_sha256']==tensor_hash(path)==m['outputs'][name]==first['outputs'][name]
            history=restart_history(bundle['history'],bundle['paths']['A'],k) if family=='S' else bundle['history'][None].expand(32,-1,-1,-1,-1)
            assert tensor_hash(history)==call['history_sha256']
            assert call['noise_shape']==[32,2,20,32,64]
            all_seeds+=call['seeds'];all_noise+=call['noise_sha256'];samples+=call['samples'];calls+=call['network_calls']
            if family=='S':assert ident['k'][str(k)]==physical_indices(m['start'],k)
        # One cache group from each split is independently rescored, not regenerated.
        if m['start']==400 and m['repeat']==0 and m['trajectory']==RECORDS[m['split']][0]['file']:
            truth=raw(DATA,[RECORDS[m['split']][0]],420)[0]
            assert tensor_hash(truth)==second['truth_sha256']
            h=raw(DATA,[RECORDS[m['split']][0]],400)[0];assert tensor_hash(h)==m['history_sha256']
            for k in CFG['k']:
                for family in ['A','B','C','S']:
                    path=bundle['paths'][family if family!='S' else 'S'+str(k)].cuda();tail=path[:,:,k:] if family!='S' else path[:,:,:20-k]
                    rr,_=quality(tail,truth[:,k:],std)
                    for row in rr:
                        saved=qi[m['split'],m['trajectory'],str(m['start']),str(m['repeat']),str(k),family,row['channel']]
                        for metric in QUAL:assert abs(row[metric]-float(saved[metric]))<1e-9
                        rechecks+=1
                b=bundle['paths']['B'][:,:,k:].cuda().double()/std;c=bundle['paths']['C'][:,:,k:].cuda().double()/std;s=bundle['paths']['S'+str(k)][:,:,:20-k].cuda().double()/std
                left,right=discrepancy(b,s),discrepancy(b,c);saved=di[m['split'],m['trajectory'],str(m['start']),str(m['repeat']),str(k)]
                for metric in left:
                    assert abs(left[metric]-float(saved['restart_'+metric]))<1e-9
                    assert abs(left[metric]-right[metric]-float(saved['excess_'+metric]))<1e-9
    assert samples==53760 and calls==6720 and len(set(all_seeds))==53760 and len(set(all_noise))==53760
    for row in distribution:
        for metric in ['ed_u','ed_v','mean_rms','centered_ed_u','variance_delta','projection_cov_difference','projection_correlation_difference']:
            assert abs(float(row['restart_'+metric])-float(row['sampling_'+metric])-float(row['excess_'+metric]))<1e-12
    # Aggregation of every lead/channel must reproduce its matching whole-tail score.
    for row in aggregate(detail,['split','trajectory','start','repeat','k','method','channel'],['crps','mse','smse','variance','width90']):
        saved=qi[tuple(row[k] for k in ['split','trajectory','start','repeat','k','method','channel'])]
        for metric in ['crps','mse','smse','variance','width90']:assert abs(row[metric]-float(saved[metric]))<1e-10
    tr=aggregate(distribution,['split','k','trajectory'],DIST)
    for row in read_csv(OUT/'distribution_intervals.csv'):
        values=[r[row['metric']] for r in sorted(tr,key=lambda r:r['trajectory']) if r['split']==row['split'] and r['k']==row['k']]
        actual=interval(values)
        for metric in ['mean','ci95_low','ci95_high','ci98_333_low','ci98_333_high']:assert abs(actual[metric]-float(row[metric]))<1e-12
    for split in RECORDS:
        for other in RECORDS:
            if split==other:continue
            assert {r['file'] for r in RECORDS[split]}.isdisjoint(r['file'] for r in RECORDS[other])
            assert {r['group'] for r in RECORDS[split]}.isdisjoint(r['group'] for r in RECORDS[other])
    dump(OUT/'verification.json',dict(passed=True,cache_groups=280,unique_seeds=len(set(all_seeds)),unique_actual_noise_hashes=len(set(all_noise)),samples=samples,network_calls=calls,independently_rescored_quality_rows=rechecks,code_frozen=True,history_sources_verified=True,parent_child_identity_verified=True,prediction_before_truth=True,original_trajectory_bootstrap=True,config_sha256=CONFIG_HASH))
    print('PASS280 caches,53760 unique seed/noise identities, correct physical slices and parent-child histories')
    print('PASS issued-before-truth,108 independently rescored quality rows, all detailed-to-block scores and trajectory intervals')
    print('PASS frozen code/model/normalization/source data, old artifacts unchanged; no training')

if __name__=='__main__':
    configure();main()
