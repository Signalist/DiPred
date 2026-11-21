"""Read-only audit of saved outputs, identities, score algebra, hashes and event order."""
from core import *


def main():
    configure();check();assert not (OUT/'VERIFICATION.json').exists()
    frozen=json.loads((OUT/'FROZEN.json').read_text())
    for p,h in frozen['code'].items():assert digest(ROOT/p)==h,p
    assert frozen['preflight_sha256']==digest(OUT/'PREFLIGHT.json')
    model,norm=model_load();std=norm['std'].cpu().double().reshape(2,1,1,1)
    manifest=json.loads((OUT/'cache_manifest.json').read_text());plan=json.loads((OUT/'plan.json').read_text())
    assert len(manifest)==len(plan)==288
    q=old.read_csv(OUT/'quality.csv');responses=old.read_csv(OUT/'response_members.csv');signed=old.read_csv(OUT/'signed_mse.csv')
    qi={tuple(str(r[k]) for k in META)+ (r['method'],r['channel']):r for r in q}
    ri={tuple(str(r[k]) for k in META)+ (str(r['member']),r['channel']):r for r in responses}
    si={tuple(str(r[k]) for k in META)+ (r['scale'],r['channel']):r for r in signed}
    events=[json.loads(s) for s in (OUT/'events.jsonl').read_text().splitlines()];assert len(events)==288*3
    original={e['path']:e for e in entries()};old_loaded={};maxscore=0.;maxalgebra=0.;maxjvp=0.;bytes_total=0;jn=0
    source_hashes=json.loads((old.DELIVERY/'data_hashes.json').read_text())
    for split in ['development','test']:
        for record in CFG['records'][split]:assert digest(old.DATA/record['file'])==source_hashes[record['file']]
    for idx,(entry,planned) in enumerate(zip(manifest,plan)):
        assert all(entry[k]==planned[k] for k in META)
        meta=tuple(str(entry[k]) for k in META);k=entry['k'];length=20-k
        assert [e['action'] for e in events[idx*3:idx*3+3]]==['oracle_prefix_revealed','paths_issued','scoring_tail_revealed']
        assert all(tuple(str(e[k]) for k in META)==meta for e in events[idx*3:idx*3+3])
        assert events[idx*3+1]['sha256']==entry['sha256']
        assert digest(entry['path'])==entry['sha256'];bytes_total+=Path(entry['path']).stat().st_size
        b=torch.load(entry['path'],map_location='cpu',weights_only=False)
        src=entry['source']
        if src not in old_loaded:old_loaded={src:load_bundle(original[src])}
        oldb=old_loaded[src]
        prefix=oracle_prefix(entry['trajectory'],entry['start'],k);r,m,s=inputs(oldb['history'],oldb['paths']['A'],prefix,k)
        assert torch.equal(r,b['history_R']) and torch.equal(m,b['history_M'])
        assert tensor_hash(s)==oldb['calls']['S'+str(k)]['history_sha256']
        z=noise_for(oldb,k,s.cuda());del z
        for group in ['R','M']:
            assert b['calls'][group]['noise_sha256']==oldb['calls']['S'+str(k)]['noise_sha256']
            assert b['calls'][group]['seeds']==oldb['calls']['S'+str(k)]['seeds']
            assert tensor_hash(b['paths'][group])==b['calls'][group]['tail_sha256']
        truth=frames(entry['trajectory'],entry['start']+20+k,entry['start']+40).double()
        assert b['compared_frames']==list(range(entry['start']+20+k,entry['start']+40))
        assert tensor_hash(truth.float())==events[idx*3+2]['sha256']
        r,m,s=b['paths']['R'].double(),b['paths']['M'].double(),oldb['paths']['S'+str(k)][:,:,:length].double()
        for method,p in [('R',r),('M',m),('S',s),('B',oldb['paths']['B'][:,:,k:].double())]:
            for ch,name in [(slice(None),'overall'),(slice(0,1),'u'),(slice(1,2),'v')]:
                a=p[:,ch]/std[ch];y=truth[ch]/std[ch];n=len(a)
                ordered=a.sort(0).values;weights=(2*torch.arange(n,dtype=torch.double)-n+1).reshape(n,1,1,1,1)
                crps=float((a-y).abs().mean()-(ordered*weights).sum(0).mean()/n**2)
                row=qi[meta+(method,name)]
                for metric,value in [('crps',crps),('smse',float((a.mean(0)-y).square().mean())),('mse',float((p[:,ch].mean(0)-truth[ch]).square().mean()))]:maxscore=max(maxscore,abs(value-float(row[metric])))
        d,dm,dd=(s-r)/std,(m-r)/std,(s-m)/std
        maxalgebra=max(maxalgebra,float((d-dm-dd).abs().max()))
        for member in range(32):
            row=ri[meta+(str(member),'overall')]
            for metric,value in [('D_square',float(d[member].square().mean())),('Dm_square',float(dm[member].square().mean())),('Dd_square',float(dd[member].square().mean())),('cross',float(2*(dm[member]*dd[member]).mean()))]:maxalgebra=max(maxalgebra,abs(value-float(row[metric])))
        for scale,st in [('physical',1),('standardized',std)]:
            er=(r.mean(0)-truth)/st;delta=(s.mean(0)-r.mean(0))/st
            row=si[meta+(scale,'overall')]
            lhs=float((((s.mean(0)-truth)/st).square()-er.square()).mean());cross=float(2*(er*delta).mean());square=float(delta.square().mean())
            maxalgebra=max(maxalgebra,abs(lhs-cross-square),abs(lhs-float(row['mse_difference'])),abs(cross-float(row['error_response_cross'])))
        if 'jvp_path' in entry:
            assert digest(entry['jvp_path'])==entry['jvp_sha256'];bytes_total+=Path(entry['jvp_path']).stat().st_size
            j=torch.load(entry['jvp_path'],map_location='cpu',weights_only=False)
            assert j['member_ids']==CFG['jvp_members'] and torch.count_nonzero(j['input_error'][:,:,:20-k])==0
            assert j['replay_passed'] and max(j['replay_R_maxabs'],j['replay_S_maxabs'])<=CFG['replay_atol']
            assert torch.isfinite(j['linear']).all() and rms(j['linear'])>0
            jr=[row for row in old.read_csv(OUT/'jvp.csv') if tuple(str(row[key]) for key in META)==meta and row['channel']=='overall']
            for i,member in enumerate(j['member_ids']):
                for dose,lin,act in [('full',j['linear'][i],j['actual'][i]),('half',.5*j['linear'][i],j['half'][i]-j['base'][i])]:
                    row=next(row for row in jr if row['member']==str(member) and row['dose']==dose)
                    err=rms(lin-act)
                    maxjvp=max(maxjvp,abs(err-float(row['error_rms'])))
            jn+=4
        if (idx+1)%48==0:print('VERIFIED',idx+1,'/288',flush=True)
    assert maxscore<1e-10 and maxalgebra<1e-10 and maxjvp<1e-6,(maxscore,maxalgebra,maxjvp)
    complete=json.loads((OUT/'COMPLETE.json').read_text());assert bytes_total==complete['cache_bytes']<=CFG['max_cache_bytes']
    assert complete['paths']==18432 and complete['seconds']<=CFG['max_seconds']
    assert jn==(288 if frozen['jvp_enabled'] else 0)
    check()
    dump(OUT/'VERIFICATION.json',dict(passed=True,groups=288,paths=18432,jvp_directions=jn,source_data_hashes_checked=24,source_cache_hashes_checked=96,event_order_checked=True,old_files_unchanged=True,checkpoint_hashes=CFG['frozen'],paired_noise_checked=True,input_prefix_checked=True,score_maxabs=maxscore,algebra_maxabs=maxalgebra,jvp_metrics_maxabs=maxjvp,cache_bytes=bytes_total,utc=time.time()))
    print('VERIFICATION_PASS',maxscore,maxalgebra,maxjvp,flush=True)

if __name__=='__main__':main()
