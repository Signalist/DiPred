"""Read-only verification of all delivered small tables; no model sampling."""
from run import *
from collections import defaultdict

def read(name):
    with (OUT/name).open() as f:return list(csv.DictReader(f))

def main():
    assert not (OUT/'VERIFICATION.json').exists()
    check();f=json.loads((OUT/'FROZEN.json').read_text());assert f['code_sha256']==digest(Path(__file__).with_name('run.py'))
    assert f['config_sha256']==digest(SPEC/'config.json') and f['manifest_sha256']==digest(SPEC/'manifest.json')
    m=json.loads((SPEC/'manifest.json').read_text());ids=json.loads((OUT/'identities.json').read_text());c=json.loads((OUT/'COMPLETE.json').read_text())
    cfg=old.CFG;groups=defaultdict(list)
    for r in cfg['records']['train']:groups[r['group']].append(r['file'])
    for g in groups:groups[g].sort()
    order=sorted(groups,key=int);expected=[groups[g][0] for g in order]+[groups[g][1] for g in order if len(groups[g])>1][:12-len(order)]
    assert CFG['train_files']==expected and len(expected)==12
    assert CFG['development_files']==[r['file'] for r in cfg['records']['development']]
    assert len(m)==len(ids)==24 and c['new_forward_groups']==48 and c['reused_step0_groups']==24 and c['optimizer_updates']==0
    for entry,ident in zip(m,ids):
        assert all(entry[k]==ident[k] for k in ['split','trajectory','Re','start','repeat'])
        assert entry['sha256']==ident['source_sha256'] and entry['outputs']['A']==ident['parent_sha256']
        assert entry['outputs']['S5']==ident['outputs']['0']
        assert entry['calls']['S5']['seeds']==ident['noise_seeds'] and entry['calls']['S5']['noise_sha256']==ident['noise_sha256']
        assert entry['split'] in ['train','development']
    events=[json.loads(x) for x in (OUT/'events.jsonl').read_text().splitlines()];assert len(events)==48
    for i,ident in enumerate(ids):
        a,b=events[2*i:2*i+2];assert a['action']=='predictions_complete' and b['action']=='truth_revealed' and a['utc']<=b['utc']
        assert a['trajectory']==b['trajectory']==ident['trajectory'] and a['outputs']==ident['outputs'] and b['truth_sha256']==ident['truth_sha256']
        assert b['indices']==list(range(425,445))
    rows=read('trajectory_scores.csv');d=read('delta_by_trajectory.csv');replay=read('score_replay_checks.csv')
    assert len(rows)==648 and len(d)==1728 and all(float(r['maxabs'])<=1e-12 for r in replay)
    index={(r['split'],r['trajectory'],int(r['step']),r['segment'],r['channel']):r for r in rows};assert len(index)==648
    error=0.;pairerror=0.
    for split in ['train','development']:
        for trajectory in CFG['train_files' if split=='train' else 'development_files']:
            for step in CFG['steps']:
                for ch in ['overall','u','v']:
                    pieces=[index[split,trajectory,step,s,ch] for s in ['full','overlap','extra']]
                    for metric in METRICS:error=max(error,abs(float(pieces[0][metric])-.75*float(pieces[1][metric])-.25*float(pieces[2][metric])))
                    for r in pieces:pairerror=max(pairerror,abs(float(r['crps'])-float(r['member_truth'])+float(r['member_pair_half'])))
    assert error<1e-12 and pairerror<1e-12
    di={(r['split'],r['trajectory'],int(r['step']),int(r['denoise']),r['region'],r['channel']):r for r in d};assert len(di)==1728
    for r in d:
        assert int(r['elements'])==32*(2 if r['channel']=='overall' else 1)*(15 if r['region']=='observed' else 5)*32*64
        assert float(r['rms'])>=0 and (int(r['step'])!=0 or float(r['rms'])==0)
    summary=read('score_summary.csv');ds=read('delta_summary.csv');assert len(summary)==54 and len(ds)==144
    maxsummary=0.
    for s in summary:
        rr=[r for r in rows if all(r[k]==s[k] for k in ['split','step','segment','channel'])];assert len(rr)==12
        for metric in METRICS:
            value=np.mean([float(r[metric]) for r in rr]);diff=np.mean([float(r[metric])-float(index[r['split'],r['trajectory'],0,r['segment'],r['channel']][metric]) for r in rr])
            maxsummary=max(maxsummary,abs(value-float(s[metric])),abs(diff-float(s[metric+'_delta'])))
    for s in ds:
        rr=[r for r in d if all(r[k]==s[k] for k in ['split','step','denoise','region','channel'])];assert len(rr)==12
        maxsummary=max(maxsummary,abs(np.mean([float(r['rms']) for r in rr])-float(s['rms_mean'])))
    assert maxsummary<1e-12
    for w in CFG['weights']:assert digest(ROOT/w['path'])==w['sha256']
    files=[p for base in [ROOT/'research/conditioning_failure_audit',OUT] for p in base.rglob('*') if p.is_file()]
    dump(OUT/'VERIFICATION.json',dict(passed=True,trajectories=24,score_rows=648,delta_rows=1728,replay_rows=len(replay),event_order_checked=True,training_selection_checked=True,checkpoint_hashes_checked=3,full_identity_maxabs=error,crps_parts_maxabs=pairerror,summary_maxabs=maxsummary,optimizer_updates=0,nonzero_forward_groups=48,reused_groups=24,prediction_cache_bytes=0,delivery_bytes_before_verification=sum(p.stat().st_size for p in files),old_dependencies_unchanged=True,utc=time.time()))
    print('PASS fixed manifests, event order, all scores/segments/deltas/summaries, checkpoint hashes and old dependencies')

if __name__=='__main__':main()
