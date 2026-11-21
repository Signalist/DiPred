"""Read-only completion audit: no fitting, model selection changes or experiment reruns."""
from collections import defaultdict
from experiment import *


def main():
    check_dependencies()
    frozen=json.loads((OUT/'FROZEN.json').read_text());assert frozen['config_sha256']==CONFIG_HASH
    for name,sha in frozen['weights'].items():assert digest(OUT/'weights'/name)==sha
    for name,sha in frozen['code'].items():
        if digest(ROOT/name)!=sha:
            fix=json.loads((OUT/'postfreeze_reporting_fix.json').read_text())['files'][name]
            assert fix['before']==sha and fix['after']==digest(ROOT/name)
            assert digest(OUT/'initial_aggregation'/Path(name).name)==sha
            assert Path(name).name in ('analyze.py','verify.py')
    pre=json.loads((OUT/'PREFLIGHT.json').read_text());assert pre['passed'] and pre['tests']==7
    for name in ('experiment.py','run.py','tests.py'):
        file=Path(__file__).resolve().parent/name
        assert digest(file)==pre['code'][str(file.relative_to(ROOT))]
    old=json.loads((ROOT/'results/conditional_innovation_pilot/selection_A.json').read_text());assert old['gate'] is False and old['neural_runs_allowed']==0
    done=json.loads((OUT/'training_complete.json').read_text());assert done['runs']==9 and done['steps']==9000
    schedule=json.loads((OUT/'batch_schedule.json').read_text())
    gen=torch.Generator(device='cpu').manual_seed(CFG['sampling_seed'])
    assert schedule==torch.randint(92,(1000,2),generator=gen).tolist()
    runs=json.loads((OUT/'runs.json').read_text());assert len(runs)==9
    assert {(r['kind'],r['seed']) for r in runs}=={(k,s) for k in ARMS for s in SEEDS}
    assert all(r['steps']==1000 and r['schedule_sha256']==digest(OUT/'batch_schedule.json') for r in runs)
    trace=read_csv(OUT/'training.csv');assert len(trace)==9000
    for k in ARMS:
        for s in SEEDS:
            rr=[r for r in trace if r['kind']==k and int(r['seed'])==s]
            assert [int(r['step']) for r in rr]==list(range(1,1001))
            assert all(np.isfinite(float(r[f])) for r in rr for f in ('crps_without_constant','gradient_l2','q_rms'))
    for s in SEEDS:
        initial=[]
        for k in ARMS:
            ck=torch.load(OUT/'weights'/f'{k}_{s}_0.pt',map_location='cpu',weights_only=False)
            initial.append(state_hash(ck['state']))
            assert int(torch.count_nonzero(ck['state']['encoder.4.weight']))==0
            assert int(torch.count_nonzero(ck['state']['encoder.4.bias']))==0
            assert initial[-1]==next(r['initial_state_sha256'] for r in runs if r['kind']==k and r['seed']==s)
        assert len(set(initial))==1
    candidates=read_csv(OUT/'candidates.csv');assert len(candidates)==45
    selections=read_csv(OUT/'selection.csv');assert selections==frozen['selection'] and len(selections)==18
    dev=read_csv(OUT/'development_scores.csv');assert len(dev)==45*24
    dev_index={(r['method'],r['seed'],r['step']):r for r in aggregate(dev,['method','seed','step'],METRICS)}
    for c in candidates:
        v=dev_index[c['kind']+'_'+c['seed'],c['seed'],c['step']]
        for m in METRICS:assert abs(float(c[m])-v[m])<1e-12
        assert digest(OUT/'weights'/c['file'])==c['sha256']
    for selected in selections:
        own=[dict(r,**{m:float(r[m]) for m in METRICS}) for r in candidates if r['kind']==selected['kind'] and r['seed']==selected['seed']]
        best=choose(own,selected['selection']=='deployment')
        assert best['step']==selected['step'] and best['file']==selected['file']
    scores=read_csv(OUT/'scores.csv');cells=read_csv(OUT/'detail.csv')
    methods=[('native','0'),('zero','0')]+[('phase3',str(s)) for s in SEEDS]+[(k+'_'+v,str(s)) for k in ARMS for v in ('scientific','deployment') for s in SEEDS]
    expected={(split,r['file'],str(start),method,seed) for split,starts in [('test',CFG['starts']),('extension',CFG['extension'])] for start in starts for r in RECORDS['test'] for method,seed in methods}
    keys=['split','trajectory','start','method','seed'];index={tuple(r[k] for k in keys):r for r in scores}
    assert len(scores)==len(expected)==6072 and set(index)==expected
    assert len(cells)==242880
    assert len({tuple(r[k] for k in keys+['lead','channel']) for r in cells})==len(cells)
    for r in aggregate(cells,keys,CELL_METRICS):
        b=index[tuple(r[k] for k in keys)]
        for m in CELL_METRICS:assert abs(float(b[m])-r[m])<1e-10,(m,b,r)
    for r in scores:
        assert all(np.isfinite(float(r[m])) for m in METRICS)
    # Reconstruct all headline paired intervals directly from raw scores.
    tr=aggregate(scores,['split','trajectory','method'],METRICS)
    lookup={(r['split'],r['trajectory'],r['method']):r for r in tr}
    for r in read_csv(OUT/'paired.csv'):
        left,right=r['contrast'].split(' minus ');split=r['split'];m=r['metric']
        def effective(method):
            rows=[s for s in selections if s['kind']+'_'+s['selection']==method]
            return 'native' if method=='zero' or rows and all(s['step']=='0' for s in rows) else method
        delta=[lookup[split,t['file'],effective(left)][m]-lookup[split,t['file'],effective(right)][m] for t in sorted(RECORDS['test'],key=lambda t:t['file'])]
        b=bootstrap(delta)
        assert abs(b['mean_delta']-float(r['mean_delta']))<1e-12
        assert abs(b['ci95'][0]-float(r['ci95_low']))<1e-12 and abs(b['ci95'][1]-float(r['ci95_high']))<1e-12
        assert b['improved']==int(r['improved'])
    events=[json.loads(line) for line in (OUT/'release_events.jsonl').read_text().splitlines()]
    assert len(events)==44
    audit=json.loads((OUT/'evaluation_audit.json').read_text());assert len(audit)==22
    for i,a in enumerate(audit):
        first,second=events[i*2:i*2+2]
        assert first['event']=='issued' and second['event']=='truth_revealed'
        assert first['start']==second['start']==a['start'] and first['utc']<=second['utc']
        assert first['corrections']==a['issued'] and second['truth_sha256']==a['truth_sha256']
        assert a['frozen_sha256']==digest(OUT/'FROZEN.json') and frozen['utc']<first['utc']
        assert digest(Path(a['cache']))==a['cache_sha256']
        for selected in selections:
            if selected['step']=='0':assert a['issued'][selected['kind']+'_'+selected['selection']+':'+selected['seed']]==a['issued']['native:0']
    cache=json.loads((ROOT/'results/conditional_innovation_pilot/cache_hashes.json').read_text())
    for name,sha in cache.items():assert digest(Path(name))==sha
    for name,sha in json.loads((PRIOR/'data_hashes.json').read_text()).items():assert digest(DATA/name)==sha
    for r in read_csv(OUT/'invariance.csv'):
        assert float(r['centered_max_abs'])<1e-6 and float(r['pair_max_abs'])<1e-6
    print('PASS: 9 runs, exactly9000 updates, equal initialization per seed and identical shared batches')
    print('PASS: 45 full development checkpoints; both selections independently reconstructed')
    print('PASS: 6072 complete block scores and242880 detailed rows, paired original-trajectory bootstrap')
    print('PASS: frozen-before-test and issued-before-truth ordering, zero aliases, member spread invariance')
    print('PASS: old artifacts, operator, weights, cache and original HDF5 hashes unchanged')

if __name__=='__main__':main()
