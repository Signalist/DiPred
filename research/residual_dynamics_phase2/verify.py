"""Audit deliverables against raw rows, prior release and saved native paths."""
from common import *
from causal import native
from metrics import energy_score
import sys
sys.path.insert(0,str(REPO))
from reliability.backbone import configure

def rows(name):
    with (OUT/(name+'.csv')).open() as f:return list(csv.DictReader(f))

def verify():
    configure()
    spec=read(OUT/'preregistered.json')
    for name,expected in spec['old_results'].items(): assert digest(REPO/name)==expected
    assert digest(REPO/'reliability/checkpoints/current.pt')==spec['backbone_sha256']
    old=read(OLD/'summary.json')
    mech=rows('mechanism_summary')
    forecast=rows('forecast_per_trajectory_window')
    summary=rows('forecast_summary')
    assert len(forecast)==768 and len(summary)==64
    for row in summary:
        take=[r for r in forecast if all(r[k]==row[k] for k in ('split','protocol','method','seed','state'))]
        assert len(take)==len({r['trajectory'] for r in take})==12
        for k in ('mae','mse','crps','crps_standardized','energy_standardized','event_brier','temporal_acf_rmse'):
            assert abs(np.mean([float(r[k]) for r in take])-float(row[k]))<1e-12
    for r in old:
        if r['arm'] in ('B','C') and r['condition']=='aligned':
            new=next(q for q in mech if q['split']==r['split'] and q['method']==r['arm'] and q['seed']=='42')
            assert abs(float(new['mse'])-r['mse'])<1e-12
    align=read(OUT/'alignment_audit.json')['entries']
    assert len(align)==6
    for r in align:
        assert digest(EXTRA/f"{r['split']}_{r['start']}_native.pt")==r['native_sha256']
        assert r['residual_replay_max_abs_error']<=2e-6
    events=[json.loads(line) for line in (OUT/'events.jsonl').read_text().splitlines()]
    for split in ('development','test'):
        for method in ('native','persistence_bias','history_only','current_fields','with_memory'):
            for seed in ((-1,) if method in ('native','persistence_bias') else SEEDS):
                seq=[(r['event'],r['window']) for r in events if r.get('mode')==method and r['seed']==seed and r['split']==split]
                assert seq[-4:]==[(ev,w) for w in range(2) for ev in ('causal_forecast_issued','complete20_revealed_state_updated')]
    assert len(rows('component_prediction_by_trajectory_lead_channel'))==24*19*2*5*2
    assert len(rows('coupling_permutation'))==24*2*10 and len(rows('auxiliary_physical_operator'))==24
    p=native('test')[0,0].double();f=aux('test');y=f['truth'][0,0].double()
    native_row=next(r for r in forecast if r['split']=='test' and r['method']=='native' and r['trajectory']==protocol()['records']['test'][0]['file'] and r['window']=='0')
    std=f['normalization']['std'].squeeze(0)
    # Independent explicit pairwise implementation on one full 20-frame example.
    n=len(p)
    a=(p-y).abs().mean()
    crps=a-sum((p[i]-p[:i]).abs().sum()/p[0].numel() for i in range(1,n))/n**2
    z=p/std
    es=(z-y/std).flatten(1).square().mean(1).sqrt().mean()-sum((z[i]-z[:i]).flatten(1).square().mean(1).sqrt().sum() for i in range(1,n))/n**2
    assert abs(float(crps)-float(native_row['crps']))<1e-12
    assert abs(float(es)-float(native_row['energy_standardized']))<1e-12
    dump(OUT/'verification.json',dict(passed=True,forecast_rows=len(forecast),old_files_checked=len(spec['old_results']),
        native_files_checked=6,phase1_B_C_exact_to_1e_minus12=True,score_macro_aggregates_checked=len(summary),
        complete_block_event_order=True,explicit_full_field_CRPS_and_ES=True,
        max_replay_difference=max(r['residual_replay_max_abs_error'] for r in align),
        max_float64_identity_error=max(r['decomposition_float64_max_abs_error'] for r in align)))
    print('PASS: preserved release, all six native caches, B/C reproduction, 768 forecasts, event order and pairwise scores')

if __name__=='__main__':verify()
