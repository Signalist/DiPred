"""Audit concrete deliverables against this round's frozen protocol."""
from common import *
import pandas as pd, ast

def main():
    setup();checks={};expected={'main':['X','P','F','XPF','Fdet','Pprojection','Frandom','Fshuffle']}
    expected.update({tag:['F','P'] for tag in ('width8','width32','labels25','labels50','symmetric')})
    weights=[]
    for tag,arms in expected.items():
        for a in arms:
            p=OUT/'risk'/tag/f'{a}.pt';ck=load(p);assert ck['seed']==42 and ck['steps']==2000;weights.append(p)
        df=pd.read_csv(OUT/'risk'/tag/'per_group.csv');assert set(arms).issubset(set(df.method));assert len(df)==(60480 if tag=='main' else 8640)
        assert set(df.start)==set(range(0,161,20));assert df.file.nunique()==12;assert len(df[['region','channel']].drop_duplicates())==20
    checks['risk_models_complete']=len(weights)
    for a in ('X','P','F','XPF'):
        p=OUT/'query'/f'{a}.pt';ck=load(p);assert ck['seed']==42 and ck['steps']==4000;weights.append(p)
    checks['query_models_complete']=4;checks['all_new_training_seeds']=[42]
    q=pd.read_csv(OUT/'query/per_query.csv');e=pd.read_csv(OUT/'query/ensemble_per_query.csv')
    for df in (q,e):
        assert len(df)==276480 and df.method.nunique()==4 and df.file.nunique()==12
        assert set(df.category)=={'seen','new_space','new_time','new_both'}
        for _,g in df.groupby(['method','file','start','channel','statistic']):assert len(g)==160
    checks['query_rows_complete']=len(q)+len(e)
    local=pd.read_csv(OUT/'local/per_group.csv');assert len(local)==69120
    assert set(local['shape'])=={'constant','P','F','XPF'} and set(local.strategy)=={'global','groups'} and set(local.target)=={.8,.85,.9,.95}
    checks['local_calibration_rows']=len(local)
    latency=pd.read_csv(OUT/'query/latency.csv');assert len(latency)==24 and set(latency.queries)=={1,8,32,160} and (latency.median_ms>0).all();checks['actual_latency_rows']=24
    decisions=pd.read_csv(OUT/'query/decisions_calibration_fpr10.csv');assert len(decisions)==192 and decisions.fpr.notna().all();checks['decisions_rows']=192
    verify=json.loads((OUT/'verification.json').read_text());assert verify['passed'] and verify['heldout_query_geometries_disjoint'] and verify['all_new_training_single_seed_42'];checks['numerical_and_causal_audits']=True
    assert 'test' not in json.loads((OUT/'data_audit.json').read_text())['splits'];checks['new_test_split_not_used']=True
    artifacts=['REPORT.md','verification.json','capacity_data_curves.csv','center_decomposition.csv','query/reliability_bins.csv','query/paired_trajectory.csv','query/inclusion_violations.json','local/summary.csv']
    artifacts += [f'figures/{name}.{ext}' for name in ('mechanism','local_tradeoff','query_generalization','query_latency') for ext in ('png','pdf')]
    for p in artifacts:assert (OUT/p).is_file() and (OUT/p).stat().st_size>0
    checks['report_tables_and_figures']=True
    code=Path(__file__).resolve().parent
    for p in code.glob('*.py'):ast.parse(p.read_text(),filename=str(p))
    assert (code/'README.md').is_file() and (code/'run.sh').is_file();checks['reproducible_entrypoints']=True
    checks['required_work_complete']=True
    dump(OUT/'artifact_hashes.json',dict(weights={str(p.relative_to(OUT)):digest(p) for p in weights},results={p:digest(OUT/p) for p in artifacts},source={p.name:digest(p) for p in code.glob('*.py')}))
    dump(OUT/'completion_audit.json',dict(passed=True,checks=checks,scope='single-seed RealPDE direction screening plus evidence-driven capacity/data/symmetry and local calibration followups; no independent confirmation claim'))
    print(json.dumps(checks,indent=2),flush=True)
if __name__=='__main__':main()
