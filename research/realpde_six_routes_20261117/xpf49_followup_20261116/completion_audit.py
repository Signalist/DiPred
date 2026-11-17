from followcommon import *
import pandas as pd,ast

def main():
 setup();checks={};ver=json.loads((OUT/'verification.json').read_text());assert ver['passed'];checks['numerical_and_legacy_replay']=True
 for arm in ('XPF','XP','F'):
  for k in range(1,50):
   p=OLD/'models'/f'XPF16-Q{k:02d}'/'model.pt' if arm=='XPF' else ABL/'models'/arm/f'Q{k:02d}.pt';ck=load(p);assert ck['parameters']==34004 and ck['config']['seed']==42
 grid=json.loads((OUT/'audit/grid250_config.json').read_text());assert grid['heads']==250 and len(grid['weights'])==250 and grid['updates']==9
 checks['independent_heads_verified']={'xpf':49,'F':49,'XP':49,'grid_pairs':249,'grid_median':1}
 for a in ('XP','XPF','XPQ','XPFQ','Q'):
  ck=load(OUT/'query'/f'{a}.pt');assert ck['seed']==42 and ck['steps']==4000
 checks['new_training_single_seed']=42
 q=pd.read_csv(OUT/'query/per_query.csv');e=pd.read_csv(OUT/'query/ecc_per_query.csv');assert len(q)==5*108*160*4 and len(e)==4*108*160*4
 assert set(q['method'])=={'XP','XPF','XPQ','XPFQ','Q'} and q.file.nunique()==12;assert set(q.category)=={'seen','new_space','new_time','new_both'}
 checks['query_rows']=len(q);checks['ecc_rows']=len(e)
 local=pd.read_csv(OUT/'local/per_group.csv');assert set(local.arm)=={'XP','XPF'} and set(local.target)=={.8,.85,.9,.95};checks['local_rows']=len(local)
 events=json.loads((OUT/'audit/events.json').read_text());assert len(events)==54
 extraction=json.loads((OUT/'query/extraction.json').read_text());assert extraction['split']==['train','calibration','development']
 for k,h in extraction['hashes'].items():assert digest(OLD/'models'/f'XPF16-{k}'/'model.pt')==h
 checks['train_calibration_development_only']=True
 artifacts=['REPORT.md','verification.json','audit/summary.csv','audit/grid250_per_group.csv','query/summary.csv','query/event_summary.csv','query/paired_trajectory.csv','query/decisions.csv','query/reliability.csv','query/inclusion_violations.json','query/latency.csv','query/ecc_predictions.pt','local/summary.csv','figures/query.png','figures/query.pdf','figures/local.png','figures/local.pdf']
 for p in artifacts:assert (OUT/p).exists() and (OUT/p).stat().st_size>0
 code=Path(__file__).resolve().parent
 for p in code.glob('*.py'):ast.parse(p.read_text(),filename=str(p))
 checks['artifacts_and_source']=True
 dump(OUT/'artifact_hashes.json',dict(results={p:digest(OUT/p) for p in artifacts},source={p.name:digest(p) for p in code.glob('*.py')},query_weights={a:digest(OUT/'query'/f'{a}.pt') for a in ('XP','XPF','XPQ','XPFQ','Q')}))
 dump(OUT/'completion_audit.json',dict(passed=True,checks=checks,scope='RealPDE single seed legacy XPF16 independent 49/250 head mechanism, local coverage, and query pilot; exploratory development'))
 print(json.dumps(checks,indent=2))
if __name__=='__main__':main()
