"""Additional CSV alignment, quantile-grid and finite-value checks."""
import csv,json,math,re,sys
import torch
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
DEST=ROOT/'github/DiPred/results/reliability/fair_x_xp_xpf'
with (DEST/'per_window.csv').open(newline='') as f:rows=list(csv.DictReader(f))
with (DEST/'quantile_metrics.csv').open(newline='') as f:quantiles=list(csv.DictReader(f))
groups=defaultdict(set)
for r in rows:
    k=(r['method'],r['seed'],r['calibration'])
    groups[k].add((r['trajectory_id'],r['window_start'],r['channel']))
alignment=len(groups)==27 and len({frozenset(s) for s in groups.values()})==1 and all(len(s)==12*41*2 for s in groups.values())
grid=defaultdict(list)
for r in quantiles:
    grid[(r['method'],r['seed'],r['calibration'],r['channel'])].append(r)
quantile_ok=len(grid)==54
for entries in grid.values():
    quantile_ok &= len(entries)==499
    for i,r in enumerate(entries):
        quantile_ok &= abs(float(r['quantile'])-(i+1)/500)<1e-7
        quantile_ok &= 0<=float(r['empirical_coverage'])<=1 and float(r['pinball'])>=0
        if i:quantile_ok &= float(r['empirical_coverage'])>=float(entries[i-1]['empirical_coverage'])-1e-8
finite=all(math.isfinite(float(r[k])) for r in rows for k in ('crps','coverage90','width90','interval_score90','adjacent_crossing','any_crossing'))
with (DEST/'paired_by_window.csv').open(newline='') as handle: by_window=list(csv.DictReader(handle))
lookup={}
for r in rows:
    for metric in ('crps','interval_score90'):
        lookup[(r['calibration'],r['seed'],r['channel'],r['method'],r['window_start'],r['trajectory_id'],metric)]=float(r[metric])
trajectory_ids=sorted({r['trajectory_id'] for r in rows})
paired_ok=len(by_window)==3*4*3*3*2*41
for r in by_window:
    seeds=('42','123','2026') if r['seed']=='pooled' else (r['seed'],)
    channels=('u','v') if r['channel']=='both' else (r['channel'],)
    def value(method,trajectory):
        return sum(lookup[(r['calibration'],seed,channel,method,r['window_start'],trajectory,r['metric'])] for seed in seeds for channel in channels)/(len(seeds)*len(channels))
    expected=sum(value(r['baseline'],t)-value(r['comparison'],t) for t in trajectory_ids)/len(trajectory_ids)
    paired_ok &= abs(expected-float(r['mean_delta']))<1e-8
with (DEST/'summary.csv').open(newline='') as handle: summary=list(csv.DictReader(handle))
head_counts={int(r['head_parameters']) for r in summary}
backbone_counts={int(r['backbone_parameters']) for r in summary if r['method']!='X'}
parameter_ok=len(head_counts)==1 and len(backbone_counts)==1 and all(int(r['total_parameters'])==int(r['head_parameters'])+int(r['backbone_parameters']) and (r['method']!='X' or int(r['backbone_parameters'])==0) for r in summary)
config=json.loads((DEST/'config.json').read_text())
splits=config['splits'];names=list(splits)
no_duplicate_prefixes=all({r['sha256_first200'] for r in splits[a]}.isdisjoint({r['sha256_first200'] for r in splits[b]}) for i,a in enumerate(names) for b in names[i+1:])
cache_checks=[]
for file,tolerance in (('train_cache_audit.log',1e-6),('late_cache_audit.log',.003)):
    content=(DEST/file).read_text()
    numbers=[float(v) for v in re.findall(r'diff(?:erence)?\s+([0-9.eE+-]+)',content)]
    cache_checks.append(len(numbers)==3 and all(x<=tolerance for x in numbers))
sys.path.insert(0,str(ROOT/'fair_x_xp_xpf'))
from model import FairQuantileHead
torch.manual_seed(0)
model=FairQuantileHead().eval()
x=torch.randn(1,2,20,32,64);p=torch.randn_like(x);f=torch.randn(1,48,32,64);mask=torch.ones(1,2);tau=torch.tensor([.231])
with torch.no_grad():
    direct=model(x,p,f,mask,tau)
    grid_output=model.quantiles(x,p,f,mask,tau)[0]
consistent=bool(torch.allclose(direct,grid_output,rtol=2e-6,atol=2e-6))
checks={'parameter_accounting':parameter_ok,'no_shared_first200_hash_between_splits':no_duplicate_prefixes,'window_paired_means_match':paired_ok,'training_and_inference_quantiles_match':consistent,'training_cache_matches_frozen_backbone':all(cache_checks),'matched_exact_window_channel_keys':alignment,'complete_valid_quantile_grid':quantile_ok,'finite_window_metrics':finite}
result={'checks':checks,'passed':all(checks.values())}
(DEST/'extra_audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2));assert result['passed']
