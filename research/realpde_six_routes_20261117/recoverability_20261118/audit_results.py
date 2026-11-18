"""Independent numeric, quota, provenance and asset sanity checks."""
from pathlib import Path
import json,hashlib,tarfile
import numpy as np
import pandas as pd
W=Path(__file__).resolve().parent.parent;O=W/'runs/recoverability_20261118';checks=[]
def check(name,flag,**kwargs):
 assert bool(flag),(name,kwargs);checks.append(dict(name=name,passed=True,**kwargs))
p=pd.read_csv(O/'policies.csv');truth=pd.read_csv(O/'windows.csv');assert len(truth)==1224
for key,v in p.groupby(['split','seed','start','policy','quota']):
 n=len(v);expected=int(np.floor(key[4]*n+.5));actual=float(v['mask'].sum());check('quota',abs(actual-expected)<1e-8,split=key[0],seed=int(key[1]),policy=key[3],quota=float(key[4]))
 q=truth[(truth.split==key[0])&(truth.seed==key[1])&(truth.start==key[2])].set_index('file').loc[v.file];q=q.reset_index();expected_score=(1-v['mask'].to_numpy())*q.base_mse.to_numpy()+v['mask'].to_numpy()*q.warp_mse.to_numpy();check('composite_mse',np.max(np.abs(expected_score-v.mse.to_numpy()))<1e-10)
g=p.groupby(['split','seed','quota','policy','file'],as_index=False).mse.mean();s=pd.read_csv(O/'primary.csv')
for row in s.to_dict('records'):
 q=g[(g.split==row['split'])&(g.seed==row['seed'])&(g.quota==row['quota'])].pivot(index='file',columns='policy',values='mse');delta=float((q.random_expected-q.benefit).mean());check('primary_delta',abs(delta-row['delta'])<1e-15,split=row['split'],seed=int(row['seed']))
man=json.loads((W/'runs/clean_direct20_20261107/manifest.json').read_text());sets={s:{r['file'] for r in man['files'][s]} for s in ['train','calibration','development','test']}
for a in sets:
 for b in sets:
  if a<b:check('file_split_disjoint',not sets[a]&sets[b],left=a,right=b)
check('seed_set',set(truth.seed.unique())=={42,123,2025});check('raw_replay',json.loads((O/'reproduce_raw_audit.json').read_text())['passed']);check('causal_issue_replay',json.loads((O/'issue_audit.json').read_text())['passed']);check('simulation_feedback_order',json.loads((O/'pde_drift_audit.json').read_text())['passed'])
a=json.loads((O/'primary_assets/asset_manifest.json').read_text())
for name,expected in a['files'].items():check('asset_sha256',hashlib.sha256((O/'primary_assets'/name).read_bytes()).hexdigest()==expected,file=name)
with tarfile.open(O/'primary_assets.tar.gz','r:gz') as tar:
 for item in tar.getmembers():check('archive_safe',item.isfile() and '/' not in item.name and item.name not in ('.','..'),file=item.name)
(O/'final_audit.json').write_text(json.dumps(dict(passed=True,checks=len(checks),details=checks,limits=['The held-out Re trajectories and cross-scene data have historical use; not fresh independent confirmation.','Same frozen backbone for three training seeds.','Unmeasured operators are excluded from profiler FLOPs.','Validation is a consistency audit, not a proof of statistical generalization.']),indent=2)+'\n');print('AUDIT_PASS',len(checks),flush=True)
