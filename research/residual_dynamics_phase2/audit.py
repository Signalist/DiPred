"""Verify all original caches and materialize native paths before reading truth."""
import sys
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import torch

from common import *
sys.path.insert(0,str(REPO))
from research.residual_dynamics.run import load_backbone, raw
from reliability.backbone import configure


@torch.no_grad()
def run():
    configure()
    spec=freeze_protocol()
    audit=read(OLD/'alignment.json')
    assert audit['passed']
    for file,expected in audit['source_files'].items():
        assert digest(file)==expected
    model,norm,_=load_backbone(SimpleNamespace(workspace=WORK))
    records=protocol()['records']
    ids=[{r['file'] for r in records[s]} for s in records]
    assert not any(ids[i]&ids[j] for i in range(len(ids)) for j in range(i))
    groups=[{r['group'] for r in records[s]} for s in records]
    assert not any(groups[i]&groups[j] for i in range(len(groups)) for j in range(i))
    source=read(OLD/'cache_manifest.json')
    entries=[]
    for split,rs in records.items():
        history,truths=[],[]
        for start in (400,420):
            oldpath=CACHE/f'{split}_{start}.pt'
            entry=next(v for v in source if Path(v['path'])==oldpath)
            assert digest(oldpath)==entry['sha256']
            ck=torch.load(oldpath,map_location='cpu',weights_only=False)
            assert ck['identity']['start']==start
            assert ck['identity']['seeds']==list(range(500000,500032))
            observed=raw(DATA,rs,start)
            assert hashlib.sha256(observed.cpu().numpy().tobytes()).hexdigest()==ck['history_sha256']
            h=(observed-norm['mean'])/norm['std']
            path=EXTRA/f'{split}_{start}_native.pt'
            if path.exists():
                saved=torch.load(path,map_location='cpu',weights_only=False)
                assert saved['identity']==ck['identity'] and saved.get('deterministic_tf32_off') is True
                p=saved['paths'].cuda()
                events('reuse_native_before_truth',split=split,start=start)
            else:
                # The old cache contains only P-Y. Do not inject Y even through
                # float-roundoff reconstruction into this causal experiment.
                p=torch.stack([model.rollout(h,s,sign=1)*norm['std']+norm['mean']
                               for s in range(500000,500032)],1)
                torch.save(dict(paths=p.cpu(),identity=ck['identity'],deterministic_tf32_off=True),path)
                events('native_all32_all20_saved_before_truth',split=split,start=start)
            target=raw(DATA,rs,start+20)
            events('complete20_truth_read_for_labels_audit',split=split,start=start)
            assert hashlib.sha256(target.cpu().numpy().tobytes()).hexdigest()==ck['truth_sha256']
            e=(p-target[:,None]).cpu()
            error=float((e-ck['residual']).abs().max())
            assert error<=2e-6, (split,start,error)
            # Bias/delta computed from issued P, independent of its labels.
            pd=p.double()
            b=pd.mean(1)-target.double()
            delta=pd-pd.mean(1,keepdim=True)
            decomposition=float(((pd-target[:,None].double())-(b[:,None]+delta)).abs().max())
            assert decomposition<1e-14
            entries.append(dict(split=split,start=start,history=[start,start+19],
                target=[start+20,start+39],pairs=19,cache_sha256=entry['sha256'],
                native_sha256=digest(path),residual_replay_max_abs_error=error,
                decomposition_float64_max_abs_error=decomposition))
            history.append(observed.cpu());truths.append(target.cpu())
            del p,pd,e,delta,b
        assert torch.equal(history[1],truths[0])
        geometry=[]
        for r in rs:
            with h5py.File(DATA/r['file'],'r') as f:
                dt=np.diff(f['t'][400:440]).astype(float)
                geometry.append(dict(file=r['file'],dt=float(np.median(dt)),
                    dt_min=float(dt.min()),dt_max=float(dt.max())))
        torch.save(dict(history=torch.stack(history,1),truth=torch.stack(truths,1),
                        geometry=geometry,normalization={k:v.cpu() for k,v in norm.items()}),EXTRA/f'{split}_fields.pt')
        print('AUDITED',split,len(rs),flush=True)
    dump(OUT/'alignment_audit.json',dict(passed=True,source_snapshots_hashes_checked=True,
        original_trajectory_and_Re_splits_disjoint=True,warm_history_equals_previous_complete_truth=True,
        all32_members_all20_frames_native_replayed=True,entries=entries))


if __name__=='__main__':
    run()
