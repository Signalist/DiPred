"""Audit no direct X in F/P/PF, trajectory timing and exported metrics."""
import csv
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT/'fair_f_p_pf_xf_sharedref'))
from evaluate import OUT,ARMS,SEEDS,INPUTS,input_batch,load_model
from clean_reliability_common import configure,manifest
DEST=ROOT/'github/DiPred/results/reliability/fair_f_p_pf_xf_sharedref'

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()

def main():
    configure();checks={}
    for arm in ARMS:
        x,p,f,mask,reference=input_batch('development',0,arm)
        cached=torch.load(INPUTS/'development'/'0000.pt',map_location='cuda',weights_only=False)['x']
        checks[f'{arm}_shared_last_frame_reference']=bool(torch.equal(reference[:,:,-1:],cached[:,:,-1:]))
        model,_=load_model(arm,42)
        tau=torch.tensor([.1,.5,.9],device='cuda')
        with torch.no_grad():
            actual=model.quantiles(x,p,f,mask,tau,reference)
            zero_ref=model.quantiles(x,p,f,mask,tau,torch.zeros_like(reference))
        checks[f'{arm}_reference_used_in_output']=bool(torch.allclose(actual-zero_ref,reference[:,:,-1:].unsqueeze(0).expand_as(actual),rtol=2e-5,atol=2e-6))
        checks[f'{arm}_direct_x_mask']=bool((x==0).all()) if arm!='XF' else bool((x!=0).any())
        checks[f'{arm}_p_mask']=bool((p==0).all()) if arm in ('F','XF') else bool((p!=0).any())
        checks[f'{arm}_f_mask']=bool((f==0).all()) if arm=='P' else bool((f!=0).any())
    results=[]
    for seed in SEEDS:
        states=[load_model(arm,seed)[1] for arm in ARMS]
        checks[f'matched_initialization_{seed}']=len({s['init_sha256'] for s in states})==1
        for arm in ARMS:
            path=OUT/'evaluated'/f'{arm}_{seed}.pt'
            part=torch.load(path,map_location='cpu',weights_only=False);events=part['events']
            checks[f'{arm}_{seed}_causal']=len(events)==84 and all(events[2*i]['event']=='issued' and events[2*i+1]['event']=='revealed' and events[2*i]['start']==events[2*i+1]['start']==20*i and events[2*i]['time']<=events[2*i+1]['time'] for i in range(42))
            checks[f'{arm}_{seed}_complete']=len(part['windows'])==123 and len(part['frames'])==123 and part['observations']==20152320
            cdf=part['cdf'].numpy();checks[f'{arm}_{seed}_cdf']=bool(np.isfinite(cdf).all() and (np.diff(cdf,axis=2)>=-1e-6).all() and cdf.min()>=-1e-6 and cdf.max()<=1+1e-6)
            checks[f'{arm}_{seed}_finite']=all(np.isfinite(w['values']).all() for w in part['windows'])
    meta=json.loads((DEST/'manifest.json').read_text())
    checks['file_hashes']=all(sha(DEST/p)==h for p,h in meta['files_sha256'].items())
    checks['all_test_trajectories']=len(manifest()['files']['test'])==12
    with (DEST/'per_window.csv').open(newline='') as f:
        w=list(csv.DictReader(f))
    with (DEST/'per_frame.csv').open(newline='') as f:
        frames=csv.DictReader(f)
        checks['frame_window_consistency']=len(w)==len(ARMS)*len(SEEDS)*3*41*12*2
        count=0
        for a,b in zip(w,frames):
            count+=1
            checks['frame_window_consistency'] &= all(a[k]==b[k] for k in ('method','seed','calibration','trajectory_id','window_start','channel')) and abs(float(a['crps'])-np.mean([float(b[f'crps_frame_{i:02d}']) for i in range(1,21)]))<1e-6
    checks['frame_count']=count==len(w)
    snap=np.load(DEST/'prediction_snapshot.npz')
    q=snap['q'];y=snap['y'];t=snap['levels'].reshape((1,499)+(1,)*(q.ndim-2))
    e=y[:,None]-q;pin=np.maximum(t*e,(t-1)*e)
    score=.004*(.5*(np.maximum(q[:,0]-y,0)+np.maximum(y-q[:,-1],0))+pin.sum(axis=1))
    scale=np.asarray(snap['normalization_std']).reshape(1,2,1,1,1)
    checks['snapshot_crps_independent']=bool(np.allclose(score*scale,snap['crps'],rtol=3e-5,atol=2e-6))
    checks={key:bool(value) for key,value in checks.items()}
    result={'passed':all(checks.values()),'checks':checks,'scope':'reused exploratory test; all arms share observed X last-frame reference; P and F depend on X'}
    (DEST/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    meta['files_sha256']['audit.json']=sha(DEST/'audit.json')
    (DEST/'manifest.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)
    assert result['passed']
if __name__=='__main__':main()
