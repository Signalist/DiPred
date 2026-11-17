"""Small target-free benefit router and synchronous fixed-budget replay."""
from pathlib import Path
import sys,os,json,argparse,time,hashlib
os.environ['TORCH_ALLOW_TF32_CUBLAS_OVERRIDE']='0'
W=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(W/'lowcost_joint_20261117'))
import study as s
import numpy as np
import pandas as pd
import torch
OUT=W/'runs/compute_allocation_20261117'
JOINT=W/'runs/lowcost_joint_20261117'
FINAL=JOINT/'conditional_scores/scorefit'
KINDS=['X','P','X+P','F_last','F_mid0','X+P+F_last']
QUOTAS=[0.,.25,.5,.75,1.]
KEYS=['file','start']
METRICS=['functional_crps','es','vs','marginal_crps','crps_tke_peak','crps_kinetic_mean','crps_enstrophy_grid_peak','crps_speed_max','crps_mvpe']

def frame(d):return pd.DataFrame(d['meta'])
def key_index(d):return pd.MultiIndex.from_frame(frame(d)[KEYS])
def align(df,d):
    a=df.set_index(KEYS).loc[key_index(d)].reset_index()
    assert len(a)==len(d['meta']) and not a.isna().any().any()
    return a
def descriptors(d,kind):return torch.cat([d['features'][k] for k in kind.split('+')],1).double()
def width(q,n):
    lo=.9*q[1]+.1*q[2];hi=.1*q[29]+.9*q[30]
    return float(((hi-lo)*n['std'][0]).mean())

@torch.no_grad()
def prepare(split):
    dest=OUT/f'{split}_observables.pt'
    if dest.exists():return
    d=s.load(s.SOURCE/f'{split}.pt');ids=list(range(len(d['meta'])))
    if split=='train':
        protocol=json.loads((FINAL/'protocol.json').read_text());ids=protocol['training_windows']
    pred=s.Predictor(library=False);feats={};widths=[];maxdiff={}
    for count,i in enumerate(ids):
        # This forecast function receives only the observed history.
        p,q,f=pred.issue(d['x'][i:i+1].cuda());widths.append(width(q,pred.n))
        for k,v in f.items():
            feats.setdefault(k,[]).append(v.cpu())
            maxdiff[k]=max(maxdiff.get(k,0),float((v.cpu()-d['features'][k][i:i+1]).abs().max()))
        if count%50==0:print('PREPARE',split,count,len(ids),flush=True)
    s.save(dict(meta=[d['meta'][i] for i in ids],source_index=ids,features={k:torch.cat(v) for k,v in feats.items()},width=np.array(widths)),dest)
    s.dump(OUT/f'{split}_observables_audit.json',dict(cached_feature_max_difference=maxdiff,windows=len(ids),targets_saved=False,predictor_inputs='observed history only'))

def basis_fit(x,dims):
    mu=x.mean(0);scale=x.std(0).clamp_min(.01);z=(x-mu)/scale
    _,_,v=torch.linalg.svd(z,full_matrices=False);b=v[:dims].T
    pcscale=(z@b).std(0).clamp_min(.01)
    return dict(mean=mu,scale=scale,basis=b,pcscale=pcscale)
def project(x,b):return ((x-b['mean'])/b['scale'])@b['basis']/b['pcscale']
def ridge(z,y,alpha):
    a=torch.cat([torch.ones_like(z[:,:1]),z],1);r=torch.eye(a.shape[1],dtype=a.dtype)*len(a)*alpha;r[0,0]=0
    return torch.linalg.solve(a.T@a+r,a.T@y.double())
def predict(d,m):
    z=project(descriptors(d,m['kind']),m['projection']);return (torch.cat([torch.ones_like(z[:,:1]),z],1)@m['weights']).numpy()

def choose(meta,priority,quota):
    """No targets: select only inside each already-available issue-time batch."""
    if not 0<=quota<=1:raise ValueError('quota outside [0,1]')
    if len(priority)!=len(meta) or not np.isfinite(priority).all():raise ValueError('invalid priorities')
    mask=np.zeros(len(meta),dtype=bool);m=pd.DataFrame(meta)
    for start,g in m.groupby('start',sort=True):
        order=sorted(g.index,key=lambda i:(-float(priority[i]),meta[i]['file']))
        mask[order[:int(np.floor(quota*len(order)+.5))]]=True
    return mask
def mean_trajectory(values,meta):
    return float(pd.DataFrame(dict(file=[m['file'] for m in meta],value=values)).groupby('file').value.mean().mean())
def paired_scores(split,d,mode='raw',expensive='smooth'):
    path=FINAL/split/'windows.csv' if mode=='raw' else JOINT/'common_calibration'/split/'windows.csv'
    df=pd.read_csv(path)
    if mode!='raw':df=df[df['mode']==mode]
    cheapname='ar__mixed' if mode=='raw' else 'lowcost'
    return align(df[df.method==cheapname],d),align(df[df.method==expensive],d)

def fit():
    train=s.load(OUT/'train_observables.pt');cal=s.load(OUT/'calibration_observables.pt')
    cfg=json.loads((FINAL/'fit.json').read_text())['config_index'];labels=pd.read_csv(FINAL/'train_grid.csv');labels=align(labels[labels.config==cfg],train)
    low,high=paired_scores('calibration',cal);cheap=low.functional_crps.to_numpy();costly=high.functional_crps.to_numpy()
    targets={'benefit':labels.functional_crps.to_numpy()-labels.base_functional_crps.to_numpy(),'risk':labels.functional_crps.to_numpy()}
    models={};rows=[];chosen={}
    for role,target in targets.items():
        y=torch.tensor(target,dtype=torch.float64)
        for kind in KINDS:
            best=None
            for dims in (4,16):
                b=basis_fit(descriptors(train,kind),dims);z=project(descriptors(train,kind),b)
                for alpha in (.01,.1,1.,10.,100.):
                    model=dict(kind=kind,dimensions=dims,alpha=alpha,projection=b,weights=ridge(z,y,alpha))
                    pr=predict(cal,model);losses=[]
                    for q in (.25,.5,.75):
                        mask=choose(cal['meta'],pr,q);losses.append(mean_trajectory(np.where(mask,costly,cheap),cal['meta']))
                    objective=float(np.mean(losses));row=dict(role=role,kind=kind,dimensions=dims,alpha=alpha,calibration_objective=objective)
                    rows.append(row)
                    if best is None or objective<best['calibration_objective']:
                        best=dict(**model,calibration_objective=objective)
            models[f'{role}/{kind}']=best
        key=min([k for k in models if k.startswith(role+'/')],key=lambda k:models[k]['calibration_objective']);chosen[role]=key
        print('SELECTED',role,key,{k:v for k,v in models[key].items() if k not in ('projection','weights')},flush=True)
    s.save(models,OUT/'models.pt');pd.DataFrame(rows).to_csv(OUT/'selection_grid.csv',index=False)
    s.dump(OUT/'selection.json',dict(chosen=chosen,models={k:{a:v for a,v in m.items() if a not in ('projection','weights')} for k,m in models.items()},training_windows=len(labels),training_trajectories=len(frame(train).file.unique()),model_sha256=s.digest(OUT/'models.pt'),protocol_sha256=s.digest(W/'compute_allocation_20261117/PROTOCOL.md'),selection_before_policy_evaluation=True,primary='functional_crps',seed=42))

def priorities(d,models,selection):
    values={k:predict(d,m) for k,m in models.items()}
    values['benefit']=values[selection['chosen']['benefit']];values['risk']=values[selection['chosen']['risk']]
    values['width']=np.asarray(d['width']);values['random']=random_priority([m['file'] for m in d['meta']],[420000+i for i in d['source_index']])
    return values

def random_priority(ids,seeds):
    # Stateless seed42 random control is identical in cached and actual batches.
    return np.array([int.from_bytes(hashlib.sha256(f'42|{name}|{seed}'.encode()).digest()[:8],'big')/2**64 for name,seed in zip(ids,seeds)])

def issue_masks(split):
    d=s.load(OUT/f'{split}_observables.pt');models=s.load(OUT/'models.pt');selection=json.loads((OUT/'selection.json').read_text());pr=priorities(d,models,selection);rows=[]
    for policy,values in pr.items():
        for q in QUOTAS:
            mask=choose(d['meta'],values,q)
            for i,m in enumerate(d['meta']):rows.append(dict(**m,policy=policy,quota=q,expensive=int(mask[i]),priority=float(values[i])))
    pd.DataFrame(rows).to_csv(OUT/f'{split}_issued_masks.csv',index=False)
    s.dump(OUT/f'{split}_issue_audit.json',dict(targets_or_scores_read=False,models_sha256=s.digest(OUT/'models.pt'),masks_sha256=s.digest(OUT/f'{split}_issued_masks.csv'),same_time_only=True,created_utc=pd.Timestamp.utcnow().isoformat()))
    print('ISSUED',split,len(rows),flush=True)

def evaluate(split):
    d=s.load(OUT/f'{split}_observables.pt');masks=pd.read_csv(OUT/f'{split}_issued_masks.csv');rows=[];oracle=[]
    modes=['raw'] if split=='calibration' else ['raw','fixed','online']
    for mode in modes:
        for action in ('smooth','secc3'):
            low,high=paired_scores(split,d,mode,action)
            for (policy,q),g in masks.groupby(['policy','quota'],sort=False):
                g=align(g,d);a=g.expensive.to_numpy(dtype=bool)
                for i,meta in enumerate(d['meta']):rows.append(dict(**meta,mode=mode,action=action,policy=policy,quota=q,expensive=float(a[i]),**{k:float(high[k][i] if a[i] else low[k][i]) for k in METRICS}))
            for q in QUOTAS:
                # These two diagnostic policies explicitly access labels after issue.
                a=choose(d['meta'],(low.functional_crps-high.functional_crps).to_numpy(),q)
                probability=np.zeros(len(d['meta']))
                for _,g in frame(d).groupby('start'):probability[g.index]=np.floor(q*len(g)+.5)/len(g)
                for policy,mask in [('oracle',a.astype(float)),('random_expected',probability)]:
                    for i,meta in enumerate(d['meta']):rows.append(dict(**meta,mode=mode,action=action,policy=policy,quota=q,expensive=float(mask[i]),**{k:float(mask[i]*high[k][i]+(1-mask[i])*low[k][i]) for k in METRICS}))
    df=pd.DataFrame(rows);df.to_csv(OUT/f'{split}_windows.csv',index=False)
    group=['mode','action','policy','quota'];tr=df.groupby(group+['file','group'])[['expensive']+METRICS].mean().reset_index();tr.to_csv(OUT/f'{split}_trajectories.csv',index=False)
    summary=tr.groupby(group)[['expensive']+METRICS].mean().reset_index();summary.to_csv(OUT/f'{split}_summary.csv',index=False)
    print(summary[(summary['mode']=='raw')&(summary.action=='smooth')&(summary.quota==.5)][['policy','expensive','functional_crps']].to_string(index=False),flush=True)

if __name__=='__main__':
    s.setup();OUT.mkdir(parents=True,exist_ok=True);a=argparse.ArgumentParser();a.add_argument('action',choices=['prepare','fit','issue','evaluate']);a.add_argument('--split',choices=['train','calibration','development','late','future']);args=a.parse_args()
    {'prepare':prepare,'fit':lambda _:fit(),'issue':issue_masks,'evaluate':evaluate}[args.action](args.split)
