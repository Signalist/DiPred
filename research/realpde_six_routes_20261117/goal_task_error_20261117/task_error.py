"""Empirical, goal-oriented errors. No closed-PDE certificate is asserted."""
from pathlib import Path
import os, sys, json, argparse, time
os.environ['TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'] = '0'
import numpy as np
import pandas as pd
import torch
W = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(W/'error_structure_20261117'))
import structure as st
from common import setup, load, save, dump, digest, DATA, MANIFEST, SHARED
OUT = W/'runs/goal_task_error_20261117'
SOURCE = W/'runs/error_structure_20261117'
ACTUAL = W/'runs/active_observation_20261117'
TASKS = ['kinetic', 'fluctuation', 'enstrophy', 'regional_u']
SPLITS = ['train', 'calibration', 'development', 'late', 'future']

def weights(meta, device='cuda'):
    files = [m['file'] for m in meta]
    counts = pd.Series(files).value_counts()
    return torch.tensor([1/counts[f]/len(counts) for f in files], device=device, dtype=torch.float64)

def coordinates():
    import h5py
    manifests = json.loads(MANIFEST.read_text())['files']
    rows = []
    for split in ['train', 'calibration', 'development']:
        for rec in manifests[split]:
            with h5py.File(DATA/rec['file'], 'r') as f:
                x, y, t = f['x'][::2, ::2], f['y'][::2, ::2], f['t'][:].astype(float)
                dx, dy, dt = np.median(np.diff(x, axis=1)), np.median(np.diff(y, axis=0)), np.median(np.diff(t))
                assert np.allclose(np.diff(x, axis=1), dx) and np.allclose(np.diff(y, axis=0), dy)
                assert np.allclose(np.diff(x, axis=0), 0) and np.allclose(np.diff(y, axis=1), 0)
                assert np.allclose(np.diff(t), dt, rtol=1e-4)
                rows.append(dict(file=rec['file'], split=split, dx=float(dx), dy=float(dy), dt=float(dt), t_start=float(t[0]), t_end=float(t[-1]), fields=list(f.keys()), attributes=dict(f.attrs)))
    assert len({round(r['dx'], 10) for r in rows}) == len({round(r['dy'], 10) for r in rows}) == 1
    assert all(r['file'] != '7575_0.h5' for r in rows)
    result = dict(dx=rows[0]['dx'], dy=rows[0]['dy'], dt=float(np.median([r['dt'] for r in rows])), rows=rows, units='stored coordinate/velocity/time units; no explicit SI metadata found', pressure='not measured', scope='Planar kinematics of 3D experimental flow; incomplete for closed Navier-Stokes/DWR assumptions', sources=['https://realpdecompetition.github.io/', 'https://huggingface.co/datasets/AI4Science-WestlakeU/RealPDE-Competition-Data/blob/main/README.md'], local_docs=['OLD/REALPDE/inform.md:194-209', 'OLD/REALPDE/realpde_t2_starting_kit_v6/docs/interface.md:74', 'OLD/REALPDE/realpde_t2_starting_kit_v6/docs/metrics.md:17'])
    dump(OUT/'coordinates.json', result)
    return result

class Tasks:
    def __init__(self, coords=None):
        self.coords = coords or json.loads((OUT/'coordinates.json').read_text())
        self.dx, self.dy, self.dt = [self.coords[k] for k in ['dx', 'dy', 'dt']]
    def curl(self, a):
        return (a[:, 1, :, 1:-1, 2:]-a[:, 1, :, 1:-1, :-2])/(2*self.dx) - (a[:, 0, :, 2:, 1:-1]-a[:, 0, :, :-2, 1:-1])/(2*self.dy)
    def centered(self, a):
        return a-a.mean(2, keepdim=True)
    def values(self, a):
        return torch.stack([.5*a.square().sum(1).mean((1,2,3)), .5*self.centered(a).square().sum(1).mean((1,2,3)), .5*self.curl(a).square().mean((1,2,3)), a[:,0,:,:,48:].mean((1,2,3))], 1)
    def expansion(self, p, r):
        linear = torch.stack([(p*r).sum(1).mean((1,2,3)), (self.centered(p)*self.centered(r)).sum(1).mean((1,2,3)), (self.curl(p)*self.curl(r)).mean((1,2,3)), r[:,0,:,:,48:].mean((1,2,3))], 1)
        quad = self.values(r); quad[:,3] = 0
        return linear, quad
    def residual_norms(self, r):
        return torch.stack([r.square().sum(1).mean((1,2,3)).sqrt(), self.centered(r).square().sum(1).mean((1,2,3)).sqrt(), self.curl(r).square().mean((1,2,3)).sqrt(), r[:,0,:,:,48:].square().mean((1,2,3)).sqrt()], 1)
    def bound(self, jp, rho):
        bound = (2*jp[:,:3]).clamp_min(0).sqrt()*rho[:,:3]+.5*rho[:,:3].square()
        return torch.cat([bound, rho[:,3:4]], 1)
    def defects(self, a):
        gx = torch.gradient(a, spacing=(self.dx,), dim=(4,))[0]
        gy = torch.gradient(a, spacing=(self.dy,), dim=(3,))[0]
        gt = torch.gradient(a, spacing=(self.dt,), dim=(2,))[0]
        div = gx[:,0]+gy[:,1]
        material = gt+a[:,0:1]*gx+a[:,1:2]*gy
        graddiv = torch.stack([torch.gradient(div, spacing=(self.dx,), dim=(3,))[0], torch.gradient(div, spacing=(self.dy,), dim=(2,))[0]], 1)
        # Seconds/length units are unverified; these are numerical descriptors only.
        return div, material, graddiv
    def descriptors(self, x, p):
        stats=[]; pd=None
        for a in [x,p]:
            div, material, graddiv = self.defects(a)
            stats.extend([self.values(a), div.mean((1,2,3))[:,None], div.square().mean((1,2,3)).sqrt()[:,None], material.mean((2,3,4)), material.square().mean((2,3,4)).sqrt(), self.curl(a).mean((1,2,3))[:,None], (a[:,:,1:]-a[:,:,:-1]).square().mean((2,3,4)).sqrt()])
            if a is p: pd=(material,graddiv)
        ordinary=torch.cat(stats,1)
        task=[]
        for r in [-pd[1],-pd[0],x[:,:,-1:].expand_as(p)-p]:
            linear, quad=self.expansion(p,r);task.extend([linear,quad[:,:3]])
        return ordinary,torch.cat([ordinary]+task,1)

@torch.no_grad()
def prepare(split):
    d=load(SOURCE/f'{split}.pt'); issued=load(ACTUAL/f'{split}.pt');n=st.norm(); tasks=Tasks()
    net=st.model();features={};jp=[];jy=[];lin=[];quad=[];rhon=[];maximum=0.
    for i in range(len(d['meta'])):
        x=d['x'][i:i+1].cuda();p,taps=st.capture(net,x)
        maximum=max(maximum,float((p[0].cpu()-issued['p'][i]).abs().max()))
        xp=(x*n['std']+n['mean']).double();pp=(p*n['std']+n['mean']).double()
        yy=(d['y'][i:i+1].cuda()*n['std']+n['mean']).double()
        desc=dict(X=st.pooled(x),P=st.pooled(p),F_last=taps['F_last'])
        desc['kinematic'],desc['task_weighted']=tasks.descriptors(xp,pp)
        for k,v in desc.items():features.setdefault(k,[]).append(v.cpu())
        a,b=tasks.expansion(pp,yy-pp);lin.append(a.cpu());quad.append(b.cpu());jp.append(tasks.values(pp).cpu());jy.append(tasks.values(yy).cpu());rhon.append(tasks.residual_norms(yy-pp).cpu())
    assert maximum==0, maximum
    # Controls also replay all four calls at batch1; the task remains correction of original P.
    for step in [1500,6000]:
        net.load_state_dict(load(SOURCE/'deterministic4'/f'model{step}.pt','cuda')['model'])
        values=[]
        for i in range(len(d['meta'])):
            _,taps=st.capture(net,d['x'][i:i+1].cuda());values.append(taps['F_last'].cpu())
        features[f'F_det4_{step}']=values
    result=dict(jp=torch.cat(jp),jy=torch.cat(jy),linear=torch.cat(lin),quad=torch.cat(quad),rho=torch.cat(rhon),features={k:torch.cat(v) for k,v in features.items()},meta=d['meta'])
    err=float((result['jy']-result['jp']-result['linear']-result['quad']).abs().max())
    assert err<1e-10,err
    save(result,OUT/f'{split}.pt')
    dump(OUT/f'{split}_cache_audit.json',dict(point_replay_max_error=maximum,taylor_max_error=err,source_sha256=digest(SOURCE/f'{split}.pt'),actual_point_sha256=digest(ACTUAL/f'{split}.pt'),windows=len(jp)))
    print('PREPARED',split,len(jp),'Taylor',err,flush=True)

def raw(d,kind):
    if kind=='task_stats':return torch.cat([d['features']['kinematic'][:,:4],d['jp']],1).double()
    return torch.cat([d['features'][k] for k in kind.split('+')],1).double()

def projection(z,w,rank):
    mean=(w[:,None]*z).sum(0);sd=((z-mean).square()*w[:,None]).sum(0).sqrt().clamp_min(1e-5)
    a=(z-mean)/sd;v,s,_=torch.linalg.svd(a*w.sqrt()[:,None],full_matrices=False)
    # Right singular vectors reconstructed, keeping deterministic full decomposition.
    basis=(v[:,:rank].T@(a*w.sqrt()[:,None]))/s[:rank,None].clamp_min(1e-12)
    scale=((a@basis.T).square()*w[:,None]).sum(0).sqrt().clamp_min(1e-5)
    return dict(mean=mean,sd=sd,basis=basis,scale=scale)

def project(z,pr):return (((z-pr['mean'])/pr['sd'])@pr['basis'].T/pr['scale']).clamp(-10,10)
def design(z):return torch.cat([torch.ones_like(z[:,:1]),z],1)
def ridge(z,y,w,alpha):
    a=design(z);reg=torch.eye(a.shape[1],device=a.device,dtype=a.dtype)*alpha;reg[0,0]=0
    return torch.linalg.solve(a.T@(a*w[:,None])+reg,a.T@(y*w[:,None]))
def apply(z,coef):return design(z)@coef

def prediction(d,ck,tasks,p=None):
    if ck['style']=='base':return d['jp'].clone()
    if ck['style']=='mean':return d['jp']+ck['offset']
    if ck['style']=='task_persistence':return d['features']['kinematic'][:,:4].double()
    if ck['style']=='task_affine':
        z=(raw(d,'task_stats')-ck['mean'])/ck['sd'];a=torch.stack([torch.ones_like(z[:,:4]),z[:,:4],z[:,4:]],1)
        return d['jp']+ck['shrink']*(a*ck['coef'][None]).sum(1)
    z=project(raw(d,ck['kind']),ck['projection']);value=apply(z,ck['coef'])
    if ck['style']=='direct':delta=value
    elif ck['style'].startswith('taylor'):
        delta=value[:,:4].clone()
        if ck['style']=='taylor':delta[:,:3]+=value[:,4:].clamp_min(0)
    else:
        if p is None:raise ValueError('POD task forecast requires physical issued field')
        rr=(value@ck['basis']).reshape_as(p)*ck['shrink']
        result=tasks.values(p+rr)
        if ck['style']=='pod_q':result+=ck['qmissing']*ck['shrink']
        return result
    return d['jp']+ck['shrink']*delta

def physical_fields(split):
    d=load(SOURCE/f'{split}.pt','cuda');p=load(ACTUAL/f'{split}.pt','cuda')['p'];n=st.norm()
    return (p*n['std']+n['mean']).double(),(d['y']*n['std']+n['mean']).double()

@torch.no_grad()
def fit():
    d=load(OUT/'train.pt','cuda');cal=load(OUT/'calibration.pt','cuda');w=weights(d['meta']);wc=weights(cal['meta']);tasks=Tasks()
    target=d['jy']-d['jp'];unit=(w[:,None]*target.square()).sum(0).clamp_min(1e-14)
    p,y=physical_fields('train');pc,yc=physical_fields('calibration');res=(y-p).flatten(1)
    a=res*w.sqrt()[:,None];lam,u=torch.linalg.eigh(a@a.T);basis=(u[:,-16:].T@a)/lam[-16:,None].clamp_min(1e-14).sqrt()
    coefficients=res@basis.T;rproj=(coefficients@basis).reshape_as(p)
    missing=tasks.expansion(p,y-p-rproj)[1];qmissing=(missing*w[:,None]).sum(0)
    save(dict(basis=basis.cpu(),qmissing=qmissing.cpu(),explained_fraction=float(lam[-16:].sum()/lam.sum())),OUT/'pod.pt')
    del res,a,rproj,p,y
    models={'base':dict(style='base'),'mean':dict(style='mean',offset=(target*w[:,None]).sum(0))};rows=[]
    specs=[('direct',kind,16) for kind in ['P','X+P','F_last','F_det4_1500','F_det4_6000','X+P+F_last','kinematic','task_weighted']]
    specs += [('direct','X+P+F_last',32),('taylor','X+P+F_last',16),('pod','X+P+F_last',16),('pod_q','X+P+F_last',16)]
    projections={}
    for style,kind,rank in specs:
        key=f'{style}/{kind}/{rank}';pk=(kind,rank)
        if pk not in projections:projections[pk]=projection(raw(d,kind),w,rank)
        pr=projections[pk];z=project(raw(d,kind),pr)
        labels=target if style=='direct' else torch.cat([d['linear'],d['quad'][:,:3]],1) if style=='taylor' else coefficients
        best=None
        for alpha in [.01,.1,1.,10.]:
            coef=ridge(z,labels,w,alpha)
            for shrink in [0.,.25,.5,1.]:
                ck=dict(style=style,kind=kind,rank=rank,projection=pr,coef=coef,alpha=alpha,shrink=shrink)
                if style.startswith('pod'):ck.update(basis=basis,qmissing=qmissing)
                pred=prediction(cal,ck,tasks,pc);loss=float((((pred-cal['jy']).square()/unit).mean(1)*wc).sum())
                rows.append(dict(model=key,alpha=alpha,shrink=shrink,score=loss))
                if best is None or loss<best['score']-1e-14:best=dict(ck,score=loss)
        models[key]=best
        print('SELECTED',key,best['alpha'],best['shrink'],best['score'],flush=True)
    chosen=min([k for k in models if k not in ['base','mean']],key=lambda k:models[k]['score'])
    # Frozen diagnostic: same selected Taylor weights and shrink, only Q omitted.
    tkey='taylor/X+P+F_last/16';models['taylor_no_q/X+P+F_last/16']=dict(models[tkey],style='taylor_no_q')
    save(dict(models=models,unit=unit,chosen=chosen),OUT/'models.pt')
    pd.DataFrame(rows).to_csv(OUT/'selection_grid.csv',index=False)
    dump(OUT/'selection.json',dict(chosen=chosen,models={k:{kk:vv for kk,vv in ck.items() if kk in ['style','kind','rank','alpha','shrink','score']} for k,ck in models.items()},fit_files=sorted({m['file'] for m in d['meta']}),calibration_files=sorted({m['file'] for m in cal['meta']}),readout_parameters={k:int(ck['coef'].numel()) if 'coef' in ck else 4 if k=='mean' else 0 for k,ck in models.items()},protocol_sha256=digest(Path(__file__).with_name('PROTOCOL.md')),train_sha256=digest(OUT/'train.pt'),calibration_sha256=digest(OUT/'calibration.pt'),model_sha256=digest(OUT/'models.pt')))
    print('OVERALL_CHOSEN',chosen,flush=True)

@torch.no_grad()
def evaluate(split):
    d=load(OUT/f'{split}.pt','cuda');ck=load(OUT/'models.pt','cuda');tasks=Tasks();p,y=physical_fields(split);rows=[];predictions={}
    for name,model in ck['models'].items():
        predictions[name]=prediction(d,model,tasks,p)
    pod=load(OUT/'pod.pt','cuda');r=y-p;rp=((r.flatten(1)@pod['basis'].T)@pod['basis']).reshape_as(p)
    predictions['oracle_pod16']=tasks.values(p+rp)
    predictions['oracle_linear']=d['jp']+d['linear']
    predictions['oracle_exact_taylor']=d['jp']+d['linear']+d['quad']
    for name,pred in predictions.items():
        e=pred-d['jy'];sq=e.square()/ck['unit']
        for i,meta in enumerate(d['meta']):
            row=dict(**meta,model=name,score=float(sq[i].mean()))
            for j,t in enumerate(TASKS):row.update({f'{t}_mse':float(e[i,j]**2),f'{t}_mae':float(e[i,j].abs()),f'{t}_bias':float(e[i,j]),f'{t}_score':float(sq[i,j])})
            rows.append(row)
    folder=OUT/split;folder.mkdir(exist_ok=True);df=pd.DataFrame(rows);df.to_csv(folder/'windows.csv',index=False)
    cols=[c for c in df.columns if c not in ['file','group','start','model']]
    tr=df.groupby(['model','file','group'])[cols].mean().reset_index();tr.to_csv(folder/'trajectories.csv',index=False);sm=tr.groupby('model')[cols].mean();sm['gain_percent']=(1-sm.score/sm.loc['base','score'])*100;sm.to_csv(folder/'summary.csv')
    save({k:v.cpu() for k,v in predictions.items()},folder/'predictions.pt')
    # A deterministic inequality holds with actual residual norms; estimated radii are empirical.
    actual=tasks.bound(d['jp'],d['rho']);cal=load(OUT/'calibration.pt','cuda')
    rho=cal['rho'].amax(0,keepdim=True);envelope=tasks.bound(d['jp'],rho.expand(len(p),-1));task_envelope=(cal['jy']-cal['jp']).abs().amax(0,keepdim=True).expand(len(p),-1)
    violations=(d['jy']-d['jp']).abs()-actual
    assert float(violations.max())<1e-10
    save(dict(oracle_bound=actual.cpu(),estimated_rho=rho.cpu(),empirical_bound=envelope.cpu(),task_envelope=task_envelope.cpu(),absolute_error=(d['jy']-d['jp']).abs().cpu()),folder/'bounds.pt')
    print('EVALUATED',split,sm[['score','gain_percent']].to_string(),flush=True)

if __name__=='__main__':
    setup();OUT.mkdir(parents=True,exist_ok=True);ap=argparse.ArgumentParser();ap.add_argument('action',choices=['coordinates','prepare','fit','evaluate']);ap.add_argument('--split',default='train',choices=SPLITS);args=ap.parse_args()
    {'coordinates':coordinates,'prepare':lambda:prepare(args.split),'fit':fit,'evaluate':lambda:evaluate(args.split)}[args.action]()
