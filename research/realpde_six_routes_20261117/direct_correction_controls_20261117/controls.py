"""Small direct field controls of frozen DiPred; no feedback in this task."""
from pathlib import Path
import os, sys, json, time, math, argparse, importlib.util
os.environ['TORCH_ALLOW_TF32_CUBLAS_OVERRIDE']='0'
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
W=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(W/'error_structure_20261117'))
import local_learn as local
from common import setup,load,save,dump,digest,DATA,MANIFEST,SHARED,records,read
st=local
OUT=W/'runs/direct_correction_controls_20261117'
SOURCE=W/'runs/error_structure_20261117'
ACTUAL=W/'runs/active_observation_20261117'
FEATURES=W/'runs/goal_task_error_20261117'
KINDS=['dense_F','additive_XP','rescrct_original','rescrct_block','refine_two']
REFERENCES=['base','warp_F_last','additive_F_last']

def explicit_module(name,path):
    spec=importlib.util.spec_from_file_location(name,str(path));mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod
original=explicit_module('direct_controls_original_models',W.parent/'Shared/models.py')
offline=explicit_module('direct_controls_offline_models',W.parent/'Shared/crossfit_corrector.py')

class DenseF(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Conv2d(32,14,1),nn.SiLU(),nn.Conv2d(14,14,3,padding=1),nn.SiLU(),nn.Conv2d(14,40,1));nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
    def forward(self,x,p,z):
        delta=.25*torch.tanh(self.net(z));delta=F.interpolate(delta,p.shape[-2:],mode='bilinear',align_corners=True).reshape_as(p);return p+delta

class AdditiveXP(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Conv3d(4,8,3,padding=1),nn.SiLU(),nn.Conv3d(8,8,3,padding=1),nn.SiLU(),nn.Conv3d(8,2,3,padding=1));nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
    def forward(self,x,p,z=None):return p+self.net(torch.cat([x,p],1))

class ColdResCrct(original.Corrector):
    def forward(self,x,p,z=None):
        zero=torch.zeros_like(p);return p+super().forward(x,p,zero,zero)

class OfflineResCrct(offline.BlockResidualCorrector):
    def forward(self,x,p,z=None):return super().forward(x,p)

class RefineTwo(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Conv3d(6,7,3,padding=1),nn.SiLU(),nn.Conv3d(7,7,3,padding=1),nn.SiLU(),nn.Conv3d(7,2,3,padding=1));nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
    def forward(self,x,p,z=None):
        q=p
        for _ in range(2):
            a=F.avg_pool3d(torch.cat([x,p,q],1),(1,2,2));delta=self.net(a);q=q+.5*F.interpolate(delta,p.shape[-3:],mode='trilinear',align_corners=False)
        return q

def factory(name):return dict(dense_F=DenseF,additive_XP=AdditiveXP,rescrct_original=ColdResCrct,rescrct_block=OfflineResCrct,refine_two=RefineTwo)[name]()
def common_projection(device='cuda'):return load(SOURCE/'local/warp_F_last.pt',device)['projection']
def project_feature(f,pr):return local.apply_projection(f.reshape(len(f),-1,4,8),pr)
def mix(head,x,p,z,gain):return p if gain==0 else p+gain*(head(x,p,z)-p)
def mse(q,y):return local.mse(q,y)
def aggregate(value,meta):return local.aggregate(value,meta)

def data(split):
    src=load(SOURCE/f'{split}.pt');p=load(ACTUAL/f'{split}.pt')['p'];f=load(FEATURES/f'{split}.pt')['features']['F_last']
    return dict(x=src['x'].cuda(),y=src['y'].cuda(),p=p.cuda(),z=project_feature(f.cuda(),common_projection()),meta=src['meta'])

class Predictor:
    def __init__(self,name='base',initial=False):
        self.name=name;self.net=st.model();self.head=None;self.norm=st.norm();self.gain=1.;self.proj=None
        if name in REFERENCES[1:]:
            ck=load(SOURCE/'local'/f'{name}.pt','cuda');self.cfg=ck['summary'];self.proj=ck['projection'];self.head=local.Head(self.cfg['style']).cuda().eval();self.head.load_state_dict(ck['model']);self.gain=self.cfg['shrink']
        elif name in KINDS:
            self.head=factory(name).cuda().eval();self.proj=common_projection() if name=='dense_F' else None
            if not initial:
                ck=load(OUT/f'{name}.pt','cuda');self.head.load_state_dict(ck['model']);self.gain=ck['summary']['gain']
        elif name!='base':raise ValueError(name)
        if self.head is not None:self.head.requires_grad_(False)
    def correct(self,x,p,f,force_gain=None):
        gain=self.gain if force_gain is None else force_gain
        if self.head is None or gain==0:return p
        if self.name in REFERENCES[1:]:
            z=project_feature(f['F_last'],self.proj);return self.head(z,p,gain)
        z=project_feature(f['F_last'],self.proj) if self.proj is not None else None
        return mix(self.head,x,p,z,gain)
    @torch.no_grad()
    def predict(self,history,normalized=False):
        history=torch.as_tensor(history)
        if history.shape!=(1,2,20,32,64) or not torch.isfinite(history).all():raise ValueError('Expected finite observed history [1,2,20,32,64]')
        x=history.to(device='cuda',dtype=torch.float32)
        if not normalized:x=(x-self.norm['mean'])/self.norm['std']
        p,f=st.capture(self.net,x);q=self.correct(x,p,f)
        return dict(normalized=q,physical=q*self.norm['std']+self.norm['mean'],base=p)

@torch.no_grad()
def error_vector(head,d,gain):
    return torch.cat([mse(mix(head,d['x'][i:i+8],d['p'][i:i+8],d['z'][i:i+8],gain),d['y'][i:i+8]) for i in range(0,len(d['meta']),8)])

def fit():
    d=data('train');cal=data('calibration');units=st.norm()['std'].square().mean();base=aggregate(mse(cal['p'],cal['y']),cal['meta']);summaries={}
    for name in KINDS:
        dest=OUT/f'{name}.pt'
        if dest.exists():
            ck=load(dest);assert ck['summary']['protocol_sha256']==digest(Path(__file__).with_name('PROTOCOL.md'));summaries[name]=ck['summary'];continue
        torch.manual_seed(42);head=factory(name).cuda();opt=torch.optim.AdamW(head.parameters(),lr=3e-4,weight_decay=1e-5);g=torch.Generator(device='cuda').manual_seed(42);steps=1000 if name=='refine_two' else 2000;logs=[];start=time.perf_counter();allix=[]
        for step in range(steps):
            ix=torch.randint(len(d['meta']),(8,),device='cuda',generator=g);allix.append(ix.cpu())
            lr=1e-6+(3e-4-1e-6)*.5*(1+math.cos(math.pi*step/steps))
            for group in opt.param_groups:group['lr']=lr
            opt.zero_grad();q=head(d['x'][ix],d['p'][ix],d['z'][ix]);loss=mse(q,d['y'][ix]).mean()/units;assert torch.isfinite(loss)
            loss.backward();grad=nn.utils.clip_grad_norm_(head.parameters(),1.);assert torch.isfinite(grad);opt.step()
            if step==0 or (step+1)%250==0:
                row=dict(step=step+1,loss=float(loss),seconds=time.perf_counter()-start);logs.append(row);print('TRAIN',name,row,flush=True);dump(OUT/f'{name}_training.json',logs)
        torch.cuda.synchronize();training_seconds=time.perf_counter()-start;head.eval()
        options={gain:aggregate(error_vector(head,cal,gain),cal['meta']) for gain in [0.,.25,.5,1.]};gain=min(options,key=options.get)
        sm=dict(name=name,gain=gain,calibration_mse=options[gain],calibration_base=base,options=options,parameters=sum(p.numel() for p in head.parameters()),inactive_parameters=4*24*27 if name=='rescrct_original' else 0,training_seconds=training_seconds,steps=steps,train_calls=steps*(2 if name=='refine_two' else 1),seed=42,train_files=sorted({m['file'] for m in d['meta']}),calibration_files=sorted({m['file'] for m in cal['meta']}),train_raw_mse=aggregate(error_vector(head,d,1.),d['meta']),protocol_sha256=digest(Path(__file__).with_name('PROTOCOL.md')))
        save(dict(model=head.state_dict(),summary=sm),dest);save(torch.stack(allix),OUT/f'{name}_training_indices.pt');summaries[name]=sm;print('SELECTED',sm,flush=True)
        del head,opt
    winner=min(summaries,key=lambda k:summaries[k]['calibration_mse'])
    dump(OUT/'selection.json',dict(models=summaries,winner=winner,criterion='calibration physical MSE only; no dev evaluation',protocol_sha256=digest(Path(__file__).with_name('PROTOCOL.md')),source_hashes={'models.py':digest(W.parent/'Shared/models.py'),'crossfit_corrector.py':digest(W.parent/'Shared/crossfit_corrector.py'),'warp_F_last.pt':digest(SOURCE/'local/warp_F_last.pt')},model_hashes={k:digest(OUT/f'{k}.pt') for k in KINDS}))
    print('WINNER',winner,flush=True)

@torch.no_grad()
def evaluate(split):
    src=load(SOURCE/f'{split}.pt');actual=load(ACTUAL/f'{split}.pt')['p'];feat=load(FEATURES/f'{split}.pt')['features']['F_last'];rows=[];saved={};maxpoint=0.
    sys.path.insert(0,str(W/'goal_task_error_20261117'));from task_error import Tasks,TASKS
    tasks=Tasks();n=st.norm()
    for name in REFERENCES+KINDS:
        api=Predictor(name);issued=[]
        for i,meta in enumerate(src['meta']):
            out=api.predict(src['x'][i:i+1],normalized=True);p,q=out['base'],out['normalized'];maxpoint=max(maxpoint,float((p[0].cpu()-actual[i]).abs().max()));issued.append(q[0].cpu())
            # Prediction is finalized before the target is used for any score.
            y=src['y'][i:i+1].cuda();raw=api.correct(src['x'][i:i+1].cuda(),p,dict(F_last=feat[i:i+1].cuda()),force_gain=1.);es=mse(q,y);base=mse(p,y);rawerr=mse(raw,y)
            jerr=tasks.values((q*n['std']+n['mean']).double())-tasks.values((y*n['std']+n['mean']).double())
            row=dict(**meta,method=name,mse=float(es[0]),base_mse=float(base[0]),gain1_mse=float(rawerr[0]),harm=int(es[0]>base[0]))
            for j,t in enumerate(TASKS):row[t+'_mse']=float(jerr[0,j]**2)
            rows.append(row)
        saved[name]=torch.stack(issued);del api
    assert maxpoint==0,maxpoint
    folder=OUT/split;folder.mkdir(exist_ok=True);df=pd.DataFrame(rows);df.to_csv(folder/'windows.csv',index=False);cols=[c for c in df if c not in ['file','group','start','method']]
    tr=df.groupby(['method','file','group'])[cols].mean().reset_index();tr.to_csv(folder/'trajectories.csv',index=False);sm=tr.groupby('method')[cols].mean();sm['gain_percent']=(1-sm.mse/sm.loc['base','mse'])*100;sm.to_csv(folder/'summary.csv');save(saved,folder/'issued.pt');dump(folder/'cache_check.json',dict(point_replay_max_error=maxpoint,actual_batch1=True));print('EVALUATED',split,sm[['mse','gain1_mse','gain_percent']].to_string(),flush=True)

if __name__=='__main__':
    setup();OUT.mkdir(parents=True,exist_ok=True);ap=argparse.ArgumentParser();ap.add_argument('action',choices=['fit','evaluate']);ap.add_argument('--split',default='development');args=ap.parse_args()
    fit() if args.action=='fit' else evaluate(args.split)
