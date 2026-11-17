"""Low-dimensional signed residual state, with explicit issue/reveal methods."""
from pathlib import Path
import os,sys,json,argparse,time
os.environ['TORCH_ALLOW_TF32_CUBLAS_OVERRIDE']='0'
import numpy as np
import pandas as pd
import torch
from torch import nn
W=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(W/'error_structure_20261117'))
import structure as st
from common import setup,load,save,dump,digest,norm
SOURCE=W/'runs/error_structure_20261117';OUT=W/'runs/online_signed_state_20261117';D=2*20*32*64

def group_ids(meta):
    d=pd.DataFrame(meta);return [list(g.sort_values('start').index) for _,g in d.groupby('file',sort=True)]
def feature_matrix(d):return torch.cat([d['features'][k] for k in ('X','P','F_last')],1).double()
def project(x,b):return ((x-b['mean'])/b['std'])@b['basis']/b['scale']
def projection(x):
    mu=x.mean(0);sd=x.std(0).clamp_min(.01);z=(x-mu)/sd;_,_,v=torch.linalg.svd(z,full_matrices=False);b=v[:16].T;scale=(z@b).std(0).clamp_min(.01)
    return dict(mean=mu,std=sd,basis=b,scale=scale)
def ridge(x,y,alpha):
    a=np.c_[np.ones(len(x)),x];reg=np.eye(a.shape[1])*alpha*len(a);reg[0,0]=0
    return np.linalg.solve(a.T@a+reg,a.T@y)
def linear(x,w):return np.r_[1.,x]@w if x.ndim==1 else np.c_[np.ones(len(x)),x]@w
def summary(values,meta):return pd.DataFrame(dict(file=[m['file'] for m in meta],loss=values)).groupby('file').loss.mean().mean()

@torch.no_grad()
def prepare():
    d=load(SOURCE/'train.pt');sd=norm()['std'];r=((d['y'].cuda()-d['p'].cuda())*sd).reshape(len(d['meta']),-1).double()
    b=torch.zeros(2,D,device='cuda',dtype=torch.float64);b[0,:D//2]=np.sqrt(2);b[1,D//2:]=np.sqrt(2)
    residual=r-(r@b.T/D)@b;gram=residual@residual.T/D;lam,u=torch.linalg.eigh(gram);modes=(u[:,-6:].T@residual)/lam[-6:].sqrt()[:,None];b=torch.cat([b,modes]);assert (b@b.T/D-torch.eye(8,device='cuda')).abs().max()<1e-7
    f=feature_matrix(d).cuda();proj=projection(f)
    model=dict(basis=b.cpu(),projection={k:v.cpu() for k,v in proj.items()},normalization=norm('cpu'),component_rms=r.reshape(-1,2,20,32,64).square().mean((0,2,3,4)).sqrt().cpu().numpy())
    save(model,OUT/'basis.pt')
    info={}
    for split in ('train','calibration','development','late','future'):
        a=load(SOURCE/f'{split}.pt');r=((a['y'].cuda()-a['p'].cuda())*sd).reshape(len(a['meta']),-1).double();c=r@b.T/D;mse=r.square().mean(1);z=project(feature_matrix(a).cuda(),proj)
        save(dict(meta=a['meta'],c=c.cpu().numpy(),base_mse=mse.cpu().numpy(),f=z.cpu().numpy()),OUT/f'{split}.pt')
        info[split]=dict(windows=len(c),trajectories=len(group_ids(a['meta'])),projected_oracle_gain_percent=float(c.square().sum(1).mean()/mse.mean()*100))
        print('PREPARED',split,info[split],flush=True)
    # Effect of forecast-side global u/v offset on the current pooled P feature.
    response=torch.zeros(2,2048,dtype=torch.float64,device='cuda');response[0,256:384]=1/float(sd[0,0,0,0,0]);response[1,384:512]=1/float(sd[0,1,0,0,0])
    model['feature_bias_response']=((response/proj['std'])@proj['basis']/proj['scale']).cpu().numpy();save(model,OUT/'basis.pt')
    dump(OUT/'preparation.json',dict(**info,basis_orthogonality_max_error=float((b@b.T/D-torch.eye(8,device='cuda')).abs().max()),protocol_sha256=digest(W/'online_signed_state_20261117/PROTOCOL.md'),train_source_sha256=digest(SOURCE/'train.pt')))

class TinyRNN(nn.Module):
    def __init__(self):
        super().__init__();self.gru=nn.GRU(26,16,batch_first=True);self.out=nn.Linear(16,8)
    def forward(self,x,h=None):
        z,h=self.gru(x,h);return self.out(z),h

def train_models():
    d=load(OUT/'train.pt');c=d['c'];f=d['f'];mean=c.mean(0);scale=c.std(0).clip(1e-5);z=(c-mean)/scale;ids=group_ids(d['meta']);pairs=[(a,b) for g in ids for a,b in zip(g[:-1],g[1:]) if d['meta'][b]['start']-d['meta'][a]['start']==20]
    prior=np.array([a for a,b in pairs]);after=np.array([b for a,b in pairs]);models=dict(mean=mean,scale=scale,static={},dynamics={},ar={})
    def dynamic(z):
        a=np.linalg.solve(z[prior].T@z[prior]+np.eye(8)*.1*len(prior),z[prior].T@z[after]).T
        a*=min(1.,.98/np.linalg.svd(a,compute_uv=False)[0]);innovation=z[after]-z[prior]@a.T;q=innovation.T@innovation/len(innovation)+np.eye(8)*1e-5
        return dict(a=a,q=q)
    models['dynamics']['constant']=dynamic(z)
    for alpha in (.1,1.,10.):
        w=ridge(f,z,alpha);models['static'][str(alpha)]=w;models['dynamics'][str(alpha)]=dynamic(z-linear(f,w))
    lag=np.zeros_like(z);ar_lag=np.zeros_like(z);available=np.zeros((len(z),1));gaps=np.zeros((len(z),1))
    for ix in ids:
        for a,b in zip(ix[:-1],ix[1:]):
            gap=(d['meta'][b]['start']-d['meta'][a]['start'])//20;lag[b]=z[a];available[b]=1;gaps[b]=gap;ar_lag[b]=np.linalg.matrix_power(models['dynamics']['constant']['a'],gap-1)@z[a]
    for alpha in (.1,1.,10.):models['ar'][str(alpha)]=ridge(np.c_[f,ar_lag,available],z,alpha)
    torch.manual_seed(42);net=TinyRNN().cuda();optimizer=torch.optim.AdamW(net.parameters(),lr=1e-3,weight_decay=1e-4);g=torch.Generator(device='cuda').manual_seed(42)
    length=max(map(len,ids));inputs=np.zeros((len(ids),length,26));target=np.zeros((len(ids),length,8));mask=np.zeros((len(ids),length,1))
    for j,ix in enumerate(ids):
        inputs[j,:len(ix)]=np.c_[f[ix],lag[ix],available[ix],gaps[ix]];target[j,:len(ix)]=z[ix];mask[j,:len(ix)]=1
    inputs=torch.tensor(inputs,dtype=torch.float32,device='cuda');target=torch.tensor(target,dtype=torch.float32,device='cuda');mask=torch.tensor(mask,dtype=torch.float32,device='cuda');weights=torch.tensor(scale/np.sqrt((scale**2).mean()),device='cuda').float();logs=[]
    for step in range(1000):
        ix=torch.randint(len(ids),(8,),generator=g,device='cuda');pred,_=net(inputs[ix]);loss=((pred-target[ix]).square()*weights.square()*mask[ix]).sum()/(mask[ix].sum()*8)
        optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(net.parameters(),1.);optimizer.step()
        if step%200==0:logs.append(dict(step=step,loss=float(loss)));print('RNN',logs[-1],flush=True)
    models['rnn']={k:v.cpu() for k,v in net.state_dict().items()};models['rnn_parameters']=sum(v.numel() for v in net.parameters());save(models,OUT/'models.pt');dump(OUT/'fit.json',dict(seed=42,train_windows=len(c),transitions=len(pairs),rnn_updates=1000,rnn_parameters=models['rnn_parameters'],rnn_logs=logs,stable_var_max_singular_values={k:float(np.linalg.svd(v['a'],compute_uv=False)[0]) for k,v in models['dynamics'].items()}))

class State:
    """Target absent from issue; reveal validates the forecast's full-block time."""
    def __init__(self,config,models):
        self.cfg=config;self.models=models;self.z=np.zeros(8);self.cov=np.eye(8);self.last=None;self.last_revealed=None;self.pending=None;self.history=[];self.rnn_h=None
        self.rnn=None
        if config['family']=='rnn':self.rnn=TinyRNN().eval();self.rnn.load_state_dict(models['rnn'])
    def static(self,f):
        key=self.cfg.get('static','constant');return np.zeros(8) if key=='constant' else linear(f,self.models['static'][key])
    def issue(self,f,start):
        if self.pending is not None:raise RuntimeError('Previous truth must be revealed or explicitly skipped before next issue')
        steps=0 if self.last is None else (start-self.last)//20
        feedback_age=0 if self.last_revealed is None else (start-self.last_revealed)//20
        if self.last is not None and (start<=self.last or (start-self.last)%20):raise ValueError('Nonmonotone/non-block issue')
        family=self.cfg['family'];m=self.static(f);mean=self.models['mean'];scale=self.models['scale'];estimate=np.zeros(8)
        if family=='base':estimate=np.zeros(8)
        elif family in ('rolling_bias','ema'):
            if self.history:
                if family=='rolling_bias':estimate=np.mean(self.history[-self.cfg['memory']:],0);estimate[2:]=0
                else:estimate=self.z.copy()
        elif family=='static':estimate=mean+scale*m
        elif family in ('var','rw_kalman','var_kalman','conditional_kalman'):
            key=self.cfg.get('static','constant');dyn=self.models['dynamics'][key]
            a=np.eye(8) if family=='rw_kalman' else dyn['a'];q=np.eye(8)*self.cfg.get('process',.1) if family=='rw_kalman' else dyn['q']
            for _ in range(steps):self.z=a@self.z;self.cov=a@self.cov@a.T+q
            estimate=mean+scale*(m+self.z)
        elif family=='linear_ar':
            lag=np.zeros(8) if not self.history else (self.history[-1]-mean)/scale
            if feedback_age>1:lag=np.linalg.matrix_power(self.models['dynamics']['constant']['a'],feedback_age-1)@lag
            estimate=mean+scale*linear(np.r_[f,lag,float(bool(self.history))],self.models['ar'][self.cfg['alpha']])
        elif family=='rnn':
            lag=np.zeros(8) if not self.history else (self.history[-1]-mean)/scale
            x=torch.tensor(np.r_[f,lag,float(bool(self.history)),float(feedback_age)],dtype=torch.float32)[None,None]
            with torch.no_grad():v,self.rnn_h=self.rnn(x,self.rnn_h)
            estimate=mean+scale*v[0,0].numpy()
        else:raise ValueError(family)
        self.pending=dict(start=start,static=m.copy());return estimate*self.cfg.get('shrink',1.)
    def reveal(self,coefficient,reveal_time):
        if self.pending is None or reveal_time<self.pending['start']+40:raise ValueError('Unissued/early feedback')
        c=np.asarray(coefficient).copy();family=self.cfg['family'];z=(c-self.models['mean'])/self.models['scale']-self.pending['static']
        if family=='ema':self.z=(1-self.cfg['rate'])*self.z+self.cfg['rate']*c
        elif family=='var':self.z=z
        elif family in ('rw_kalman','var_kalman','conditional_kalman'):
            r=np.eye(8)*self.cfg['noise'];gain=np.linalg.solve((self.cov+r).T,self.cov.T).T;self.z+=gain@(z-self.z);self.cov=(np.eye(8)-gain)@self.cov@(np.eye(8)-gain).T+gain@r@gain.T
        self.history.append(c);self.history=self.history[-8:];self.last=self.pending['start'];self.last_revealed=self.last;self.pending=None
    def skip_feedback(self):
        if self.pending is None:raise RuntimeError('No pending issue')
        self.last=self.pending['start'];self.pending=None

def configs():
    c=[dict(family='base',shrink=1.)]
    c += [dict(family='rolling_bias',memory=m) for m in (1,4,8)]
    c += [dict(family='ema',rate=r) for r in (.1,.5,1.)]
    c += [dict(family='static',static=k) for k in ('constant','.1','1.0','10.0')]
    c += [dict(family='var')]
    c += [dict(family='rw_kalman',process=p,noise=r) for p in (.01,.1,1.) for r in (.25,1.,4.)]
    c += [dict(family='var_kalman',noise=r) for r in (.25,1.,4.)]
    c += [dict(family='conditional_kalman',static=k,noise=r) for k in ('0.1','1.0','10.0') for r in (.25,1.,4.)]
    c += [dict(family='linear_ar',alpha=k) for k in ('0.1','1.0','10.0')]+[dict(family='rnn')]
    # Canonical float string of0.1, not the shorthand used above.
    for cfg in c:
        if cfg.get('static')=='.1':cfg['static']='0.1'
    return [dict(cfg,shrink=g) for cfg in c for g in ((1.,) if cfg['family']=='base' else (0.,.25,.5,1.))]

def shifted(d,scenario,basis):
    result={k:(v.copy() if isinstance(v,np.ndarray) else v) for k,v in d.items()};shift=np.zeros_like(d['c']);age=np.full(len(shift),-1,dtype=int)
    for index,ix in enumerate(group_ids(d['meta'])):
        cut=len(ix)//2
        for j,i in enumerate(ix):
            age[i]=j-cut
            if scenario=='natural' or j<cut:continue
            rate=1. if scenario=='step' else (j-cut+1)/(len(ix)-cut);offset=basis['component_rms']*.5*np.array([(-1.)**index,(-1.)**(index+1)])*rate
            shift[i,:2]=offset/np.sqrt(2);result['f'][i]+=offset@basis['feature_bias_response']
    result['c']=d['c']-shift;result['base_mse']=d['base_mse']+(shift**2).sum(1)-2*(shift*d['c']).sum(1);result['drift_coefficient']=shift;result['age']=age;return result

def simulate(d,cfg,models,intervene=None):
    estimates=np.zeros_like(d['c']);events=[]
    for ix in group_ids(d['meta']):
        state=State(cfg,models)
        for count,i in enumerate(ix):
            meta=d['meta'][i];estimates[i]=state.issue(d['f'][i],meta['start']);events.append(dict(**meta,event='issue',revealed=count))
            truth=d['c'][i].copy()
            if intervene is not None:truth=intervene(i,truth)
            state.reveal(truth,meta['start']+40);events.append(dict(**meta,event='reveal',revealed=count+1))
    mse=d['base_mse']+(estimates**2).sum(1)-2*(estimates*d['c']).sum(1)
    return estimates,mse,events

def calibrate():
    d=load(OUT/'calibration.pt');models=load(OUT/'models.pt');rows=[];selected={}
    for number,cfg in enumerate(configs()):
        _,error,_=simulate(d,cfg,models);loss=summary(error,d['meta']);rows.append(dict(id=number,**cfg,mse=loss))
        if cfg['family'] not in selected or loss<selected[cfg['family']]['mse']:selected[cfg['family']]=dict(config=cfg,mse=loss)
    pd.DataFrame(rows).to_csv(OUT/'calibration_grid.csv',index=False);dump(OUT/'selection.json',dict(models=selected,chosen=min([k for k in selected if k not in ('base','static')],key=lambda k:selected[k]['mse']),criterion='natural calibration trajectory-equal physical MSE',models_sha256=digest(OUT/'models.pt'),protocol_sha256=digest(W/'online_signed_state_20261117/PROTOCOL.md')))
    print('SELECTION',selected,flush=True)

def evaluate(split):
    raw=load(OUT/f'{split}.pt');models=load(OUT/'models.pt');basis=load(OUT/'basis.pt');selection=json.loads((OUT/'selection.json').read_text());rows=[]
    for scenario in ('natural','step','ramp'):
        d=shifted(raw,scenario,basis);issued={};events=None
        for name,candidate in selection['models'].items():
            est,errors,events=simulate(d,candidate['config'],models);issued[name]=est
            for i,m in enumerate(d['meta']):rows.append(dict(**m,scenario=scenario,method=name,age=int(d['age'][i]),mse=float(errors[i])))
        oracle=d['base_mse']-(d['c']**2).sum(1)
        for i,m in enumerate(d['meta']):rows.append(dict(**m,scenario=scenario,method='projected_oracle',age=int(d['age'][i]),mse=float(oracle[i])))
        save(dict(issued=issued,scenario=scenario,meta=d['meta'],events=events),OUT/f'{split}_{scenario}_issued.pt')
    df=pd.DataFrame(rows);df.to_csv(OUT/f'{split}_windows.csv',index=False);tr=df.groupby(['scenario','method','file','group']).mse.mean().reset_index();tr.to_csv(OUT/f'{split}_trajectories.csv',index=False);report=tr.groupby(['scenario','method']).mse.mean().unstack();report.to_csv(OUT/f'{split}_summary.csv');print(split,report.to_string(),flush=True)

if __name__=='__main__':
    setup();OUT.mkdir(parents=True,exist_ok=True);p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','fit','calibrate','evaluate']);p.add_argument('--split',default='development');args=p.parse_args();{'prepare':prepare,'fit':train_models,'calibrate':calibrate,'evaluate':lambda:evaluate(args.split)}[args.action]()
