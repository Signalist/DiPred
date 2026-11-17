"""Sparse observations at lead5, common low-rank update of future leads6:20."""
from pathlib import Path
import os,sys,json,argparse,time,hashlib
os.environ['TORCH_ALLOW_TF32_CUBLAS_OVERRIDE']='0'
import numpy as np
import pandas as pd
import torch
W=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(W/'lowcost_joint_20261117'))
import study as joint
from common import setup,load,save,dump,digest,norm,SHARED
SOURCE=W/'runs/error_structure_20261117';OUT=W/'runs/active_observation_20261117';BUDGETS=[0,1,2,4,8,16,32]
YY,XX=np.meshgrid(np.arange(2,32,4),np.arange(2,64,4),indexing='ij');SITES=np.stack([YY.flatten(),XX.flatten()],1);RANDOM_DRAWS=16

def observation_rows(a):
    # a [...,channel,time,y,x] -> [...,site,channel]
    return a[..., :,4,SITES[:,0],SITES[:,1]].movedim(-2,-1)

def balanced(meta,device='cuda'):
    df=pd.DataFrame(meta);counts=df.file.value_counts();return torch.tensor([1/(len(counts)*counts[m['file']]) for m in meta],device=device,dtype=torch.float64)

@torch.no_grad()
def prepare(split):
    d=load(SOURCE/f'{split}.pt');pred=joint.Predictor(library=False);n=pred.n
    if split=='train':
        r=(d['y'].cuda()-d['p'].cuda())*n['std'];weights=balanced(d['meta']);floor=(r.double().square().mean((2,3,4))*weights[:,None]).sum(0).sqrt()*.05;save(floor.cpu(),OUT/'scale_floor.pt')
    floor=load(OUT/'scale_floor.pt','cuda').float()[:,None,None,None];ps=[];scales=[];risks=[]
    for i in range(len(d['meta'])):
        p,q,_=pred.issue(d['x'][i:i+1].cuda());physical_sd=q.std(0,unbiased=False)*n['std'][0];scale=torch.maximum(physical_sd,floor);bias=(q.mean(0)-p)*n['std'][0];risk=physical_sd.square()+bias.square();ps.append(p.cpu());scales.append(scale.cpu());risks.append(observation_rows(risk).cpu())
        if i%100==0:print('PREPARE',split,i,len(d['meta']),flush=True)
    save(dict(p=torch.stack(ps),scale=torch.stack(scales),risk=torch.stack(risks),meta=d['meta']),OUT/f'{split}.pt');print('PREPARE_DONE',split,flush=True)

@torch.no_grad()
def fit():
    d=load(OUT/'train.pt','cuda');source=load(SOURCE/'train.pt','cuda');w=balanced(d['meta']);u=(((source['y']-d['p'])*norm()['std'])/d['scale']).flatten(1).double();weighted=u*w.sqrt()[:,None];gram=weighted@weighted.T;lam,v=torch.linalg.eigh(gram);basis=(v[:,-32:].flip(1).T@weighted).reshape(32,2,20,32,64);second=(w[:,None]*u.square()).sum(0).reshape(2,20,32,64);average_scale=(w[:,None]*d['scale'].flatten(1).double()).sum(0).reshape(2,20,32,64)
    save(dict(basis=basis.cpu(),second=second.cpu(),average_scale=average_scale.cpu(),floor=load(OUT/'scale_floor.pt'),meta=d['meta']),OUT/'model.pt');dump(OUT/'fit.json',dict(seed=42,windows=len(d['meta']),trajectories=len(set(m['file'] for m in d['meta'])),uncentered_second_moment=True,trajectory_balanced=True,rank32_explained_second_moment=float(lam[-32:].sum()/lam.sum()),protocol_sha256=digest(Path(__file__).with_name('PROTOCOL.md')),train_source_sha256=digest(SOURCE/'train.pt'),main_model_sha256=digest(SHARED)))
    print('FIT_DONE',float(lam[-32:].sum()/lam.sum()),flush=True)

def uniform(budget):
    if budget==0:return np.array([],dtype=int)
    nr=2**int(np.floor(np.log2(budget)/2));nc=budget//nr;yi=np.floor((np.arange(nr)+.5)*8/nr).astype(int);xi=np.floor((np.arange(nc)+.5)*16/nc).astype(int);return (yi[:,None]*16+xi[None,:]).flatten()

def spacefill():
    xy=SITES.astype(float);distance=np.full(128,np.inf);order=[int(np.argmin(((xy-[15.5,31.5])**2).sum(1)))]
    for _ in range(31):
        distance=np.minimum(distance,((xy-xy[order[-1]])**2).sum(1));distance[order]=-1;order.append(int(np.argmax(distance)))
    return np.array(order)

class Assimilator:
    def __init__(self,config,model=None):
        self.cfg=config;self.model=load(OUT/'model.pt','cuda') if model is None else model;self.rank=config['rank'];self.b=self.model['basis'][:self.rank];self.remainder=(self.model['second']-self.b.square().sum(0)).clamp_min(.01)*config['remainder'];self.pending=None;self.normalization=norm()
    def pack(self,scale):
        scale=scale.double();a=(self.b*scale)[:,:,5:].flatten(1);h=observation_rows(self.b*scale).permute(1,2,0);noise=observation_rows(self.remainder*scale.square());g=a@a.T/a.shape[1];return dict(a=a,h=h,noise=noise,g=g,irreducible=float((self.remainder[:,5:]*scale[:,5:].square()).mean()))
    def posterior(self,pack,indices):
        ids=torch.as_tensor(indices,device='cuda',dtype=torch.long);h=pack['h'][ids].reshape(-1,self.rank);noise=pack['noise'][ids].flatten();precision=torch.eye(self.rank,device='cuda',dtype=torch.float64)+(h.T/noise)@h;cov=torch.linalg.inv(precision);return cov,(cov@h.T)/noise
    def correction(self,pack,indices,observed_error):
        if len(indices)==0:return torch.zeros(self.rank,device='cuda',dtype=torch.float64)
        cov,gain=self.posterior(pack,indices);return self.cfg['gain']*(gain@observed_error.double().reshape(-1))
    def expected_gain(self,pack,indices):
        cov,_=self.posterior(pack,indices);beta=self.cfg['gain'];return float((2*beta-beta*beta)*torch.sum(pack['g']*(torch.eye(self.rank,device='cuda')-cov).T))
    def greedy(self,pack,maximum=32):
        cov=torch.eye(self.rank,device='cuda',dtype=torch.float64);selected=[];gains=[];h=pack['h'];noise=pack['noise'];g=pack['g']
        for _ in range(maximum):
            a=h@cov;innovation=a@h.transpose(1,2)+torch.diag_embed(noise);sol=torch.linalg.solve(innovation,a);benefit=((a@g)*sol).sum((1,2));benefit[selected]=-torch.inf;site=int(benefit.argmax());selected.append(site);gains.append(float(benefit[site]));cov-=a[site].T@sol[site];cov=(cov+cov.T)*.5
        return np.asarray(selected,dtype=int),np.asarray(gains)
    def issue(self,p,scale,indices,start):
        if self.pending is not None:raise RuntimeError('An observation update is pending')
        indices=np.asarray(indices,dtype=int)
        if len(set(indices))!=len(indices) or np.any(indices<0) or np.any(indices>=128):raise ValueError('Invalid observation mask')
        self.pending=dict(p=p.detach().clone(),pack=self.pack(scale),indices=indices.copy(),start=start);return indices.copy()
    def observe(self,values,at):
        if self.pending is None or at!=self.pending['start']+25:raise ValueError('Expected fixed lead5 observation time')
        d=self.pending;values=torch.as_tensor(values,device='cuda',dtype=torch.float64)
        if values.shape!=(len(d['indices']),2) or not torch.isfinite(values).all():raise ValueError('Expected only requested physical u/v values')
        p=d['p'];physical=p*self.normalization['std'][0]+self.normalization['mean'][0];error=values-observation_rows(physical)[d['indices']];z=self.correction(d['pack'],d['indices'],error);result=physical[:,5:]+(z@d['pack']['a']).reshape(2,15,32,64);self.pending=None;return result

def configs():return [dict(rank=r,remainder=n,gain=g) for r in (8,16,32) for n in (.25,1.,4.) for g in (.5,1.)]

def loss_stats(pack,residual):
    r=residual[:,5:].double().flatten();return float(r.square().mean()),pack['a']@r/len(r)
def mse_from_latent(pack,base,dot,z):return float(base-2*z@dot+z@pack['g']@z)

@torch.no_grad()
def calibrate():
    d=load(OUT/'calibration.pt','cuda');source=load(SOURCE/'calibration.pt','cuda');m=load(OUT/'model.pt','cuda');rows=[]
    for ci,cfg in enumerate(configs()):
        assimilator=Assimilator(cfg,m)
        for i,meta in enumerate(d['meta']):
            pack=assimilator.pack(d['scale'][i]);r=(source['y'][i]-d['p'][i])*norm()['std'][0];base,dot=loss_stats(pack,r);observations=observation_rows(r)
            for budget in (4,8,16,32):
                ix=uniform(budget);z=assimilator.correction(pack,ix,observations[ix]);rows.append(dict(**meta,config=ci,budget=budget,base_mse=base,mse=mse_from_latent(pack,base,dot,z)))
        print('CALIBRATE',ci,cfg,flush=True)
    df=pd.DataFrame(rows);df.to_csv(OUT/'calibration_grid.csv',index=False);tr=df.groupby(['config','budget','file'])[['mse','base_mse']].mean().groupby(['config','budget']).mean();score=(tr.mse/tr.base_mse).groupby('config').mean();chosen=int(score.idxmin());dump(OUT/'selection.json',dict(config=configs()[chosen],config_index=chosen,criterion='Uniform-only calibration: mean loss/base ratios over4/8/16/32 sites, trajectory equal',scores={str(k):float(v) for k,v in score.items()},protocol_sha256=digest(Path(__file__).with_name('PROTOCOL.md')),model_sha256=digest(OUT/'model.pt')));print('SELECTED',configs()[chosen],float(score.loc[chosen]),flush=True)

def random_orders(meta):
    key=int.from_bytes(hashlib.sha256((meta['file']+':'+str(meta['start'])).encode()).digest()[:8],'little');rng=np.random.default_rng(np.random.SeedSequence([42,key]));return np.stack([rng.permutation(128) for _ in range(RANDOM_DRAWS)])

def policies(assim,pack,scale,risk,meta):
    variance=observation_rows(scale.square()).sum(-1);predrisk=risk.sum(-1);voi,benefit=assim.greedy(pack);constant=assim.pack(assim.model['average_scale']);fixed,_=assim.greedy(constant);orders=dict(spacefill=spacefill(),random=random_orders(meta)[0],variance=torch.argsort(variance,descending=True,stable=True).cpu().numpy(),risk=torch.argsort(predrisk,descending=True,stable=True).cpu().numpy(),voi=voi,voi_average_scale=fixed)
    return orders,benefit

@torch.no_grad()
def evaluate(split):
    d=load(OUT/f'{split}.pt','cuda');source=load(SOURCE/f'{split}.pt','cuda');sel=json.loads((OUT/'selection.json').read_text());assim=Assimilator(sel['config']);rows=[];requests=[];events=[];folder=OUT/split;folder.mkdir(exist_ok=True)
    for i,meta in enumerate(d['meta']):
        pack=assim.pack(d['scale'][i]);orders,benefit=policies(assim,pack,d['scale'][i],d['risk'][i],meta);actions={('uniform',b):uniform(b) for b in BUDGETS}
        for method,order in orders.items():actions.update({(method,b):order[:b] for b in BUDGETS})
        randoms=random_orders(meta)
        for draw,order in enumerate(randoms):actions.update({(f'random_draw{draw:02d}',b):order[:b] for b in BUDGETS})
        events.append(dict(**meta,event='issue_forecast_and_all_requests',time=meta['start']+20));requests.append(dict(meta=meta,actions={f'{k[0]}_{k[1]}':v.tolist() for k,v in actions.items()},voi_greedy_marginal_gains=benefit.tolist()))
        # The simulated sensor reveals only each requested subset to that action.
        # No later target enters even the scoring algebra until all z are frozen.
        observed=(observation_rows(source['y'][i])-observation_rows(d['p'][i]))*norm()['std'][0,:,0,0,0];events.append(dict(**meta,event='sparse_values_arrive',time=meta['start']+25));updates={}
        for (method,budget),indices in actions.items():
            updates[(method,budget)]=assim.correction(pack,indices,observed[indices])
        events.append(dict(**meta,event='reissue_remaining15',time=meta['start']+25));r=(source['y'][i]-d['p'][i])*norm()['std'][0];base,dot=loss_stats(pack,r)
        for (method,budget),indices in actions.items():
            z=updates[(method,budget)];loss=mse_from_latent(pack,base,dot,z);expected=assim.expected_gain(pack,indices) if not method.startswith('random_draw') else np.nan;rows.append(dict(**meta,method=method,budget=budget,scalar_measurements=2*budget,base_mse=base,mse=loss,predicted_reduction=expected,realized_reduction=base-loss))
            if i in (0,4) and method in ('uniform','voi','variance','risk') and budget==8:save(dict(p=d['p'][i].cpu(),scale=d['scale'][i].cpu(),indices=indices,observed_physical=(observed[indices]+observation_rows(d['p'][i]*norm()['std'][0]+norm()['mean'][0])[indices]).cpu(),truth=source['y'][i].cpu(),z=z.cpu(),analytic_mse=loss,meta=meta,config=sel['config']),folder/f'audit_{i:04d}_{method}.pt')
        events.append(dict(**meta,event='future15_score_only',time=meta['start']+40))
        if i%20==0:print('EVALUATE',split,i,len(d['meta']),flush=True)
    df=pd.DataFrame(rows);rand=df[df.method.str.startswith('random_draw')].groupby(['file','group','start','budget'])[['scalar_measurements','base_mse','mse','realized_reduction']].mean().reset_index().assign(method='random_expectation16',predicted_reduction=np.nan);df=pd.concat([df,rand],ignore_index=True);df.to_csv(folder/'windows.csv',index=False);keep=df[~df.method.str.startswith('random_draw')];cols=['base_mse','mse','predicted_reduction','realized_reduction'];tr=keep.groupby(['method','budget','file','group'])[cols].mean().reset_index();tr.to_csv(folder/'trajectories.csv',index=False);tr.groupby(['method','budget'])[cols].mean().to_csv(folder/'summary.csv');save(requests,folder/'requests.pt');(folder/'events.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in events));dump(folder/'provenance.json',dict(selection_sha256=digest(OUT/'selection.json'),source_sha256=digest(Path(__file__)),windows=len(d['meta']),target='unobserved future15 frames',sensor_time='start+25, lead5',original_task=False));print('DONE',split,flush=True)

if __name__=='__main__':
    setup();OUT.mkdir(parents=True,exist_ok=True);p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','fit','calibrate','evaluate']);p.add_argument('--split',default='development');args=p.parse_args();{'prepare':lambda:prepare(args.split),'fit':fit,'calibrate':calibrate,'evaluate':lambda:evaluate(args.split)}[args.action]()
