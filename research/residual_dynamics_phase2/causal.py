"""Complete-block prediction/reveal state machine, lightweight conditional bias heads."""
import sys

import torch
from torch import nn

from common import *
sys.path.insert(0,str(REPO))
from mechanisms import model_new,network_predict
from metrics import scores


def native(split):
    # [trajectory,window,member,channel,time,y,x], issued before current truth.
    return torch.stack([torch.load(EXTRA/f'{split}_{start}_native.pt',map_location='cpu',weights_only=False)['paths']
                        for start in (400,420)],1).cuda()


def inputs(history,paths,feedback,mode,norm):
    """No target argument. Only already-issued native forecasts and revealed past."""
    # Batch native fields: [B,M,C,T,H,W]. Each lead is a separate batch item.
    n,_,_,t,h,w=paths.shape
    mean=paths.mean(1)
    last=(history[:,:,-1:]-norm['mean'])/norm['std']
    avg=(history.mean(2,keepdim=True)-norm['mean'])/norm['std']
    diff=(history[:,:,-1:]-history[:,:,-2:-1])/norm['std']
    hist=torch.cat([last,avg,diff],1).expand(n,6,t,h,w)
    pred=torch.cat([(mean-norm['mean'])/norm['std'],paths.std(1)/norm['std'],
                    (mean-history[:,:,-1:])/norm['std']],1)
    if mode=='history_only':pred=torch.zeros_like(pred)
    memory=paths.new_zeros(n,5,t,h,w)
    if mode=='with_memory' and feedback is not None:
        # Feedback belongs to another issued window. Not a known initial state.
        memory=torch.cat([feedback/norm['std'],
            feedback[:,:,-1:].expand_as(feedback)/norm['std'],torch.ones_like(feedback[:,:1])],1)
    yy,xx=torch.meshgrid(torch.linspace(-1,1,h,device=paths.device),torch.linspace(-1,1,w,device=paths.device),indexing='ij')
    lead=torch.linspace(-1,1,t,device=paths.device)[None,None,:,None,None].expand(n,1,t,h,w)
    coords=torch.stack([xx,yy])[None,:,None].expand(n,2,t,h,w)
    # All arms allocate20 channels, inactive fields explicitly zero.
    return torch.cat([hist,pred,memory,coords,lead],1).permute(0,2,1,3,4).reshape(n*t,20,h,w)


class BiasHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Conv2d(20,24,1),nn.SiLU(),nn.Conv2d(24,24,1),nn.SiLU(),nn.Conv2d(24,2,1))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)

    def forward(self,x):return self.net(x)


class BlockPredictor:
    def __init__(self,mode,norm,model=None,scale=1.):
        self.mode,self.norm,self.model,self.scale=mode,norm,model,scale
        self.previous=None;self.pending=None;self.completed=0

    @torch.no_grad()
    def predict(self,history,paths):
        if self.pending is not None:raise RuntimeError('reveal complete20 before next predict')
        if not torch.isfinite(history).all() or not torch.isfinite(paths).all():raise ValueError('nonfinite inputs')
        if history.shape[1:]!=(2,20,32,64) or paths.shape[2:]!=(2,20,32,64):raise ValueError('full20 required')
        if self.completed and not torch.equal(history,self.last_truth):raise ValueError('next history must equal previously revealed block')
        if self.mode=='native':bias=torch.zeros_like(paths.mean(1))
        elif self.mode=='persistence_bias':
            bias=torch.zeros_like(paths.mean(1)) if self.previous is None else self.previous[:,:,-1:].expand_as(paths.mean(1))
        else:
            x=inputs(history,paths,self.previous,self.mode,self.norm)
            bias=self.model(x).reshape(len(history),20,2,32,64).permute(0,2,1,3,4)*self.scale
        self.pending=paths.mean(1).clone()
        return paths-bias[:,None]

    @torch.no_grad()
    def reveal(self,truth):
        if self.pending is None:raise RuntimeError('predict before reveal')
        if truth.shape!=self.pending.shape or not torch.isfinite(truth).all():raise ValueError('reveal finite complete20 truth')
        self.previous=self.pending-truth
        self.last_truth=truth.clone()
        self.pending=None;self.completed+=1


def features_for_training(fields,paths,mode):
    h,y=fields['history'],fields['truth']
    out=[]
    for w in range(2):
        memory=paths[:,0].mean(1)-y[:,0] if w==1 else None
        out.append(inputs(h[:,w],paths[:,w],memory,mode,fields['normalization']).reshape(len(h),20,20,32,64))
    return torch.stack(out,1)


def fit():
    assert read(OUT/'alignment_audit.json')['passed']
    fields,devfields=aux('train'),aux('development')
    p,dp=native('train'),native('development')
    target=p.mean(2)-fields['truth']
    dtarget=dp.mean(2)-devfields['truth']
    scale=float(target.square().mean().sqrt())
    threshold=float(torch.quantile(fields['truth'][:,: ,1].square().mean((-1,-2)).sqrt().flatten(),.9))
    dump(OUT/'causal_training_protocol.json',dict(scale=scale,event_threshold=threshold,
        parameters=sum(v.numel() for v in BiasHead().parameters()),allocated_input_channels=20,seeds=list(SEEDS),steps=2000,
        method_boundaries=dict(history_only='observed physical history+coords+lead',
            current_fields='history+current issued ensemble mean/std/departure+coords+lead; no memory',
            with_memory='current_fields+previous complete native ensemble bias, lead-aligned and last-frame+availability'),
        cold='history400:420; target420:440; no invented prior feedback',
        warm='history420:440 equals previous truth; target440:460; previous whole20 bias allowed',
        primary_selection='macro MSE of ensemble mean over development trajectories and two blocks',
        no_current_target_in_predict=True))
    logs=[]
    for mode in ('history_only','current_fields','with_memory'):
        x=features_for_training(fields,p,mode)
        dx=features_for_training(devfields,dp,mode)
        for seed in SEEDS:
            dest=EXTRA/f'causal_{mode}_{seed}.pt'
            if dest.exists():continue
            torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
            model=BiasHead().cuda()
            opt=torch.optim.Adam(model.parameters(),lr=.001)
            rng=torch.Generator(device=DEVICE).manual_seed(20261120)
            best=float('inf')
            for step in range(1,2001):
                tr=torch.randint(len(x),(16,),device=DEVICE,generator=rng)
                wi=torch.randint(2,(16,),device=DEVICE,generator=rng)
                ti=torch.randint(20,(16,),device=DEVICE,generator=rng)
                loss=(model(x[tr,wi,ti])-target[tr,wi,:,ti]/scale).square().mean()
                opt.zero_grad(set_to_none=True);loss.backward();opt.step()
                if step%250==0:
                    model.eval()
                    with torch.no_grad():
                        values=[]
                        for i in range(len(dx)):
                            estimate=model(dx[i].flatten(0,1)).reshape(2,20,2,32,64).permute(0,2,1,3,4)*scale
                            values.append(float((estimate-dtarget[i]).double().square().mean()))
                        score=float(np.mean(values))
                    logs.append(dict(mode=mode,seed=seed,step=step,development_mse=score))
                    if score<best:
                        best=score
                        torch.save(dict(model=model.state_dict(),scale=scale,mode=mode,seed=seed,step=step,
                                        parameters=sum(v.numel() for v in model.parameters())),dest)
                    model.train()
                    print('CAUSAL_FIT',mode,seed,step,score,flush=True)
            dump(OUT/'causal_training.json',logs)
        del x,dx


@torch.no_grad()
def oracle_or_rollout(paths,truth,mode):
    # Single trajectory paths[M,C,T,H,W]. Oracle and causal kept separate.
    output=paths.clone()
    ck=None;m=None
    if mode!='oracle_persistence':
        arm='B' if mode.endswith('_B') else 'C'
        ck=torch.load(EXTRA/f'mechanism_{arm}_42.pt',map_location=DEVICE,weights_only=False)
        m=model_new(arm,42).eval();m.load_state_dict(ck['model'])
    state=torch.zeros_like(paths[:,:,0])
    for t in range(1,20):
        if mode.startswith('oracle'):
            previous=paths[:,:,t-1]-truth[:,t-1]
        else:previous=state
        estimate=previous if mode=='oracle_persistence' else network_predict(previous,
            torch.full((len(paths),),t+1,device=paths.device),m,ck['scale'])
        output[:,:,t]-=estimate
        state=estimate
    return output


@torch.no_grad()
def score():
    threshold=read(OUT/'causal_training_protocol.json')['event_threshold']
    rows=[];acfs=[];invariance=[]
    for split in ('development','test'):
        fields,p=aux(split),native(split)
        norm=fields['normalization'];std=norm['std'].squeeze(0)
        jobs=[('native',-1,None,1.),('persistence_bias',-1,None,1.)]
        for mode in ('history_only','current_fields','with_memory'):
            for seed in SEEDS:
                ck=torch.load(EXTRA/f'causal_{mode}_{seed}.pt',map_location=DEVICE,weights_only=False)
                m=BiasHead().cuda().eval();m.load_state_dict(ck['model'])
                jobs.append((mode,seed,m,ck['scale']))
        for mode,seed,m,scale in jobs:
            predictor=BlockPredictor(mode,norm,m,scale)
            for w in range(2):
                issued=predictor.predict(fields['history'][:,w],p[:,w])
                events('causal_forecast_issued',split=split,mode=mode,seed=seed,window=w)
                target=fields['truth'][:,w]
                for i,r in enumerate(protocol()['records'][split]):
                    v,forecast,reference=scores(issued[i],target[i],std,threshold)
                    rows.append(dict(split=split,protocol='causal',method=mode,seed=seed,trajectory=r['file'],
                        Re=r['group'],window=w,start=400+20*w,state='cold' if w==0 else 'warm',**v))
                    for lag in range(19):
                        for c,ch in enumerate(('u','v')):
                            acfs.append(dict(split=split,protocol='causal',method=mode,seed=seed,trajectory=r['file'],
                                window=w,channel=ch,lag=lag+1,predicted=float(forecast[lag,c]),truth=float(reference[lag,c])))
                native_delta=p[:,w]-p[:,w].mean(1,keepdim=True)
                issued_delta=issued-issued.mean(1,keepdim=True)
                invariance.append(dict(split=split,method=mode,seed=seed,window=w,
                    max_member_delta_change=float((native_delta-issued_delta).abs().max())))
                predictor.reveal(target)
                events('complete20_revealed_state_updated',split=split,mode=mode,seed=seed,window=w)
            print('CAUSAL_SCORE',split,mode,seed,flush=True)
        for mode in ('oracle_persistence','oracle_B','oracle_C','zero_init_rollout_B','zero_init_rollout_C'):
            oracle=mode.startswith('oracle')
            for w in range(2):
                for i,r in enumerate(protocol()['records'][split]):
                    target=fields['truth'][i,w]
                    # zero-init code receives no target at all.
                    issued=oracle_or_rollout(p[i,w],target if oracle else None,mode)
                    v,_,_=scores(issued,target,std,threshold)
                    rows.append(dict(split=split,protocol='oracle' if oracle else 'causal',method=mode,seed=42,
                        trajectory=r['file'],Re=r['group'],window=w,start=400+20*w,state='cold' if w==0 else 'warm',**v))
            print('ORACLE_OR_ZERO_INIT',split,mode,flush=True)
        del fields,p
    csvwrite(OUT/'forecast_per_trajectory_window.csv',rows)
    metrics=['mse','mae','crps','crps_standardized','energy','energy_standardized','temporal_acf_rmse','event_brier',
             'event_probability','event_observed','mse_u','mse_v','crps_u','crps_v']
    csvwrite(OUT/'forecast_summary.csv',aggregate(rows,['split','protocol','method','seed','state'],metrics))
    csvwrite(OUT/'forecast_by_Re.csv',aggregate(rows,['split','protocol','method','seed','state','Re'],metrics))
    csvwrite(OUT/'forecast_acf_all19lags.csv',acfs)
    dump(OUT/'common_shift_member_structure_invariance.json',invariance)
    comparisons=[]
    grouped=aggregate(rows,['split','method','seed','state','trajectory'],metrics)
    lookup={(r['split'],r['method'],r['seed'],r['state'],r['trajectory']):r for r in grouped}
    for split in ('development','test'):
        for state in ('cold','warm'):
            for a,b in [('history_only','native'),('current_fields','native'),('with_memory','current_fields'),
                        ('persistence_bias','native'),('zero_init_rollout_C','native'),('oracle_C','native')]:
                seeds=SEEDS if a in ('history_only','current_fields','with_memory') else (42 if a!='persistence_bias' else -1,)
                for seed in seeds:
                    sb=seed if b=='current_fields' else -1
                    for metric in ('mse','crps_standardized','energy_standardized','event_brier','temporal_acf_rmse'):
                        delta=[lookup[split,a,seed,state,r['file']][metric]-lookup[split,b,sb,state,r['file']][metric]
                               for r in protocol()['records'][split]]
                        comparisons.append(dict(split=split,state=state,contrast=a+' minus '+b,seed=seed,metric=metric,**bootstrap(delta)))
    dump(OUT/'forecast_bootstrap.json',comparisons)


if __name__=='__main__':
    import argparse
    from reliability.backbone import configure
    configure()
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['fit','score'])
    args=p.parse_args();(fit if args.stage=='fit' else score)()
