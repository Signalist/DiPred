"""Fixed experiment primitives; clean-state correction, never history editing."""
import copy,csv,hashlib,json,os,subprocess
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from research.trust_causal_audit_v2.common import ROOT,DATA,MEMBERS,digest,tensor_hash,state_hash,load_backbone,records
from research.trust_causal_audit_v2.data import Key,CacheIndex,H5Reader,batch_noise
from reliability.features import forecast_features,base_features,grid_derivative
from reliability.selected16_heads import Selected16QuantileHead,BASE_INDICES

CODE=ROOT/'research/residual_uncertainty_diffusion'
OUT=ROOT/'results/residual_uncertainty_diffusion'
CACHE=ROOT/'.cache/residual_uncertainty_diffusion'
AUDIT=ROOT/'results/trust_causal_audit_v2'
REFERENCE='0c8296b24f9d089958b038a77933fcf641f9bf60'
ARMS=('C0','C1')
METRICS=('crps','fair_crps','energy','mse','smse','time_mean_crps','variance','std_variance','width90','std_width90','pair_half')


def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def read(path):return json.loads(Path(path).read_text())


def csvwrite(path,rows):
    if not rows:return
    with Path(path).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(rows)


def code_hashes():return {str(p.relative_to(ROOT)):digest(p) for p in sorted(CODE.glob('*.py'))}


def check_sources():
    v=read(AUDIT/'VERIFICATION.json');assert v['passed'] and v['READY_FOR_RETRAINING']
    for p,sha in v['code_hashes'].items():assert digest(ROOT/p)==sha,p
    # Archive evidence, checkpoints, and prior feedback remain byte-identical.
    for p,sha in read(AUDIT/'PROTECTED_BEFORE.json').items():assert digest(ROOT/p)==sha,p
    for p,sha in read(AUDIT/'ARTIFACT_HASHES.json').items():assert digest(ROOT/p)==sha,p


class FrozenXPF:
    def __init__(self,device='cuda'):
        shared=torch.load(ROOT/'reliability/checkpoints/current.pt',map_location=device,weights_only=False)
        packed=torch.load(ROOT/'reliability/checkpoints/selected16_comparison.pt',map_location=device,weights_only=False)
        self.native,self.norm=load_backbone(device)
        self.stats=shared['statistics']
        self.head=Selected16QuantileHead(shared['risk_head']).to(device)
        self.head.load_state_dict(packed['heads']['selected16'],strict=True)
        self.head.eval().requires_grad_(False)
        self.weight_hash=state_hash(self.head)

    @torch.no_grad()
    def extract(self,history):
        """No geometry required: selected six base channels omit curl entirely.

        Preserve actual training half->float rounding and frozen statistics.
        No online calibrator or memory state is constructed.
        """
        device=next(self.native.parameters()).device
        H=history.to(device)[None];h=(H-self.norm['mean'])/self.norm['std']
        point,features=forecast_features(self.native,h)
        physical=point*self.norm['std']+self.norm['mean']
        hp=(H-self.norm['mean'])/self.norm['std']
        pp=(physical-self.norm['mean'])/self.norm['std']
        variability=hp.std(2,keepdim=True).expand_as(pp)
        dx=grid_derivative(pp,-1,torch.ones(1,device=device))
        dy=grid_derivative(pp,-2,torch.ones(1,device=device))
        gradient=(dx.square()+dy.square()+1e-8).sqrt()
        change=torch.cat((pp[:,:,:1]-hp[:,:,-1:],pp.diff(dim=2)),2).abs()
        selected=torch.cat((variability,gradient,change),1)
        base=selected.new_zeros(1,14,20,32,64)
        base[:,list(BASE_INDICES)]=selected
        base=((base-self.stats['base_mean'])/self.stats['base_std']).clamp(-10,10)
        last=((features[:,-48:].half().float()-self.stats['last_mean'])/self.stats['last_std']).clamp(-10,10)
        lower,upper=self.head.bounds(base,last,h,None,False)
        width=(upper-lower)/2 # normalized velocity units, raw uncalibrated bounds
        assert (width>0).all() and torch.isfinite(width).all()
        return dict(W=width[0].cpu(),point=point[0].cpu(),base=base.cpu(),last=last.cpu(),
                    lower=lower[0].cpu(),upper=upper[0].cpu(),history=h[0].cpu())

    @torch.no_grad()
    def verify_equivalence(self,H):
        result=self.extract(H)
        h=result['history'][None].cuda();p=result['point'][None].cuda()
        # Legitimate nonselected features can be varied without passing any
        # invented physical geometry. Test a synthetic geometry ONLY as a
        # reference calculation: chosen feature slots never use it.
        observed=H[None].cuda();physical=p*self.norm['std']+self.norm['mean']
        errors=[]
        for geometry in (torch.tensor([[1.,2.,.1]],device='cuda'),torch.tensor([[-3.,7.,.2]],device='cuda')):
            reference=base_features(observed,physical,self.norm['mean'],self.norm['std'],geometry)
            reference=((reference-self.stats['base_mean'])/self.stats['base_std']).clamp(-10,10)
            assert torch.equal(reference[:,list(BASE_INDICES)],result['base'][:,list(BASE_INDICES)].cuda())
            lo,hi=self.head.bounds(reference,result['last'].cuda(),h,None,False)
            error=float(((hi-lo)[0]/2-result['W'].cuda()).abs().max());assert error==0
            errors.append(error)
        assert state_hash(self.head)==self.weight_hash
        return dict(selected_feature_indices=list(BASE_INDICES),feature_processing='original float16 rounding; frozen stats; clip[-10,10]',
            geometry_not_required_for_selected_features=True,synthetic_geometry_reference_only=True,
            width_maxabs_errors=errors,xpf_weight_state_sha256=self.weight_hash,no_calibrator=True,backbone_calls=4)


def release(reader,key,previous,norm,xpf):
    assert previous.key==Key(key.split,key.trajectory_id,key.start-20)
    assert previous.target_interval==(key.start,key.start+20)
    H=reader(key.trajectory_id,key.start,key.start+20)
    E=(previous.paths.mean(0)-H)/norm['std'].cpu()
    condition=xpf.extract(H)
    return dict(key=key,H=(H-norm['mean'].cpu())/norm['std'].cpu(),E=E,W=condition['W'],
        history_sha256=tensor_hash(H),E_sha256=tensor_hash(E),W_sha256=tensor_hash(condition['W']),
        previous_path_sha256=tensor_hash(previous.paths),previous_interval=list(previous.target_interval),
        condition_point_sha256=tensor_hash(condition['point']))


def condition_inputs(item,scales,noise,arm):
    device=noise.device;n=len(noise)
    h=item['H'].to(device)[None].expand(n,-1,-1,-1,-1)
    e=item['E'].to(device).clamp(-scales['E_clip'],scales['E_clip'])
    w=item['W'].to(device)/torch.tensor(scales['W_scale'],device=device).reshape(2,1,1,1)
    w=w.clamp(0,scales['W_clip'])
    # Whole-window risk retained; never normalize a window to equal mean/std.
    avg=w.mean((1,2,3))[:,None,None].expand(2,32,64)
    if arm=='C0':e=torch.zeros_like(e);w=torch.zeros_like(w);avg=torch.zeros_like(avg)
    return dict(history=h,E=e[None].expand(n,-1,-1,-1,-1),W=w[None].expand(n,-1,-1,-1,-1),
        W_mean=avg[None].expand(n,-1,-1,-1),noise=noise)


class Correction(nn.Module):
    def __init__(self):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42)
            self.net=nn.Sequential(nn.Conv2d(211,16,1),nn.SiLU(),nn.Conv2d(16,16,3,padding=1),nn.SiLU(),nn.Conv2d(16,40,1))
            nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
        assert sum(p.numel() for p in self.parameters())==6392
    def forward(self,x,h,e,w,wmean,feature,label):
        n=len(x);z=torch.cat((x.reshape(n,40,32,64),h.reshape(n,40,32,64),e.reshape(n,40,32,64),
            w.reshape(n,40,32,64),feature,wmean,(label.float()/99)[:,None,None,None].expand(n,1,32,64)),1)
        return .1*self.net(z).reshape_as(x) # fixed pre-training scale


class CorrectedDiffusion(nn.Module):
    def __init__(self,native):
        super().__init__();self.native=copy.deepcopy(native).eval().requires_grad_(False)
        self.native.net.out.requires_grad_(True)
        self.correction=Correction().to(next(native.parameters()).device)
    def sample(self,history,E,W,W_mean,noise,enabled=True,trace=None):
        x=noise.clone()
        for step in (3,2,1,0):
            label=torch.full((len(x),),step*33,device=x.device,dtype=torch.long)
            features=[];handle=self.native.net.up2.register_forward_hook(lambda m,a,o:features.append(o))
            try:base=self.native.net(x,label,history)
            finally:handle.remove()
            assert len(features)==1
            delta=self.correction(x,history,E,W,W_mean,features[0],label) if enabled else torch.zeros_like(base)
            clean=base+delta
            if trace is not None:
                if clean.requires_grad:clean.retain_grad()
                trace.append(dict(label=step*33,delta=delta,clean=clean,base=base))
            alpha=self.native.abar[step];eps=(x-alpha.sqrt()*clean)/(1-alpha).sqrt()
            prev=self.native.abar[step-1] if step else x.new_tensor(1.)
            x=prev.sqrt()*clean+(1-prev).sqrt()*eps
        return x


def crps(paths,truth,fair=False,parts=False):
    k=len(paths);assert k>1
    weight=(2*torch.arange(k,device=paths.device,dtype=paths.dtype)-k+1).reshape(k,*([1]*(paths.ndim-1)))
    first=(paths-truth).abs().mean()
    half=(paths.sort(0).values*weight).sum(0).mean()/(k*(k-1) if fair else k*k)
    return (first-half,first,half) if parts else first-half


def energy(paths,truth):
    a=(paths-truth).flatten(1).square().mean(1).sqrt().mean()
    # Explicit differences avoid diagonal roundoff in Gram distance formulas.
    half=sum((paths[i:i+1]-paths[:i]).flatten(1).square().mean(1).sqrt().sum() for i in range(1,len(paths)))/len(paths)**2
    return a-half


def score(paths,target,norm):
    p=paths.double();y=target.to(p.device).double();std=norm['std'].double();mean=norm['mean'].double()
    yn=(y-mean)/std;raw=p*std+mean;rows=[]
    for sl,ch in ((slice(None),'overall'),(slice(0,1),'u'),(slice(1,2),'v')):
        pc=p[:,sl];yc=yn[sl];err=raw[:,sl].mean(0)-y[sl];s=std[sl]
        value,_,pair=crps(pc,yc,parts=True);width=torch.quantile(raw[:,sl],.95,0)-torch.quantile(raw[:,sl],.05,0)
        rows.append(dict(channel=ch,crps=float(value),fair_crps=float(crps(pc,yc,True)),energy=float(energy(pc,yc)),
           mse=float(err.square().mean()),smse=float((err/s).square().mean()),time_mean_crps=float(crps(pc.mean(2),yc.mean(1))),
           variance=float(raw[:,sl].var(0,unbiased=False).mean()),std_variance=float(pc.var(0,unbiased=False).mean()),
           width90=float(width.mean()),std_width90=float((width/s).mean()),pair_half=float(pair)))
    return rows


def trajectory_summary(rows):
    from collections import defaultdict
    group=defaultdict(list)
    for r in rows:group[(r['arm'],r['step'],r['trajectory_id'],r['Re'],r['channel'])].append(r)
    tr=[dict(zip(('arm','step','trajectory_id','Re','channel'),k),**{m:float(np.mean([r[m] for r in rr])) for m in METRICS}) for k,rr in group.items()]
    macro=[]
    for arm,step,ch in sorted({(r['arm'],r['step'],r['channel']) for r in tr}):
        rr=[r for r in tr if (r['arm'],r['step'],r['channel'])==(arm,step,ch)]
        macro.append(dict(arm=arm,step=step,channel=ch,trajectories=len(rr),**{m:float(np.mean([r[m] for r in rr])) for m in METRICS}))
    return tr,macro


def select(rows):
    tr,macro=trajectory_summary(rows);native=next(r for r in macro if r['arm']=='native' and r['channel']=='overall')
    selected=[]
    for arm in ARMS:
        candidates=[native]+[r for r in macro if r['arm']==arm and r['channel']=='overall' and r['step'] in (250,500)]
        best=min(candidates,key=lambda r:(r['crps'],r['step']))
        selected.append(dict(arm=arm,chosen_arm=best['arm'],step=best['step'],crps=best['crps']))
    return selected,tr,macro
