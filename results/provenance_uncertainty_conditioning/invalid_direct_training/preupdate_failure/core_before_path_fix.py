"""Frozen four-step DiPred with equal-capacity history-condition adapters."""
import os
os.environ['TORCH_ALLOW_TF32_CUBLAS_OVERRIDE']='0'
import sys,json,csv,time,hashlib,gzip
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch import nn
from research.native_restart_consistency import core as native
from research.restart_input_attribution import core as prior
from reliability.backbone import configure
DELIVERY=ROOT/'results/provenance_uncertainty_conditioning'
OUT=Path(os.environ.get('DIPRED_CONDITION_OUT',str(DELIVERY)))
CACHE=Path(os.environ.get('DIPRED_CONDITION_CACHE',str(native.WORK/'runs/provenance_uncertainty_conditioning')))
CFG=json.loads((DELIVERY/'config.json').read_text())
digest,dump,csvwrite,tensor_hash=native.digest,native.dump,native.csvwrite,native.tensor_hash
QUAL=['crps','energy','mse','smse','variance','standardized_variance','width90','standardized_width90','time_mean_crps']
META=['split','trajectory','Re','start','repeat','k']


def check():
    assert digest(DELIVERY/'PROTOCOL.md')==CFG['protocol_sha256']
    for path,sha in CFG['dependencies'].items():assert digest(ROOT/path)==sha,path


def load_model():
    model,norm,frozen=native.load_backbone(SimpleNamespace(workspace=native.WORK))
    assert frozen==CFG['frozen'] and all(not p.requires_grad for p in model.parameters())
    return model,norm


class Adapter(nn.Module):
    def __init__(self,arm,seed):
        super().__init__();assert arm in CFG['arms'];self.arm=arm
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.encoder=nn.Sequential(nn.Conv2d(101,8,1),nn.SiLU(),nn.Conv2d(8,8,3,padding=1,padding_mode='replicate'),nn.SiLU())
            self.out=nn.Conv2d(8,40,1);nn.init.zeros_(self.out.weight);nn.init.zeros_(self.out.bias)
        assert sum(p.numel() for p in self.parameters())==CFG['parameters']

    def forward(self,h,mask,spread,label):
        n,c,t,ny,nx=h.shape
        if self.arm=='C_aug':mask=torch.zeros_like(mask)
        if self.arm!='C_unc':spread=torch.zeros_like(spread)
        step=h.new_full((n,1,ny,nx),label/99)
        x=torch.cat([h.reshape(n,40,ny,nx),mask.expand(n,-1,-1,-1),spread.expand(n,-1,-1,-1,-1).reshape(n,40,ny,nx),step],1)
        return self.out(self.encoder(x)).reshape_as(h)


def sample(model,adapter,h,mask,spread,noise,trace=None):
    x=noise.clone()
    for step in reversed(range(4)):
        delta=adapter(h,mask,spread,step*33)
        if trace is not None:delta.retain_grad();trace.append(delta)
        clean=model.net(x,torch.full((len(h),),step*33,device=h.device,dtype=torch.long),h+delta)
        alpha=model.abar[step];residual=(x-alpha.sqrt()*clean)/(1-alpha).sqrt()
        prev=model.abar[step-1] if step else x.new_tensor(1.)
        x=prev.sqrt()*clean+(1-prev).sqrt()*residual
    return x


def conditions(history,parent,k,norm):
    """No labels accepted. The complete ordered parent ensemble is retained."""
    assert history.shape==(2,20,32,64) and parent.shape==(32,2,20,32,64)
    assert k in CFG['train_k']
    if k:
        h=native.restart_history(history,parent,k)
    else:h=history[None].expand(32,-1,-1,-1,-1)
    hn=(h-norm['mean'])/norm['std']
    mask=h.new_zeros((1,20,32,64));spread=h.new_zeros((1,2,20,32,64))
    if k:
        mask[:,-k:]=1
        spread[:,:,-k:]=parent[:,:,:k].std(0,unbiased=False)[None]/norm['std']
    return hn.contiguous(),mask,spread


def crps_parts(paths,truth):
    """Exact empirical CRPS for all32 members, with differentiable pair term."""
    n=len(paths);ordered=paths.sort(dim=0).values
    w=(2*torch.arange(n,device=paths.device,dtype=paths.dtype)-n+1).reshape(n,*([1]*(paths.ndim-1)))
    first=(paths-truth).abs().mean();pair=(ordered*w).sum(0).mean()/(n*n)
    return first-pair,first,pair


def state_hash(module):
    h=hashlib.sha256()
    for k,v in module.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def entries(split):return [e for e in prior.entries() if e['split']==split]


def prepare(bundle,k,norm):
    h,mask,spread=conditions(bundle['history'].cuda(),bundle['paths']['A'].cuda(),k,norm)
    call=bundle['calls']['B' if k==0 else 'S'+str(k)]
    z=native.seeded_noise(h,call['seeds'])
    assert [tensor_hash(v) for v in z]==call['noise_sha256']
    return h,mask,spread,z


def truth_for(entry,k):return prior.frames(entry['trajectory'],entry['start']+20+k,entry['start']+40+k).cuda()


def decode(p,norm):return p*norm['std']+norm['mean']


def event(action,**kw):
    with (OUT/'events.jsonl').open('a') as f:f.write(json.dumps(dict(action=action,utc=time.time(),**kw))+'\n')


def code_hashes():return {str(p.relative_to(ROOT)):digest(p) for p in Path(__file__).parent.glob('*.py')}


def checkpoint(adapter,seed,step):
    p=OUT/'weights'/f'{adapter.arm}_{seed}_{step}.pt';assert not p.exists();p.parent.mkdir(parents=True,exist_ok=True)
    state={k:v.detach().cpu() for k,v in adapter.state_dict().items()}
    torch.save(dict(arm=adapter.arm,seed=seed,step=step,state=state,config_sha256=digest(DELIVERY/'config.json')),p)
    return dict(arm=adapter.arm,seed=seed,step=step,path=str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p),sha256=digest(p),state_sha256=state_hash(adapter))


def read_checkpoint(item):
    p=ROOT/item['path'];assert digest(p)==item['sha256']
    ck=torch.load(p,map_location='cpu',weights_only=False);a=Adapter(item['arm'],item['seed']).cuda();a.load_state_dict(ck['state']);a.eval()
    assert state_hash(a)==item['state_sha256'];return a


def read_csv(p):
    opener=gzip.open if str(p).endswith('.gz') else open
    with opener(p,'rt') as f:return list(csv.DictReader(f))


class Table:
    def __init__(self,path):
        assert not path.exists();self.file=gzip.open(path,'wt',newline='') if str(path).endswith('.gz') else path.open('w',newline='');self.writer=None;self.count=0
    def write(self,rows):
        if not rows:return
        if self.writer is None:self.writer=csv.DictWriter(self.file,fieldnames=list(rows[0]),lineterminator='\n');self.writer.writeheader()
        self.writer.writerows(rows);self.count+=len(rows)
    def flush(self):self.file.flush()
    def close(self):self.file.close()
