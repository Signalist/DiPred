"""RealPDE-only, single-seed research utilities. No test split is opened."""
from pathlib import Path
import hashlib, json, sys, os
import numpy as np
import torch
W = Path(__file__).resolve().parent.parent
PROJECT = W.parent
DATA = PROJECT.parent / 'OLD/REALPDE/realpdebench-competition/data/train_real'
OUT = W / 'runs/f_risk_query_20261116'
sys.path.insert(0, str(W/'github/DiPred'))
from reliability.networks import DynamicsDiffusion20
SHARED = W/'github/DiPred/reliability/checkpoints/current.pt'
DET = W/'runs/physical20_augmented_dynamics_20261109/augmented_cnn/model.pt'
MANIFEST = W/'runs/clean_direct20_20261107/manifest.json'
SEED = 42

def setup():
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False
    torch.manual_seed(SEED); np.random.seed(SEED)
    OUT.mkdir(exist_ok=True,parents=True)

def load(path,device='cpu'): return torch.load(path,map_location=device,weights_only=False)
def save(value,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');torch.save(value,tmp);os.replace(tmp,path)
def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def dump(path,value):Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')
def records(split):
    assert split in ('train','calibration','development'), 'This pilot must not open test data'
    return json.loads(MANIFEST.read_text())['files'][split]
def norm(device='cuda'):return {k:v.to(device) for k,v in load(SHARED)['normalization'].items()}
def read(rec,start):
    import h5py
    with h5py.File(DATA/rec['file'],'r') as f:
        a=np.stack([f[c][start:start+40,::2,::2] for c in ('u','v')]).astype('float32')
    assert a.shape==(2,40,32,64) and np.isfinite(a).all()
    return torch.from_numpy(a)
def model(kind='diffusion'):
    ck=load(SHARED,'cuda'); torch.manual_seed(SEED)
    m=DynamicsDiffusion20(ck['dynamics_scale']).cuda().eval().requires_grad_(False)
    if kind=='diffusion':m.load_state_dict(ck['predictor'])
    elif kind=='deterministic':m.load_state_dict(load(DET,'cuda')['model'])
    elif kind!='random':raise ValueError(kind)
    return m
@torch.no_grad()
def feature(m,x,kind):
    stages=[];h=m.net.up2.register_forward_hook(lambda mod,args,out:stages.append(out.detach()))
    try:
        if kind=='deterministic':p=m.net(torch.zeros_like(x),torch.full((len(x),),99,device=x.device,dtype=torch.long),x)
        else:p=m.rollout(x,500000,sign=0)
    finally:h.remove()
    assert len(stages)==(1 if kind=='deterministic' else 4)
    return p,stages[-1]

def cache(split):return load(OUT/'data'/f'{split}.pt','cuda')
def bases(t):
    k=t.new_tensor([.2,.4,.6,.8]);z=t[:,None]
    return torch.cat((z-.5,torch.logit(z)/4,(torch.nn.functional.softplus(8*(z-k))-torch.nn.functional.softplus(8*(.5-k)))/8),1)
def crps(q,y,levels):
    # Actual numerical trapezoid, with flat quantile tails to 0 and 1.
    t=levels.reshape(-1,*([1]*y.ndim));e=y[None]-q
    loss=torch.maximum(t*e,(t-1)*e)
    interior=torch.trapz(loss,levels,dim=0)
    return 2*(interior+levels[0]*(loss[0]+(q[0]-y).clamp_min(0))/2+(1-levels[-1])*(loss[-1]+(y-q[-1]).clamp_min(0))/2)
