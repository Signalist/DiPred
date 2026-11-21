"""Independent nonlinear protocol; read-only reuse of the completed pilot primitives."""
import sys
import os
os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"]="0"
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from research.conditional_innovation_pilot import core as pc
from research.conditional_innovation_pilot.core import *
import time

DELIVERY=ROOT/'results/nonlinear_parameterization_control'
OUT=Path(os.environ.get('DIPRED_NONLINEAR_OUT',str(DELIVERY)))
CFG=json.loads((DELIVERY/'config.json').read_text())
CONFIG_HASH=digest(DELIVERY/'config.json')
assert digest(DELIVERY/'PROTOCOL.md')==CFG['protocol_sha256']
RECORDS=CFG['records']
ARMS=CFG['arms']
SEEDS=CFG['seeds']
METRICS=['crps','mse','smse','energy','spread','sspread','r_mse','r_smse','brier']
CELL_METRICS=['crps','mse','smse','spread','sspread']


def check_dependencies():
    for name,sha in CFG['dependencies'].items():assert digest(ROOT/name)==sha,name
    assert digest(ROOT/'reliability/checkpoints/current.pt')==CFG['frozen']['packaged_checkpoint_sha256']
    for left,right in [('train','development'),('train','test'),('development','test')]:
        for key in ('file','group'):
            assert {r[key] for r in RECORDS[left]}.isdisjoint({r[key] for r in RECORDS[right]})


def state_hash(state):
    h=hashlib.sha256()
    for name,value in sorted(state.items()):
        h.update(name.encode());h.update(value.detach().contiguous().cpu().numpy().tobytes())
    return h.hexdigest()


def cpu_state(model):return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}


def make_model(kind,seed,std):
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    a,d=operators(std.device)
    return SmallModel(kind,a,d,std).to(std.device)


def loss(paths,truth,correction,std):
    # Labels enter only here, never the predictor; constant member term omitted.
    return ((paths-truth[:,None]-correction[:,None])/std).abs().mean()


@torch.no_grad()
def full_scores(data,q,method,seed=0):
    a,_=operators(data.p.device)
    rows,cells=pc.score(data,q,method,a,seed)
    for i,row in enumerate(rows):
        paths=(data.p[i]-q[i,None]).double()
        variance=paths.var(0,unbiased=False).mean((-1,-2))
        normalized=variance/data.std.reshape(1,2).double().square()
        row.update(spread=float(variance.mean()),sspread=float(normalized.mean()))
        for j in range(20):
            for ch in range(2):cells[i*40+j*2+ch].update(spread=float(variance[j,ch]),sspread=float(normalized[j,ch]))
    return rows,cells


def aggregate(rows,keys,metrics):
    from collections import defaultdict
    groups=defaultdict(list)
    for r in rows:groups[tuple(r[k] for k in keys)].append(r)
    return [dict(zip(keys,key),**{m:float(np.mean([float(r[m]) for r in rr if r[m]!=''])) for m in metrics}) for key,rr in groups.items()]


def mean_metrics(rows):return aggregate(rows,[],METRICS)[0]


def choose(rows,protected):
    native=next(r for r in rows if int(r['step'])==0)
    choices=[r for r in rows if not protected or admissible(r,native)]
    return min(choices,key=lambda r:(float(r['crps']),int(r['step'])))


def expanded(source,operator):
    values=[]
    for lead in range(source.shape[1]):
        value=torch.zeros_like(source[:,0])
        for j in range(lead+1):
            term=source[:,j]
            for _ in range(lead-j):term=simple_predict(term,None,operator)
            value=value+term
        values.append(value)
    return torch.stack(values,1)


def sensitivity(model,x,label):
    """One full real/synthetic input, centre output, leads 1 and20, both channels."""
    rows=[];maps=[]
    for lead in (0,19):
        for ch in range(2):
            inp=x[:1].detach().clone().requires_grad_(True)
            q=model(inp);cy,cx=q.shape[-2]//2,q.shape[-1]//2
            g=torch.autograd.grad(q[0,lead,ch,cy,cx],inp)[0].abs().sum(1)[0]
            yy,xx=torch.meshgrid(torch.arange(g.shape[0],device=g.device),torch.arange(g.shape[1],device=g.device),indexing='ij')
            radius=torch.maximum((yy-cy).abs(),(xx-cx).abs())
            total=float(g.sum());mask=g>max(float(g.max())*1e-6,1e-15)
            rows.append(dict(label=label,kind=model.kind,lead=lead+1,channel='uv'[ch],input_gradient_l1=total,
                max_radius_relative_1e6=int(radius[mask].max()) if bool(mask.any()) else 0,
                gradient_fraction_outside_radius1=float(g[radius>1].sum())/max(total,1e-30),
                q_rms=float(q.detach().square().mean().sqrt())))
            maps.append(g.detach().cpu())
    return rows,torch.stack(maps)
