"""Nonperiodic discrete Hodge/rank alternating projection for vector fields."""
from sensitivity import *
import sensitivity as sens
OUT=BASE/'physics'
DX=.003421664
CANDIDATE_PREFIX='hodge'
CONTROL_PREFIX='smooth'
EXTRAS=[]

def physical_functionals(z,p,n):
    old=exp.functionals(z,p,n)
    a=z*n['std']+n['mean']
    vort=(a[:,1,:,1:-1,2:]-a[:,1,:,1:-1,:-2])/(2*DX)+(a[:,0,:,2:,1:-1]-a[:,0,:,:-2,1:-1])/(2*DX)
    old[:,2]=vort.square().mean((-1,-2)).amax(-1)
    return old

def divergence(a):
    return (a[:,0,:,1:-1,2:]-a[:,0,:,1:-1,:-2])/2-(a[:,1,:,2:,1:-1]-a[:,1,:,:-2,1:-1])/2

def adjoint(q):
    u=F.pad(q,(2,0,1,1))/2-F.pad(q,(0,2,1,1))/2
    v=-F.pad(q,(1,1,2,0))/2+F.pad(q,(1,1,0,2))/2
    return torch.stack((u,v),1)

def eig(n,device):
    a=torch.eye(n,device=device,dtype=torch.float64)*.5
    i=torch.arange(n-2,device=device);a[i,i+2]=-.25;a[i+2,i]=-.25
    return torch.linalg.eigh(a)

class Hodge:
    def __init__(self):
        dy,uy=eig(30,'cuda');dx,ux=eig(62,'cuda');self.uy=uy.float();self.ux=ux.float();self.lam=(dy[:,None]+dx[None,:]).float()
    def __call__(self,a):
        d=divergence(a);q=self.uy.T@d@self.ux/self.lam;q=self.uy@q@self.ux.T
        return a-adjoint(q)

def transform(values,z,n,kind,beta,iterations,project,p=None):
    center=z.mean(0,keepdim=True)
    for _ in range(iterations):
        delta=(z-center)*n['std']
        if kind=='hodge':new=project(delta)
        else:
            shape=delta.shape;a=delta.reshape(N*2*20,1,32,64)
            new=F.avg_pool2d(F.pad(a,(1,1,1,1),mode='replicate'),3,1).reshape(shape)
        z=couple(values,center+((1-beta)*delta+beta*new)/n['std'])
    return z

def names():
    return [f'{kind}_{source}_b{beta}_i{it}' for kind in ('hodge','smooth') for source in ('ecc','emp') for beta in (25,50,100) for it in (1,3)]

def parse(name):
    kind,source,b,it=name.split('_');return kind,source,int(b[1:])/100,int(it[1:])

def physical_stats(lib,n):
    path=OUT/'physical_stats.pt'
    if path.exists():return load(path,'cuda')
    d=cache('train');fs=[]
    for i in range(0,len(d['x']),16):fs.append(physical_functionals(d['y'][i:i+16],d['p'][i:i+16],n))
    fs=torch.cat(fs);stats=dict(scale=fs.std(0).clamp_min(1e-6),thresholds=torch.quantile(fs,fs.new_tensor([.75,.9]),dim=0))
    save(stats,path);return stats

@torch.no_grad()
def run(split):
    setup();seed();OUT.mkdir(exist_ok=True,parents=True)
    selected=json.loads((OUT/'selection.json').read_text()) if split!='calibration' else None
    lib=exp.library();n=norm();stats=physical_stats(lib,n)
    if split in ('late','future'):d=load(OUT/f'{split}_data.pt','cuda')
    else:d=cache(split)
    pred=model();heads,batches=models_for('XPF');project=Hodge()
    folder=OUT/split/'parts';folder.mkdir(parents=True,exist_ok=True)
    methods=names() if selected is None else list(dict.fromkeys([selected['selected'],selected['smooth_selected'],selected['matched_smooth']]))
    methods=list(dict.fromkeys(methods+EXTRAS))
    original_func=exp.functionals
    # scores resolves functionals in experiment globals; avoid recursive wrapper.
    def actual(z,p,n):
        old=original_func(z,p,n);a=z*n['std']+n['mean']
        vort=(a[:,1,:,1:-1,2:]-a[:,1,:,1:-1,:-2])/(2*DX)+(a[:,0,:,2:,1:-1]-a[:,0,:,:-2,1:-1])/(2*DX)
        old[:,2]=vort.square().mean((-1,-2)).amax(-1);return old
    exp.functionals=actual
    for i in range(len(d['x'])):
        path=folder/f'{i:04d}.pt'
        if path.exists():continue
        x=d['x'][i:i+1];p,f=feature(pred,x,'diffusion');base,last=features(x,p,f);values=quantile_values(all_bounds(batches,base,last),p[0]);p=p[0]
        native=torch.stack([pred.rollout(x,500000+k,sign=1)[0] for k in range(N)])
        initial=dict(ecc=couple(values,native),emp=couple(values,lib['residual'][lib['unconditional']]))
        fields=dict(ecc32=initial['ecc'],empirical=initial['emp'])
        for name in methods:
            kind,source,beta,it=parse(name);fields[name]=transform(values,initial[source],n,kind,beta,it,project,p)
        sc={};func={};marginal={}
        for name,z in fields.items():
            sc[name],func[name]=scores(z,d['y'][i],p,n,stats);marginal[name]=float((z.sort(0).values-values).abs().max())
            a=z*n['std']+n['mean'];dy=divergence(d['y'][i:i+1]*n['std']+n['mean'])/DX;dz=divergence(a)/DX
            divcrps=empirical_crps(dz.square().mean((1,2,3)),dy.square().mean((1,2,3))[0])
            sc[name].update(divergence_power=float(dz.square().mean()),truth_divergence_power=float(dy.square().mean()),divergence_crps=float(divcrps),perturbation_divergence_power=float(divergence(a-a.mean(0,keepdim=True)).square().mean()/DX**2))
        save(dict(meta=d['meta'][i],scores=sc,functionals=func,marginal_error=marginal),path)
        if i in (0,len(d['x'])-1):save(dict(values=values.cpu(),samples={k:v.cpu() for k,v in fields.items()},y=d['y'][i].cpu(),p=p.cpu(),stats={k:v.cpu() for k,v in stats.items()},norm={k:v.cpu() for k,v in n.items()}),OUT/split/f'audit_{i:04d}.pt')
        if i%10==0:print('PHYSICS',split,i,len(d['x']),flush=True)
    sens.OUT=OUT;s=sens.summary(split)
    if split=='calibration':
        cols=['es','vs','functional_crps'];objective=(s.loc[names(),cols]/s.loc['ecc32',cols]).mean(1)
        hs=[m for m in names() if m.startswith(CANDIDATE_PREFIX)];ss=[m for m in names() if m.startswith(CONTROL_PREFIX)];chosen=objective.loc[hs].idxmin()
        result=dict(selected=chosen,smooth_selected=objective.loc[ss].idxmin(),matched_smooth=chosen.replace(CANDIDATE_PREFIX,CONTROL_PREFIX),objective=objective.to_dict(),criterion='mean ES VS physical_functional_CRPS ratios vs ECC32; calibration only',tf32_override=0,seed=42,dx=DX,dy=-DX)
        dump(OUT/'selection.json',result);print(result,flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('split',choices=['calibration','development','late','future']);args=a.parse_args();run(args.split)
