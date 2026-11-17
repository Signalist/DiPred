"""Spatial covariance taper of tangent sketches; same-cost nonlinear controls."""
from sensitivity import *
import sensitivity as sens
OUT=BASE/'localized'

def local_sample(basis,k,length,center=False):
    b=basis[:k]
    if center:b=b-b.mean(0,keepdim=True)
    # Independent Gaussian coefficient fields, shared by u/v and all leads.
    # This realizes Cov = (B B^T / k) elementwise-multiplied by a spatial kernel.
    gen=torch.Generator(device='cuda').manual_seed(42)
    white=torch.randn(N,k,64,128,device='cuda',generator=gen)
    fy=torch.fft.fftfreq(64,device='cuda')[:,None];fx=torch.fft.rfftfreq(128,device='cuda')[None,:]
    filt=torch.exp(-2*np.pi**2*length**2*(fx.square()+fy.square()))
    coef=torch.fft.irfft2(torch.fft.rfft2(white)*filt,s=(64,128))[:,:,16:48,32:96]
    coef=coef/coef.square().mean().sqrt()
    z=torch.einsum('nkxy,kctxy->nctxy',coef,b)/np.sqrt(k-1 if center else k)
    return z

def names():
    return [f'loc_{kind}{k}_l{l}' for kind,ks in [('fd',(4,8)),('native',(8,16))] for k in ks for l in (4,8,16)]

def parse(name):
    _,kind,l=name.split('_');return ('fd' if kind.startswith('fd') else 'native'),int(kind[2:] if kind.startswith('fd') else kind[6:]),int(l[1:])

@torch.no_grad()
def run(split):
    setup();seed();OUT.mkdir(exist_ok=True,parents=True)
    selected=json.loads((OUT/'selection.json').read_text()) if split=='development' else None
    lib=exp.library();d=cache(split);n=norm();pred=model();heads,batches=models_for('XPF')
    folder=OUT/split/'parts';folder.mkdir(parents=True,exist_ok=True)
    methods=names() if selected is None else list(dict.fromkeys([selected['selected'],selected['nonlinear_selected']]))
    for i in range(len(d['x'])):
        path=folder/f'{i:04d}.pt'
        if path.exists():continue
        x=d['x'][i:i+1];p,f=feature(pred,x,'diffusion');base,last=features(x,p,f);values=quantile_values(all_bounds(batches,base,last),p[0]);p=p[0]
        native=torch.stack([pred.rollout(x,500000+k,sign=1)[0] for k in range(N)])
        count=max(parse(m)[1] for m in methods if parse(m)[0]=='fd')
        directions=torch.stack([(pred.rollout(x,500000+k,sign=.1)[0]-pred.rollout(x,500000+k,sign=-.1)[0])/.2 for k in range(count)])
        templates={'ecc32':native,'fd8':basis_sample(directions,count),'native_basis8':basis_sample(native,8,True),'native_basis16':basis_sample(native,16,True)}
        if count==4:templates['fd4']=templates.pop('fd8')
        for name in methods:
            kind,k,l=parse(name);templates[name]=local_sample(directions if kind=='fd' else native,k,l,kind=='native')
        fields={name:couple(values,v) for name,v in templates.items()};sc={};func={};marginal={}
        for name,z in fields.items():
            sc[name],func[name]=scores(z,d['y'][i],p,n,lib['stats']);marginal[name]=float((z.sort(0).values-values).abs().max())
        save(dict(meta=d['meta'][i],scores=sc,functionals=func,marginal_error=marginal),path)
        if i in (0,len(d['x'])-1):save(dict(values=values.cpu(),samples={k:v.cpu() for k,v in fields.items()},y=d['y'][i].cpu(),p=p.cpu(),stats={k:v.cpu() for k,v in lib['stats'].items()},norm={k:v.cpu() for k,v in n.items()}),OUT/split/f'audit_{i:04d}.pt')
        if i%10==0:print('LOCALIZED',split,i,len(d['x']),flush=True)
    sens.OUT=OUT;s=sens.summary(split)
    if split=='calibration':
        cols=['es','vs','functional_crps'];objective=(s.loc[names(),cols]/s.loc['ecc32',cols]).mean(1)
        fd=[name for name in names() if parse(name)[0]=='fd'];nat=[name for name in names() if parse(name)[0]=='native']
        result=dict(selected=objective.loc[fd].idxmin(),nonlinear_selected=objective.loc[nat].idxmin(),objective=objective.to_dict(),criterion='mean ratios ES VS functional_CRPS to ECC32; calibration only',tf32_override=0,seed=42)
        dump(OUT/'selection.json',result);print(result,flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('split',choices=['calibration','development']);args=a.parse_args();run(args.split)
