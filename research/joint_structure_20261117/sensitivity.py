import os
os.environ['TORCH_ALLOW_TF32_CUBLAS_OVERRIDE']='0'
from experiment import *
import experiment as exp
BASE=OUT
OUT=BASE/'sensitivity'

def basis_sample(basis,k,center=False):
    b=basis[:k].flatten(1)
    if center:b=b-b.mean(0)
    gen=torch.Generator(device='cuda').manual_seed(42)
    coef=torch.randn(N,k,device='cuda',generator=gen);coef-=coef.mean(0)
    coef=torch.linalg.qr(coef).Q*np.sqrt(N-1)
    z=coef@b/np.sqrt(k-1 if center else k)
    return z.reshape(N,2,20,32,64)

def summary(split):
    import pandas as pd
    rows=[]
    for path in sorted((OUT/split/'parts').glob('*.pt')):
        v=load(path)
        for method,s in v['scores'].items():rows.append(dict(method=method,**v['meta'],**s))
    df=pd.DataFrame(rows);df.to_csv(OUT/split/'per_window.csv',index=False)
    metrics=[c for c in df if c not in ('method','file','group','start')]
    byfile=df.groupby(['method','file','group'])[metrics].mean().reset_index();byfile.to_csv(OUT/split/'per_trajectory.csv',index=False)
    s=byfile.groupby('method')[metrics].mean();s.to_csv(OUT/split/'summary.csv')
    print(s[['es','vs','functional_crps','brier']].to_string(),flush=True)
    return s

@torch.no_grad()
def run(split):
    setup();seed();OUT.mkdir(exist_ok=True,parents=True)
    chosen=json.loads((OUT/'selection.json').read_text())['selected'] if split=='development' else None
    lib=exp.library();d=cache(split);n=norm();pred=model();heads,batches=models_for('XPF')
    destdir=OUT/split/'parts';destdir.mkdir(exist_ok=True,parents=True)
    for i in range(len(d['x'])):
        path=destdir/f'{i:04d}.pt'
        if path.exists():continue
        x=d['x'][i:i+1];p,f=feature(pred,x,'diffusion');base,last=features(x,p,f);values=quantile_values(all_bounds(batches,base,last),p[0]);p=p[0]
        timing={};native=[]
        torch.cuda.synchronize();start=time.perf_counter()
        for k in range(N):
            native.append(pred.rollout(x,500000+k,sign=1)[0])
            if k in (3,7,15,31):torch.cuda.synchronize();timing[f'native{k+1}']=time.perf_counter()-start
        native=torch.stack(native);directions=[]
        torch.cuda.synchronize();start=time.perf_counter()
        count=8 if chosen is None or chosen=='fd8' else 4
        for k in range(count):
            plus=pred.rollout(x,500000+k,sign=.1)[0];minus=pred.rollout(x,500000+k,sign=-.1)[0]
            directions.append((plus-minus)/.2)
            if k in (3,7):torch.cuda.synchronize();timing[f'fd{k+1}']=time.perf_counter()-start
        directions=torch.stack(directions);templates={'ecc32':native}
        for k in (4,8,16):templates[f'native_basis{k}']=basis_sample(native,k,True)
        for k in (4,8):
            if k<=count and (chosen is None or chosen==f'fd{k}'):templates[f'fd{k}']=basis_sample(directions,k)
        fields={name:couple(values,v) for name,v in templates.items()};fields['native32']=native
        # Extra first-direction diagnostic is not part of prediction costs.
        small=(pred.rollout(x,500000,sign=.05)[0]-pred.rollout(x,500000,sign=-.05)[0])/.1
        denom=directions[0].square().mean().sqrt().clamp_min(1e-8)
        derivative_error=float((small-directions[0]).square().mean().sqrt()/denom)
        full_error=float((directions[0]-(native[0]-p)).square().mean().sqrt()/(native[0]-p).square().mean().sqrt().clamp_min(1e-8))
        sc={};func={};marginal={}
        for name,z in fields.items():
            sc[name],func[name]=scores(z,d['y'][i],p,n,lib['stats'])
            if name!='native32':marginal[name]=float((z.sort(0).values-values).abs().max())
        save(dict(meta=d['meta'][i],scores=sc,functionals=func,marginal_error=marginal,timing=timing,derivative_halfstep_relative_rmse=derivative_error,full_noise_linearization_relative_rmse=full_error),path)
        if i in (0,len(d['x'])-1):save(dict(values=values.cpu(),samples={k:v.cpu() for k,v in fields.items()},y=d['y'][i].cpu(),p=p.cpu(),stats={k:v.cpu() for k,v in lib['stats'].items()},norm={k:v.cpu() for k,v in n.items()}),OUT/split/f'audit_{i:04d}.pt')
        if i%10==0:print('SENSITIVITY',split,i,len(d['x']),derivative_error,full_error,flush=True)
    s=summary(split)
    if split=='calibration':
        cols=['es','vs','functional_crps'];objective=(s.loc[['fd4','fd8'],cols]/s.loc['ecc32',cols]).mean(1);selected=objective.idxmin()
        dump(OUT/'selection.json',dict(selected=selected,objective=objective.to_dict(),criterion='mean ES, VS, functional CRPS relative to ECC32; calibration only',seed=42,tf32_override=0))
        print('SELECTED',selected,flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('split',choices=['calibration','development']);args=a.parse_args();run(args.split)
