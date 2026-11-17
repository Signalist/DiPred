"""Independent NumPy numerical audit, trajectory bootstrap, provenance checks."""
from experiment import *
import pandas as pd

def numpy_crps(z,y):
    # Direct pairwise definition independent of the sorted weighted implementation.
    return np.abs(z-y).mean(0)-sum(np.abs(z[i]-z).sum(0) for i in range(len(z)))/(2*len(z)**2)

def numpy_functionals(z,p,n):
    a=z*n['std']+n['mean'];center=a.mean((-1,-2),keepdims=True)
    tke=(.5*((a-center)**2).sum(1).mean((-1,-2))).max(-1)
    kinetic=.5*(a*a).sum(1).mean((1,2,3))
    vort=(a[:,1,:,1:-1,2:]-a[:,1,:,1:-1,:-2])/2-(a[:,0,:,2:,1:-1]-a[:,0,:,:-2,1:-1])/2
    en=(vort*vort).mean((-1,-2)).max(-1)
    speed=np.sqrt((a*a).sum(1)).reshape(len(a),-1).max(1)
    mvpe=np.sqrt((((z-p)*n['std'])**2).sum(1)).reshape(len(a),-1).max(1)
    return np.stack((tke,kinetic,en,speed,mvpe),1)

def numerical(split):
    errors=[]
    for path in sorted((OUT/split).glob('audit_*.pt')):
        d=load(path);i=int(path.stem.split('_')[-1]);scores_saved=load(OUT/split/'parts'/f'{i:04d}.pt')['scores']
        y=d['y'].numpy().astype('float64');p=d['p'].numpy().astype('float64');n={k:v.numpy().astype('float64') for k,v in d['norm'].items()}
        stats={k:v.numpy().astype('float64') for k,v in d['stats'].items()}
        for name,tensor in d['samples'].items():
            z=tensor.numpy().astype('float64');flat=z.reshape(N,-1);target=y.reshape(-1)
            es=np.sqrt(((flat-target)**2).mean(1)).mean()-sum(np.sqrt(((flat[i]-flat)**2).mean(1)).sum() for i in range(N))/(2*N*N)
            vs=np.mean([((np.sqrt(np.abs(np.diff(z,axis=dim+1))).mean(0)-np.sqrt(np.abs(np.diff(y,axis=dim))))**2).mean() for dim in (3,2,1,0)])
            f=numpy_functionals(z,p,n);fy=numpy_functionals(y[None],p,n)[0]
            fc=numpy_crps(f,fy)/stats['scale'];prob=(f[:,None]>stats['thresholds'][None]).mean(0);br=((prob-(fy[None]>stats['thresholds']))**2).mean()
            measures=dict(es=es,vs=vs,functional_crps=fc.mean(),brier=br)
            # 64 fixed spatial/temporal points independently audit marginal CRPS.
            point=np.linspace(0,target.size-1,64,dtype=int);small=numpy_crps(flat[:,point],target[point])
            ours=empirical_crps(tensor.flatten(1)[:,point],d['y'].flatten()[point]).numpy()
            errors.append(dict(split=split,index=i,method=name,metric='marginal_64points',error=float(np.max(np.abs(small-ours)))))
            for metric,value in measures.items():errors.append(dict(split=split,index=i,method=name,metric=metric,error=abs(float(value)-scores_saved[name][metric])))
    return errors

def compare(split):
    df=pd.read_csv(OUT/split/'per_trajectory.csv');selected=json.loads((OUT/'selection.json').read_text())['selected']
    metrics=['es','vs','functional_crps','brier']+['crps_'+n for n in FUNCTIONS]
    methods=[m for m in ('matched',selected) if m in df.method.unique()];rows=[];rng=np.random.RandomState(42)
    for method in sorted(set(methods)):
        a=df[df.method==method].set_index('file')
        for baseline in ('independent','empirical','ecc32','native32'):
            b=df[df.method==baseline].set_index('file').reindex(a.index)
            for metric in metrics:
                diff=(b[metric]-a[metric]).to_numpy();boot=diff[rng.randint(len(diff),size=(10000,len(diff)))].mean(1)
                rows.append(dict(method=method,baseline=baseline,metric=metric,improvement=diff.mean(),relative_percent=100*diff.mean()/b[metric].mean(),ci_low=np.quantile(boot,.025),ci_high=np.quantile(boot,.975),wins=int((diff>0).sum()),trajectories=len(diff)))
    pd.DataFrame(rows).to_csv(OUT/split/'paired.csv',index=False)
    df.groupby(['method','group'])[metrics].mean().to_csv(OUT/split/'per_re.csv')

def main():
    files=json.loads(MANIFEST.read_text())['files'];split_audit={}
    for a in ('train','calibration','development'):
        for b in files:
            if a!=b:
                assert {r['file'] for r in files[a]}.isdisjoint({r['file'] for r in files[b]})
                assert {r['group'] for r in files[a]}.isdisjoint({r['group'] for r in files[b]})
    lib=load(OUT/'library.pt');train_names={r['file'] for r in files['train']}
    assert {m['file'] for m in lib['meta']}<=train_names
    assert len({lib['meta'][i]['file'] for i in lib['unconditional'].tolist()})==N
    errors=[]
    for split in ('calibration','development'):
        if not (OUT/split/'summary.csv').exists():continue
        parts=sorted((OUT/split/'parts').glob('*.pt'));count=90 if split=='calibration' else 108
        assert len(parts)==count,(split,len(parts))
        maxerr=0.;max_crps=0.
        for path in parts:
            v=load(path);maxerr=max(maxerr,max(v['marginal_error'].values()))
            vals=[s['marginal_crps'] for k,s in v['scores'].items() if k!='native32'];max_crps=max(max_crps,max(vals)-min(vals))
            ids=v['matched_indices'].tolist();assert len(ids)==N and len({lib['meta'][i]['file'] for i in ids})==N
            assert v['meta']['file'] in {r['file'] for r in files[split]}
            for shift in v['shifts'].values():assert shift[...,0].abs().max()<=4 and shift[...,1].abs().max()<=2
        assert maxerr==0.;assert max_crps<1e-6
        split_audit[split]=dict(windows=count,identical_sorted_marginals_max_error=maxerr,marginal_crps_max_difference=max_crps)
        errors.extend(numerical(split));compare(split)
    edf=pd.DataFrame(errors);edf.to_csv(OUT/'independent_numerical.csv',index=False)
    assert edf.error.max()<3e-5,edf.loc[edf.error.idxmax()].to_dict()
    dump(OUT/'audit.json',dict(passed=True,splits=split_audit,numerical_max_error=edf.error.max(),data='original RealPDE; disjoint original trajectory and Re partitions',training_seed=42,bootstrap_unit='trajectory (Re shared, exploratory intervals)',code_sha256={p.name:digest(p) for p in Path(__file__).resolve().parent.glob('*.py')}))
    print(json.dumps(split_audit,indent=2),flush=True)

if __name__=='__main__':main()
