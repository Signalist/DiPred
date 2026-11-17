"""Audit sensitivity/localization variants without treating pixels as replicates."""
from experiment import *
import pandas as pd
import audit as independent

def analyze(folder):
    dest=OUT/folder;selection=json.loads((dest/'selection.json').read_text());chosen=selection['selected'];errors=[];checks={}
    if folder in ('physics','advection','helmholtz_filter'):
        old=independent.numpy_functionals
        def physical(z,p,n):
            f=old(z,p,n);a=z*n['std']+n['mean'];dx=.003421664
            vort=(a[:,1,:,1:-1,2:]-a[:,1,:,1:-1,:-2])/(2*dx)+(a[:,0,:,2:,1:-1]-a[:,0,:,:-2,1:-1])/(2*dx)
            f[:,2]=(vort*vort).mean((-1,-2)).max(-1);return f
        independent.numpy_functionals=physical
    for split in ('calibration','development','late','future'):
        if not (dest/split/'summary.csv').exists():continue
        parts=sorted((dest/split/'parts').glob('*.pt'));num={'calibration':90,'development':108}.get(split)
        if num:assert len(parts)==num
        errs=[];half=[];full=[];times=[];prob=[]
        for path in parts:
            v=load(path);errs.extend(v['marginal_error'].values())
            if 'derivative_halfstep_relative_rmse' in v:
                half.append(v['derivative_halfstep_relative_rmse']);full.append(v['full_noise_linearization_relative_rmse']);times.append(v['timing'])
            for method,func in v['functionals'].items():
                prob.append(dict(method=method,**v['meta'],**{f'truth_{name}':float(func['truth'][j]) for j,name in enumerate(FUNCTIONS)}))
        assert max(errs)==0.
        checks[split]=dict(windows=len(parts),identical_marginal_max_error=max(errs))
        if half:checks[split].update(halfstep_relative_rmse_mean=float(np.mean(half)),full_noise_linearization_relative_rmse_mean=float(np.mean(full)))
        if times:pd.DataFrame(times).to_csv(dest/split/'sampling_times.csv',index=False)
        pd.DataFrame(prob).to_csv(dest/split/'functional_truth.csv',index=False)
        independent.OUT=dest;errors.extend(independent.numerical(split))
        df=pd.read_csv(dest/split/'per_trajectory.csv');metrics=['es','vs','functional_crps','brier']+['crps_'+f for f in FUNCTIONS]
        df.groupby(['method','group'])[metrics].mean().to_csv(dest/split/'per_re.csv')
        comparisons=[];rng=np.random.RandomState(42)
        methods=list(dict.fromkeys([chosen,selection.get('nonlinear_selected',chosen),selection.get('smooth_selected',chosen),selection.get('matched_smooth',chosen)]))
        for method in methods:
            a=df[df.method==method].set_index('file')
            for baseline in list(dict.fromkeys(['ecc32','empirical','native_basis8','native_basis16','smooth_ecc_b50_i1']+[x for x in methods if x!=method])):
                if baseline==method:continue
                if baseline not in df.method.unique():continue
                b=df[df.method==baseline].set_index('file').reindex(a.index)
                for metric in metrics:
                    diff=(b[metric]-a[metric]).to_numpy();boot=diff[rng.randint(len(diff),size=(10000,len(diff)))].mean(1)
                    comparisons.append(dict(method=method,baseline=baseline,metric=metric,improvement=diff.mean(),relative_percent=100*diff.mean()/b[metric].mean(),ci_low=np.quantile(boot,.025),ci_high=np.quantile(boot,.975),wins=int((diff>0).sum()),trajectories=len(diff)))
        pd.DataFrame(comparisons).to_csv(dest/split/'paired.csv',index=False)
    edf=pd.DataFrame(errors);edf.to_csv(dest/'independent_numerical.csv',index=False);assert edf.error.max()<3e-5
    dump(dest/'audit.json',dict(passed=True,checks=checks,numerical_max_error=edf.error.max(),selection=selection,seed=42,tf32_override=0,bootstrap='trajectory; shared Re limits interpretation'))
    print(folder,checks,'max independent error',edf.error.max(),flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('folder',choices=['sensitivity','localized','physics','advection','helmholtz_filter']);args=a.parse_args();analyze(args.folder)
