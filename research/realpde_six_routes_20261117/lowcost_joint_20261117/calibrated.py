"""Identical marginal calibration for all final joint-model comparisons."""
from study import *
import study
FOLDER=OUT/'common_calibration'
FINAL=OUT/'conditional_scores/scorefit'
LEVELS=(torch.arange(32)+.5)/32

def quantile_diagonal(a):
    # a [32,2,...]; each row uses its own target quantile, no pooled tails.
    return torch.stack([torch.quantile(a[k].flatten(1),float(LEVELS[k]),dim=1) for k in range(32)])

@torch.no_grad()
def fit():
    FOLDER.mkdir(parents=True,exist_ok=True);d=load(SOURCE/'calibration.pt','cuda');pred=Predictor(library=False);errors=[]
    for i in range(len(d['x'])):
        _,q,_=pred.issue(d['x'][i:i+1]);errors.append((d['y'][i]-q)[...,::4,::4].cpu())
    e=torch.stack(errors,2).cuda();offsets=quantile_diagonal(e)
    save(dict(offsets=offsets,levels=LEVELS),FOLDER/'offsets.pt');dump(FOLDER/'protocol.json',dict(seed=42,fit='90 calibration windows, 10 trajectories,32 quantile-specific component offsets; spatial stride4',online='For each split/trajectory initialize from calibration offsets; after full20 truth reveal update offsets=.5 old+.5 quantile(Y-Qraw); no skipped-window feedback and no retroactive changes.',comparison='All methods use exactly the same issued quantile nodes, sort after offsets; frozen copula parameters; no recalibration-dependent method selection.',scope='Empirical calibration, no finite-sample conditional coverage guarantee. This follows the same two-tail location-update principle with32 common nodes instead of49 head pairs.'))
    print('OFFSETS_READY',offsets.min().item(),offsets.max().item(),flush=True)

@torch.no_grad()
def run(split):
    study.OUT=FINAL;pred=Predictor();d=load(SOURCE/f'{split}.pt','cuda');initial=load(FOLDER/'offsets.pt','cuda')['offsets'];states={};rows=[]
    stats=pred.generator.lib['stats'];old=legacy_exp.functionals
    def funcs(z,p,n):
        a=z*n['std']+n['mean'];r=old(z,p,n);v=(a[:,1,:,1:-1,2:]-a[:,1,:,1:-1,:-2])/(2*DX)+(a[:,0,:,2:,1:-1]-a[:,0,:,:-2,1:-1])/(2*DX);r[:,2]=v.square().mean((-1,-2)).amax(-1);return r
    legacy_exp.functionals=funcs
    folder=FOLDER/split;folder.mkdir(exist_ok=True);folder.joinpath('events.jsonl').write_text('')
    # Existing caches are ordered in time; explicit sorting makes the rule visible.
    order=sorted(range(len(d['meta'])),key=lambda i:(d['meta'][i]['file'],d['meta'][i]['start']))
    maximum_margin=0.;issued=[]
    for count,i in enumerate(order):
        meta=d['meta'][i];name=meta['file'];state=states.get(name,dict(offsets=initial.clone(),last_end=-1,revealed=0))
        assert meta['start']+20>=state['last_end'];x=d['x'][i:i+1];p,raw,desc=pred.issue(x)
        native=torch.stack([pred.pred.rollout(x,500000+k,sign=1)[0] for k in range(32)])
        latent=pred.generator.sample('ar__mixed',desc,seed=420000+i)
        values={'raw':raw,'fixed':(raw+initial[:,:,None,None,None]).sort(0).values,'online':(raw+state['offsets'][:,:,None,None,None]).sort(0).values}
        # Every output is issued before scoring/feedback sees this window's Y.
        fields={}
        for mode,v in values.items():
            fields[mode]={m:baselines(v,native,m) for m in ('ecc32','smooth','secc3','necc3')}
            fields[mode]['lowcost']=couple(v,latent)
        with (folder/'events.jsonl').open('a') as f:f.write(json.dumps(dict(event='issue',**meta,previous_reveals=state['revealed']))+'\n')
        truth=d['y'][i]
        for mode,methods_ in fields.items():
            for method,z in methods_.items():
                sc,_=scores(z,truth,p,pred.n,stats);rows.append(dict(**meta,mode=mode,method=method,**sc))
                if method!='necc3':maximum_margin=max(maximum_margin,float((z.sort(0).values-values[mode]).abs().max()))
        estimate=quantile_diagonal((truth-raw)[...,::4,::4]);state['offsets']=.5*state['offsets']+.5*estimate;state['revealed']+=1;state['last_end']=meta['start']+40;states[name]=state
        with (folder/'events.jsonl').open('a') as f:f.write(json.dumps(dict(event='full20_reveal',**meta,revealed=state['revealed']))+'\n')
        if count%20==0:print('CALIBRATED',split,count,len(order),flush=True)
    df=pd.DataFrame(rows);df.to_csv(folder/'windows.csv',index=False);cols=[c for c in df if c not in ('file','group','start','method','mode')]
    tr=df.groupby(['mode','method','file','group'])[cols].mean().reset_index();tr.to_csv(folder/'trajectories.csv',index=False);summary=tr.groupby(['mode','method'])[cols].mean();summary.to_csv(folder/'summary.csv')
    dump(folder/'audit.json',dict(passed=maximum_margin==0,sorted_marginal_max_error=maximum_margin,windows=len(order),trajectories=len(states),all_issued_before_feedback=True,feedback_windows={k:v['revealed'] for k,v in states.items()},issue_time_convention='history start s has observed s:s+20, issue at s+20; target s+20:s+40; next consecutive window can issue at s+40 after this full reveal'))
    print(summary[['marginal_crps','es','vs','functional_crps']].to_string(),flush=True)

if __name__=='__main__':
    setup();a=argparse.ArgumentParser();a.add_argument('action',choices=['fit','development','late','future']);args=a.parse_args();fit() if args.action=='fit' else run(args.action)
