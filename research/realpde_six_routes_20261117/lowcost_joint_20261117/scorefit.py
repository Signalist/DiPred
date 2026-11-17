"""Fit only spatial/temporal correlation scales using train joint scores."""
from study import *
import study
FOLDER=OUT/'conditional_scores/scorefit'

@torch.no_grad()
def main():
    setup();FOLDER.mkdir(parents=True,exist_ok=True)
    d=load(SOURCE/'train.pt','cuda');study.OUT=OUT/'conditional_scores/ar';gen=Generator();pred=Predictor(library=False);stats=gen.lib['stats']
    old=legacy_exp.functionals
    def funcs(z,p,n):
        out=old(z,p,n);a=z*n['std']+n['mean'];v=(a[:,1,:,1:-1,2:]-a[:,1,:,1:-1,:-2])/(2*DX)+(a[:,0,:,2:,1:-1]-a[:,0,:,:-2,1:-1])/(2*DX)
        out[:,2]=v.square().mean((-1,-2)).amax(-1);return out
    legacy_exp.functionals=funcs
    configs=[]
    for space in (.125,.25,.5,1.):
        for temporal in (.25,.5,1.):
            for mixed in (False,True):configs.append(dict(space=space,temporal=temporal,method='ar__mixed' if mixed else 'ar__plain'))
    names=sorted({m['file'] for m in d['meta']});ids=[]
    for name in names:
        ix=sorted([i for i,m in enumerate(d['meta']) if m['file']==name],key=lambda i:d['meta'][i]['start'])
        ids.extend([ix[len(ix)//4],ix[3*len(ix)//4]])
    dump(FOLDER/'protocol.json',dict(seed=42,selection='Only training scores select the configuration. Calibration is a gate, never used to refit these scales.',configs=configs,training_windows=ids,trajectories=len(names),window_rule='two fixed quartile positions per training trajectory',loss='mean of train ES/VS/functional-CRPS ratios relative to train smooth, trajectory equal',motivation='Moment-fitted conditional covariance failed calibration; test direct scoring rule training without more model capacity.'))
    original=gen.ar['mean'].clone();rows=[]
    for counter,i in enumerate(ids):
        path=FOLDER/f'train_{counter:03d}.pt'
        if path.exists():rows.extend(load(path));continue
        x=d['x'][i:i+1];p,values,desc=pred.issue(x);native=torch.stack([pred.pred.rollout(x,500000+k,sign=1)[0] for k in range(32)])
        baseline=baselines(values,native,'smooth');base,_=scores(baseline,d['y'][i],p,pred.n,stats);block=[]
        for j,cfg in enumerate(configs):
            rho=original.clone();rho[:2]=rho[:2].sign()*rho[:2].abs()**cfg['space'];rho[2]=rho[2].sign()*rho[2].abs()**cfg['temporal'];gen.ar['mean']=rho.clamp(-.995,.995)
            z=couple(values,gen.sample(cfg['method'],desc,420000+i));sc,_=scores(z,d['y'][i],p,pred.n,stats)
            block.append(dict(config=j,**d['meta'][i],**{k:sc[k] for k in ('es','vs','functional_crps')},**{'base_'+k:base[k] for k in ('es','vs','functional_crps')}))
        save(block,path);rows.extend(block)
        if counter%10==0:print('SCOREFIT',counter,len(ids),flush=True)
    df=pd.DataFrame(rows);df.to_csv(FOLDER/'train_grid.csv',index=False)
    summary=df.groupby(['config','file']).mean(numeric_only=True).groupby('config').mean();summary['objective']=sum(summary[k]/summary['base_'+k] for k in ('es','vs','functional_crps'))/3
    chosen=int(summary.objective.idxmin());cfg=configs[chosen];rho=original.clone();rho[:2]=rho[:2].sign()*rho[:2].abs()**cfg['space'];rho[2]=rho[2].sign()*rho[2].abs()**cfg['temporal'];gen.ar['mean']=rho.clamp(-.995,.995)
    save(gen.ar,FOLDER/'ar_model.pt');link=FOLDER/'library.pt'
    if not link.exists():link.symlink_to(OUT/'conditional_scores/library.pt')
    dump(FOLDER/'fit.json',dict(selected=cfg,config_index=chosen,training_objective=summary.loc[chosen].to_dict(),rho=gen.ar['mean'].tolist(),criterion='train scores only',seed=42));summary.to_csv(FOLDER/'train_summary.csv')
    print('SELECTED',cfg,gen.ar['mean'].tolist(),summary.loc[chosen].to_dict(),flush=True)

if __name__=='__main__':main()
