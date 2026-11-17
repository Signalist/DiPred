"""Empirical single-observation benefit prediction; common update remains fixed."""
from predictive import *
CRITIC=BASE/'critic'
PROJECTION=W/'runs/online_signed_state_20261117/basis.pt'

def project_feature(descriptor,projection):
    raw=torch.cat([descriptor[k] for k in ('X','P','F_last')],1).double();return ((((raw-projection['mean'])/projection['std'])@projection['basis'])/projection['scale']).clamp(-10,10)[0]

@torch.no_grad()
def features():
    proj=load(PROJECTION,'cuda')['projection'];net=joint.st.model();maximum=0.
    for split in ('train','calibration','development','late','future'):
        d=load(SOURCE/f'{split}.pt');cached=load(BASE/f'{split}.pt');values=[]
        for i in range(len(d['meta'])):
            x=d['x'][i:i+1].cuda();p,taps=joint.st.capture(net,x);desc=dict(X=joint.st.pooled(x),P=joint.st.pooled(p),F_last=taps['F_last']);values.append(project_feature(desc,proj).cpu());maximum=max(maximum,float((p[0].cpu()-cached['p'][i]).abs().max()))
        save(torch.stack(values),CRITIC/f'features_{split}.pt');print('CRITIC_FEATURES',split,flush=True)
    assert maximum==0;dump(CRITIC/'feature_audit.json',dict(actual_single_window=True,point_replay_max_error=maximum,projection_sha256=digest(PROJECTION)))

def single_gain(assim,pack):
    ix=torch.arange(128,device='cuda')[:,None]*2+torch.arange(2,device='cuda')[None];s=pack['cov'][ix[:,:,None],ix[:,None,:]];h=pack['h'];v=torch.linalg.solve(s,h);return (2*assim.cfg['gain']-assim.cfg['gain']**2)*((h@pack['g'])*v).sum((1,2))

def inputs(assim,pack,scale,risk,f):
    unit=scale[:,5:].double().square().mean().clamp_min(1e-12);single=single_gain(assim,pack);local=torch.stack([single/unit,risk.double().sum(-1)/unit],-1);return local,unit,single

@torch.no_grad()
def fit_critic():
    d=load(BASE/'train.pt','cuda');truth=load(SOURCE/'train.pt','cuda')['y'];f=load(CRITIC/'features_train.pt','cuda');assim=PredictiveAssimilator(json.loads((ROOT/'selection.json').read_text())['config']);local=[];targets=[];units=[];w=balanced(d['meta'])
    for i,meta in enumerate(d['meta']):
        pack=assim.pack(d['scale'][i]);loc,unit,single=inputs(assim,pack,d['scale'][i],d['risk'][i],f[i]);r=(truth[i]-d['p'][i])*norm()['std'][0];base,dot=loss_stats(pack,r);errors=observation_rows(r).double();ix=torch.arange(128,device='cuda')[:,None]*2+torch.arange(2,device='cuda')[None];s=pack['cov'][ix[:,:,None],ix[:,None,:]];z=assim.cfg['gain']*(pack['h'].transpose(1,2)@torch.linalg.solve(s,errors[:,:,None]))[:,:,0];reduction=2*(z*dot).sum(-1)-((z@pack['g'])*z).sum(-1);local.append(loc);targets.append(reduction/unit);units.append(unit)
        if i%100==0:print('CRITIC_LABELS',i,len(d['meta']),flush=True)
    local=torch.stack(local);target=torch.stack(targets);mu=(w[:,None,None]*local).sum(0);sd=(w[:,None,None]*(local-mu).square()).sum(0).sqrt().clamp_min(.01);x=torch.cat([torch.ones(len(f),128,1,device='cuda',dtype=torch.float64),f[:,None,:].expand(-1,128,-1),(local-mu)/sd],-1);xt=x.transpose(0,1).transpose(1,2);weighted=x*w[:,None,None];gram=xt@weighted.transpose(0,1);rhs=xt@(target*w[:,None]).T[:,:,None];weights={};reg=torch.eye(19,device='cuda',dtype=torch.float64);reg[0,0]=0
    for alpha in (.01,.1,1.):weights[str(alpha)]=torch.linalg.solve(gram+alpha*reg,rhs)[:,:,0].cpu()
    save(dict(weights=weights,local_mean=mu.cpu(),local_scale=sd.cpu(),projection=load(PROJECTION)['projection'],meta=d['meta']),CRITIC/'model.pt');save(dict(features=f.cpu(),local=local.cpu(),target=target.cpu(),unit=torch.stack(units).cpu(),meta=d['meta']),CRITIC/'training_labels.pt');dump(CRITIC/'fit.json',dict(coefficients=128*19,training_windows=819,trajectory_balanced=True,common_assimilation_selection_sha256=digest(ROOT/'selection.json'),protocol_sha256=digest(Path(__file__).with_name('CRITIC_PROTOCOL.md')),projection_sha256=digest(PROJECTION)));print('CRITIC_FIT_DONE',flush=True)

def scores(assim,pack,scale,risk,f,model,alpha):
    local,unit,single=inputs(assim,pack,scale,risk,f);x=torch.cat([torch.ones(128,1,device='cuda',dtype=torch.float64),f[None].expand(128,-1),(local-model['local_mean'])/model['local_scale']],-1);gain=(x*model['weights'][str(alpha)]).sum(-1)*unit;return gain,single

def select(assim,pack,gain,single,penalty,maximum=32):
    if penalty==0:return torch.argsort(gain,descending=True,stable=True)[:maximum].cpu().numpy()
    covariance=pack['cov'].clone();cross=pack['h'].reshape(256,assim.rank).T.clone();chosen=[];ixall=torch.arange(128,device='cuda')[:,None]*2+torch.arange(2,device='cuda')[None];factor=2*assim.cfg['gain']-assim.cfg['gain']**2
    for _ in range(maximum):
        a=cross.T.reshape(128,2,assim.rank);s=covariance[ixall[:,:,None],ixall[:,None,:]].clone()
        if chosen:s[chosen]=torch.eye(2,device='cuda',dtype=torch.float64)
        sol=torch.linalg.solve(s,a);conditional=factor*((a@pack['g'])*sol).sum((1,2));score=gain-penalty*(single-conditional).clamp_min(0);score[chosen]=-torch.inf;j=int(score.argmax());ix=ixall[j];v=torch.linalg.solve(s[j],covariance[ix,:]);cross-=cross[:,ix]@v;covariance-=covariance[:,ix]@v;covariance=(covariance+covariance.T)*.5;chosen.append(j)
    return np.asarray(chosen,dtype=int)

@torch.no_grad()
def calibrate_critic():
    d=load(BASE/'calibration.pt','cuda');truth=load(SOURCE/'calibration.pt','cuda')['y'];f=load(CRITIC/'features_calibration.pt','cuda');model=load(CRITIC/'model.pt','cuda');assim=PredictiveAssimilator(json.loads((ROOT/'selection.json').read_text())['config']);rows=[]
    for i,meta in enumerate(d['meta']):
        pack=assim.pack(d['scale'][i]);r=(truth[i]-d['p'][i])*norm()['std'][0];base,dot=loss_stats(pack,r);observed=observation_rows(r)
        for alpha in (.01,.1,1.):
            gain,single=scores(assim,pack,d['scale'][i],d['risk'][i],f[i],model,alpha)
            for penalty in (0.,.5,1.):
                indices=select(assim,pack,gain,single,penalty,8);z=assim.correction(pack,indices,observed[indices]);rows.append(dict(**meta,alpha=alpha,penalty=penalty,mse=mse_from_latent(pack,base,dot,z)))
    df=pd.DataFrame(rows);df.to_csv(CRITIC/'calibration_grid.csv',index=False);loss=df.groupby(['alpha','penalty','file']).mse.mean().groupby(['alpha','penalty']).mean();alpha,penalty=loss.idxmin();dump(CRITIC/'selection.json',dict(alpha=float(alpha),penalty=float(penalty),criterion='calibration physical future15 MSE at8 sites only',mse=float(loss.min()),protocol_sha256=digest(Path(__file__).with_name('CRITIC_PROTOCOL.md')),model_sha256=digest(CRITIC/'model.pt'),common_assimilation_selection_sha256=digest(ROOT/'selection.json')));print('CRITIC_SELECTED',alpha,penalty,float(loss.min()),flush=True)

@torch.no_grad()
def evaluate_critic(split):
    d=load(BASE/f'{split}.pt','cuda');truth=load(SOURCE/f'{split}.pt','cuda')['y'];f=load(CRITIC/f'features_{split}.pt','cuda');model=load(CRITIC/'model.pt','cuda');selection=json.loads((CRITIC/'selection.json').read_text());assim=PredictiveAssimilator(json.loads((ROOT/'selection.json').read_text())['config']);rows=[];requests=[];events=[];folder=CRITIC/split;folder.mkdir(exist_ok=True)
    for i,meta in enumerate(d['meta']):
        pack=assim.pack(d['scale'][i]);gain,single=scores(assim,pack,d['scale'][i],d['risk'][i],f[i],model,selection['alpha']);order=select(assim,pack,gain,single,selection['penalty']);requests.append(dict(meta=meta,order=order.tolist(),predicted_single_gains=gain.cpu().tolist()));events.append(dict(**meta,event='issue_all_requests',time=meta['start']+20));observed=(observation_rows(truth[i])-observation_rows(d['p'][i]))*norm()['std'][0,:,0,0,0];updates={b:assim.correction(pack,order[:b],observed[order[:b]]) for b in BUDGETS};events.append(dict(**meta,event='selected_observations_and_reissue',time=meta['start']+25));r=(truth[i]-d['p'][i])*norm()['std'][0];base,dot=loss_stats(pack,r)
        for budget,z in updates.items():
            loss=mse_from_latent(pack,base,dot,z);rows.append(dict(**meta,method='critic',budget=budget,scalar_measurements=2*budget,base_mse=base,mse=loss,realized_reduction=base-loss,predicted_negative_selected=int((gain[order[:budget]]<0).sum())))
            if i in (0,4) and budget==8:save(dict(p=d['p'][i].cpu(),scale=d['scale'][i].cpu(),indices=order[:8],observed_physical=(observed[order[:8]]+observation_rows(d['p'][i]*norm()['std'][0]+norm()['mean'][0])[order[:8]]).cpu(),truth=truth[i].cpu(),z=z.cpu(),analytic_mse=loss,meta=meta,config=assim.cfg),folder/f'audit_{i:04d}_critic.pt')
        events.append(dict(**meta,event='future15_scoring',time=meta['start']+40))
    df=pd.DataFrame(rows);df.to_csv(folder/'windows.csv',index=False);tr=df.groupby(['method','budget','file','group'])[['base_mse','mse','realized_reduction','predicted_negative_selected']].mean().reset_index();tr.to_csv(folder/'trajectories.csv',index=False);tr.groupby(['method','budget'])[['base_mse','mse','realized_reduction','predicted_negative_selected']].mean().to_csv(folder/'summary.csv');save(requests,folder/'requests.pt');(folder/'events.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in events));print('CRITIC_EVALUATED',split,flush=True)

class Planner:
    def __init__(self):self.model=load(CRITIC/'model.pt','cuda');self.selection=json.loads((CRITIC/'selection.json').read_text())
    def __call__(self,assim,pack,scale,risk,descriptor,budget):
        f=project_feature(descriptor,self.model['projection']);gain,single=scores(assim,pack,scale,risk,f,self.model,self.selection['alpha']);return select(assim,pack,gain,single,self.selection['penalty'],budget)

def make_critic_predictor(budget=8):
    from active_stream import SparseUpdatePredictor
    return SparseUpdatePredictor('critic',budget,assimilator=PredictiveAssimilator(json.loads((ROOT/'selection.json').read_text())['config']),planner=Planner())

if __name__=='__main__':
    setup();CRITIC.mkdir(parents=True,exist_ok=True);p=argparse.ArgumentParser();p.add_argument('action',choices=['features','fit','calibrate','evaluate']);p.add_argument('--split',default='development');args=p.parse_args();{'features':features,'fit':fit_critic,'calibrate':calibrate_critic,'evaluate':lambda:evaluate_critic(args.split)}[args.action]()
