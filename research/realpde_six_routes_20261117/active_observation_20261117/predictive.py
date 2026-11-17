"""Low-rank future-error regression with full correlated sensor residuals."""
from observations import *
import observations as common
BASE=OUT;ROOT=BASE/'predictive'

@torch.no_grad()
def fit_predictive():
    d=load(BASE/'train.pt','cuda');truth=load(SOURCE/'train.pt','cuda')['y'];w=balanced(d['meta']);u=((truth-d['p'])*norm()['std']/d['scale']).double();o=observation_rows(u).reshape(len(u),256);target=u[:,:,5:].flatten(1);coo=o.T@(w[:,None]*o);cross=target.T@(w[:,None]*o);families={};rows=[]
    for shrink in (.1,.5,.9):
        covariance=(1-shrink)*coo+shrink*torch.diag(coo.diag());factor=torch.linalg.cholesky(covariance);white=torch.linalg.solve(factor,((1-shrink)*cross).T).T;left,s,right=torch.linalg.svd(white,full_matrices=False);b=(left[:,:32]*s[:32]).T.reshape(32,2,15,32,64);h=right[:32]@factor.T;families[str(shrink)]=dict(basis=b.cpu(),cross=h.cpu(),obs_cov=covariance.cpu());identity=h@torch.linalg.solve(covariance,h.T);err=float((identity-torch.eye(32,device='cuda')).abs().max());assert err<1e-8;rows.append(dict(shrink=shrink,rank32_fraction_predictable_energy=float(s[:32].square().sum()/s.square().sum()),latent_prior_identity_error=err));print('FIT_PREDICTIVE',rows[-1],flush=True)
    prior=load(BASE/'model.pt');save(dict(families=families,average_scale=prior['average_scale'],meta=d['meta']),ROOT/'model.pt');dump(ROOT/'fit.json',dict(seed=42,trajectory_balanced=True,rows=rows,main_protocol_sha256=digest(Path(__file__).with_name('PROTOCOL.md')),extension_protocol_sha256=digest(Path(__file__).with_name('PREDICTIVE_PROTOCOL.md')),source_sha256=digest(Path(__file__)),train_source_sha256=digest(SOURCE/'train.pt')))

class PredictiveAssimilator(Assimilator):
    def __init__(self,config,model=None):
        full=load(ROOT/'model.pt','cuda') if model is None else model;family=full['families'][str(config['shrink'])];self.model=dict(**family,average_scale=full['average_scale']);self.cfg=config;self.rank=config['rank'];self.b=self.model['basis'][:self.rank];self.h=self.model['cross'][:self.rank];self.pending=None;self.normalization=norm()
    def pack(self,scale):
        scale=scale.double();a=(self.b*scale[:,5:]).flatten(1);sd=observation_rows(scale);h=self.h.T.reshape(128,2,self.rank)*sd[:,:,None];flat=sd.flatten();cov=self.model['obs_cov']*flat[:,None]*flat[None,:];return dict(a=a,h=h,cov=cov,g=a@a.T/a.shape[1],irreducible=np.nan)
    def posterior(self,pack,indices):
        if len(indices)==0:return torch.eye(self.rank,device='cuda',dtype=torch.float64),torch.empty(self.rank,0,device='cuda',dtype=torch.float64)
        ix=(torch.as_tensor(indices,device='cuda',dtype=torch.long)[:,None]*2+torch.arange(2,device='cuda')[None]).flatten();h=pack['h'].reshape(256,self.rank)[ix];cov=pack['cov'][ix][:,ix];gain=torch.linalg.solve(cov,h).T;latent=torch.eye(self.rank,device='cuda',dtype=torch.float64)-gain@h;return latent,gain
    def greedy(self,pack,maximum=32):
        covariance=pack['cov'].clone();cross=pack['h'].reshape(256,self.rank).T.clone();selected=[];gains=[];blockix=torch.arange(128,device='cuda')[:,None]*2+torch.arange(2,device='cuda')[None]
        for _ in range(maximum):
            a=cross.T.reshape(128,2,self.rank);innovation=covariance[blockix[:,:,None],blockix[:,None,:]].clone()
            if selected:innovation[selected]=torch.eye(2,device='cuda',dtype=torch.float64)
            sol=torch.linalg.solve(innovation,a);benefit=((a@pack['g'])*sol).sum((1,2));benefit[selected]=-torch.inf;site=int(benefit.argmax());ix=blockix[site];solve=torch.linalg.solve(innovation[site],covariance[ix,:]);cross-=cross[:,ix]@solve;covariance-=covariance[:,ix]@solve;covariance=(covariance+covariance.T)*.5;selected.append(site);gains.append(float(benefit[site]))
        return np.asarray(selected,dtype=int),np.asarray(gains)

def predictive_configs():return [dict(rank=r,shrink=s,gain=g) for r in (8,16,32) for s in (.1,.5,.9) for g in (.5,1.)]

def make_predictor(method='voi',budget=8):
    from active_stream import SparseUpdatePredictor
    return SparseUpdatePredictor(method,budget,assimilator=PredictiveAssimilator(json.loads((ROOT/'selection.json').read_text())['config']))

def configure():
    ROOT.mkdir(exist_ok=True);common.OUT=ROOT;common.Assimilator=PredictiveAssimilator;common.configs=predictive_configs
    for split in ('train','calibration','development','late','future'):
        p=ROOT/f'{split}.pt'
        if not p.exists():p.symlink_to(BASE/f'{split}.pt')

if __name__=='__main__':
    setup();configure();p=argparse.ArgumentParser();p.add_argument('action',choices=['fit','calibrate','evaluate']);p.add_argument('--split',default='development');args=p.parse_args()
    if args.action=='fit':fit_predictive()
    elif args.action=='calibrate':
        common.calibrate();selection=json.loads((ROOT/'selection.json').read_text());selection['extension_protocol_sha256']=digest(Path(__file__).with_name('PREDICTIVE_PROTOCOL.md'));selection['predictive_source_sha256']=digest(Path(__file__));dump(ROOT/'selection.json',selection)
    else:
        common.evaluate(args.split);manifest=json.loads((ROOT/args.split/'provenance.json').read_text());manifest['predictive_source_sha256']=digest(Path(__file__));manifest['extension_protocol_sha256']=digest(Path(__file__).with_name('PREDICTIVE_PROTOCOL.md'));dump(ROOT/args.split/'provenance.json',manifest)
