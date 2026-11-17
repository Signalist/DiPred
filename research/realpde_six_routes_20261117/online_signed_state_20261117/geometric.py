"""A seven-dimensional online state attached to current forecast geometry."""
from state import *
GEO=OUT/'geometric'
IDENTIFIER='nonlinear'

def feedback_coefficients(split):
    source=SOURCE/f'{split}_oracle.pt';destination=GEO/f'{split}_identified.pt'
    if source.exists():
        d=load(source)
        return dict(d,coefficients=d['tangent_coefficients']) if IDENTIFIER=='linear' else d
    if destination.exists():return load(destination)
    raw=load(SOURCE/f'{split}.pt');parts=[]
    for i in range(0,len(raw['meta']),8):
        c,_=(st.linear_oracle if IDENTIFIER=='linear' else st.optimize_oracle)(raw['p'][i:i+8].cuda(),raw['y'][i:i+8].cuda());parts.append(c.cpu())
    d=dict(coefficients=torch.cat(parts),meta=raw['meta'],scope='Replay observation labels; consumed only after each full-block reveal; not current forecast input');save(d,destination);return d

def fit():
    d=feedback_coefficients('train');c=d['coefficients'].numpy().astype(float);meta=d['meta'];mean=c.mean(0);sd=c.std(0).clip(.01);z=(c-mean)/sd;pairs=[(a,b) for g in group_ids(meta) for a,b in zip(g[:-1],g[1:]) if meta[b]['start']-meta[a]['start']==20];a=np.array([a for a,b in pairs]);b=np.array([b for a,b in pairs]);mat=np.linalg.solve(z[a].T@z[a]+.1*len(a)*np.eye(7),z[a].T@z[b]).T;mat*=min(1.,.98/np.linalg.svd(mat,compute_uv=False)[0]);innovation=z[b]-z[a]@mat.T;q=innovation.T@innovation/len(a)+np.eye(7)*1e-5
    save(dict(mean=mean,scale=sd,a=mat,q=q),GEO/'model.pt');dump(GEO/'fit.json',dict(training_windows=len(c),transitions=len(a),max_singular_value=float(np.linalg.svd(mat,compute_uv=False)[0]),source_coefficients_sha256=digest(SOURCE/'train_oracle.pt'),protocol_sha256=digest(W/'online_signed_state_20261117/GEOMETRIC_PROTOCOL.md')))

class GeometricState:
    def __init__(self,config,model):self.cfg=config;self.m=model;self.history=[];self.z=np.zeros(7);self.cov=np.eye(7);self.last=None;self.pending=None
    def issue(self,start):
        if self.pending is not None:raise RuntimeError('Pending full-block feedback')
        family=self.cfg['family'];estimate=np.zeros(7);steps=0 if self.last is None else (start-self.last)//20
        if self.last is not None and (start<=self.last or (start-self.last)%20):raise ValueError('Invalid issue time')
        if family=='mean':estimate=self.m['mean'].copy()
        elif family=='rolling' and self.history:estimate=np.mean(self.history[-self.cfg['memory']:],0)
        elif family=='ema':estimate=self.z.copy()
        elif family in ('rw','var'):
            a=np.eye(7) if family=='rw' else self.m['a'];q=np.eye(7)*self.cfg.get('process',.1) if family=='rw' else self.m['q']
            for _ in range(steps):self.z=a@self.z;self.cov=a@self.cov@a.T+q
            estimate=self.m['mean']+self.m['scale']*self.z
        self.pending=start;return np.clip(estimate,-np.array(st.LIMITS),np.array(st.LIMITS))*self.cfg['shrink']
    def reveal(self,c,time):
        if self.pending is None or time<self.pending+40:raise ValueError('Early/unissued target')
        family=self.cfg['family']
        if family=='ema':self.z=(1-self.cfg['rate'])*self.z+self.cfg['rate']*c
        elif family in ('rw','var'):
            gain=np.linalg.solve((self.cov+np.eye(7)*self.cfg.get('noise',1.)).T,self.cov.T).T;self.z+=gain@((c-self.m['mean'])/self.m['scale']-self.z);r=np.eye(7)*self.cfg.get('noise',1.);self.cov=(np.eye(7)-gain)@self.cov@(np.eye(7)-gain).T+gain@r@gain.T
        self.history.append(c.copy());self.history=self.history[-8:];self.last=self.pending;self.pending=None
    def skip_feedback(self):
        if self.pending is None:raise RuntimeError('No pending issue')
        self.last=self.pending;self.pending=None

def configs():
    cfg=[dict(family='zero'),dict(family='mean')]
    cfg += [dict(family='rolling',memory=n) for n in (1,4,8)]
    cfg += [dict(family='ema',rate=r) for r in (.1,.5,1.)]
    cfg += [dict(family='rw',process=p,noise=1.) for p in (.01,.1,1.)]
    cfg += [dict(family='var',noise=r) for r in (.25,1.,4.)]
    return [dict(c,shrink=g) for c in cfg for g in ((1.,) if c['family']=='zero' else (0.,.25,.5,1.))]

def issue(d,cfg,model):
    output=np.zeros((len(d['meta']),7));coef=d['coefficients'].numpy()
    for ix in group_ids(d['meta']):
        state=GeometricState(cfg,model)
        for i in ix:
            start=d['meta'][i]['start'];output[i]=state.issue(start);state.reveal(coef[i],start+40)
    return output

@torch.no_grad()
def errors(raw,c):
    values=[]
    for i in range(0,len(c),16):
        q=st.transform(raw['p'][i:i+16].cuda(),torch.tensor(c[i:i+16],dtype=torch.float32,device='cuda'));values.append(st.mse(q,raw['y'][i:i+16].cuda()).cpu().numpy())
    return np.concatenate(values)

def calibrate():
    d=feedback_coefficients('calibration');raw=load(SOURCE/'calibration.pt');model=load(GEO/'model.pt');rows=[];selected={}
    for cfg in configs():
        c=issue(d,cfg,model);e=errors(raw,c);loss=float(summary(e,d['meta']));rows.append(dict(**cfg,mse=loss));name=cfg['family']
        if name not in selected or loss<selected[name]['mse']:selected[name]=dict(config=cfg,mse=loss)
    pd.DataFrame(rows).to_csv(GEO/'calibration_grid.csv',index=False);dump(GEO/'selection.json',dict(models=selected,chosen=min([k for k in selected if k not in ('zero','mean')],key=lambda k:selected[k]['mse']),criterion='natural calibration MSE only',model_sha256=digest(GEO/'model.pt'),protocol_sha256=digest(W/'online_signed_state_20261117/GEOMETRIC_PROTOCOL.md')));print(selected,flush=True)

def evaluate(split):
    selection=json.loads((GEO/'selection.json').read_text());d=feedback_coefficients(split);raw=load(SOURCE/f'{split}.pt');model=load(GEO/'model.pt');rows=[];issued={}
    for method,cfg in selection['models'].items():
        c=issue(d,cfg['config'],model);issued[method]=c;e=errors(raw,c)
        for i,m in enumerate(d['meta']):rows.append(dict(**m,method='geo_'+method,scenario='natural',mse=e[i]))
    save(issued,GEO/f'{split}_issued.pt');df=pd.DataFrame(rows);df.to_csv(GEO/f'{split}_windows.csv',index=False);v=df.groupby(['method','file']).mse.mean().groupby('method').mean();print(split,((1-v/v['geo_zero'])*100).to_dict(),flush=True)

if __name__=='__main__':
    setup();p=argparse.ArgumentParser();p.add_argument('action',choices=['fit','calibrate','evaluate']);p.add_argument('--split',default='development');p.add_argument('--linear',action='store_true');a=p.parse_args()
    if a.linear:GEO=OUT/'linear_geometry';IDENTIFIER='linear'
    GEO.mkdir(parents=True,exist_ok=True);{'fit':fit,'calibrate':calibrate,'evaluate':lambda:evaluate(a.split)}[a.action]()
