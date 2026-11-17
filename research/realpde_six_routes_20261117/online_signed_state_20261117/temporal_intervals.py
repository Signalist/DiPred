"""Task-interval controls under common full20 feedback and charged warm-up."""
from probability import *
from common import records,read
# The system sklearn has an optional pkg_resources version parser whose
# package-directory scan hits unrelated unreadable system dist-info folders.
# Use sklearn's own ImportError fallback during this import only; no system edits.
_pkg_resources_previous=sys.modules.get('pkg_resources',None)
sys.modules['pkg_resources']=None
try:
    from sklearn.ensemble import RandomForestRegressor
finally:
    if _pkg_resources_previous is None:sys.modules.pop('pkg_resources',None)
    else:sys.modules['pkg_resources']=_pkg_resources_previous
ROOT=OUT/'temporal_intervals';TASKS=joint.legacy_exp.FUNCTIONS;ALPHA=.1

def collect():
    configure_scores();pred=Provider();initial=load(FOLDER/'offsets.pt','cuda')['offsets'];folder=ROOT/'parts';folder.mkdir(parents=True,exist_ok=True);events=[]
    for ri,rec in enumerate(records('development')):
        offsets={k:v.clone() for k,v in initial.items()}
        for j,start in enumerate(range(0,801,20)):
            meta={k:rec[k] for k in ('file','group')};meta['start']=start;data=(read(rec,start)[None].cuda()-pred.n['mean'])/pred.n['std'];p,raw,points=pred.issue(data[:,:,:20],meta,620000+ri*41+j);functions={m:{} for m in ('raw','fixed','online')};nodes={}
            with torch.no_grad():
                for k,z in raw.items():
                    q,order=z.sort(dim=0,stable=True);nodes[k]=q
                    for mode,offset in [('raw',None),('fixed',initial[k]),('online',offsets[k])]:
                        v=z if offset is None else torch.empty_like(z).scatter_(0,order,(q+offset[:,:,None,None,None]).sort(dim=0,stable=True).values)
                        functions[mode][k]=joint.legacy_exp.functionals(v,p,pred.n).cpu()
                pointfunc={k:joint.legacy_exp.functionals(v,p,pred.n)[0].cpu() for k,v in points.items()}
            events.append(dict(**meta,event='issue_all',previous_reveals=j));y=data[:,:,20:]
            with torch.no_grad():
                truth=joint.legacy_exp.functionals(y,p,pred.n)[0].cpu();mses={k:float(st.mse(v,y)[0]) for k,v in points.items()}
                for k,q in nodes.items():offsets[k]=.5*offsets[k]+.5*diagonal_quantile((y[0]-q)[...,::4,::4])
            save(dict(meta=meta,index=j,scored=j>=10,functions=functions,point=pointfunc,truth=truth,point_mse=mses),folder/f'{ri:02d}_{j:02d}.pt');pred.reveal(y,start+40);events.append(dict(**meta,event='full20_reveal',previous_reveals=j+1))
        print('DENSE_COLLECT',rec['file'],flush=True)
    (ROOT/'events.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in events));dump(ROOT/'collection.json',dict(trajectories=12,windows=492,warmup_windows=120,scored_windows=372,protocol_sha256=digest(Path(__file__).with_name('TEMPORAL_INTERVAL_PROTOCOL.md')),source_sha256=digest(Path(__file__))))

class AcMCPBlock:
    """The author's h=1 asymmetric recurrence, without future-series defaults."""
    def __init__(self,ki,lr=.1,scorecast=True):
        self.ki=np.asarray(ki);self.lr=lr;self.scorecast=scorecast;self.csat=2/np.pi*(np.ceil(np.log(1000)*.01)-1/np.log(1000));self.history=[];self.qts=np.zeros((2,len(ki)));self.qs=self.qts.copy();self.errors=self.qts.copy()
    def issue(self,point):return np.stack([point-self.qs[0],point+self.qs[1]])
    def reveal(self,error):
        tail=np.stack([-error,error])>self.qs;self.errors+=tail;self.history.append(error.copy());t=len(self.history);recent=np.asarray(self.history[-10:]);rate=self.lr if t==1 else self.lr*np.ptp(recent,axis=0);self.qts+=rate*(tail-ALPHA/2)
        arg=(self.errors-t*ALPHA/2)*np.log(t)/(self.csat*t);integ=np.where(arg>=np.pi/2,np.inf,np.where(arg<=-np.pi/2,-np.inf,np.tan(arg)))*self.ki;integ[:]=0 if t==1 else integ
        score=recent.mean(0) if self.scorecast and t>=10 else np.zeros_like(error);self.qs=self.qts+integ+np.stack([-score,score])

def qrf_interval(history,lag=3):
    """Standard empirical leaf weights; no cross-trajectory pseudo-history."""
    a=np.asarray(history);x=np.array([a[i-lag:i] for i in range(lag,len(a))]);y=a[lag:];query=a[-lag:][None];model=RandomForestRegressor(n_estimators=10,max_depth=2,random_state=42,n_jobs=1).fit(x,y);leaf=model.apply(x);queryleaf=model.apply(query)[0];weight=np.zeros(len(y))
    for k,node in enumerate(queryleaf):
        mask=leaf[:,k]==node;weight+=mask/mask.sum()/len(queryleaf)
    order=np.argsort(y,kind='stable');cumulative=np.cumsum(weight[order]);beta=np.linspace(0,ALPHA,5);quant=np.r_[beta,1-ALPHA+beta];idx=np.minimum(np.searchsorted(cumulative,quant,side='left'),len(y)-1);q=y[order[idx]];best=np.argmin(q[5:]-q[:5]);return np.array([q[best],q[best+5]])

def interval_metrics(bounds,y,scale):
    low,high=bounds;valid=low<=high;width=high-low;score=width+2/ALPHA*np.where(y<low,low-y,0)+2/ALPHA*np.where(y>high,y-high,0);score=np.where(valid,score,np.inf)
    return dict(coverage=((y>=low)&(y<=high)&valid).astype(float),width=width/scale,interval_score=score/scale,infinite=(~np.isfinite(bounds).all(0)).astype(float),reversed=(~valid).astype(float))

def calibrate_tasks():
    # Re-evaluate deterministic corrected tasks under the identical cold calibration
    # stream, because older scalar caches retained only the common baseline point.
    configure_scores();pred=Provider();d=load(SOURCE/'calibration.pt');residual={k:[] for k in NAMES}
    for ix in group_ids(d['meta']):
        for i in ix:
            p,fields,points=pred.issue(d['x'][i:i+1].cuda(),d['meta'][i],420000+i)
            with torch.no_grad():
                target=joint.legacy_exp.functionals(d['y'][i:i+1].cuda(),p,pred.n)[0]
                for k,v in points.items():residual[k].append((target-joint.legacy_exp.functionals(v,p,pred.n)[0]).cpu().numpy())
            pred.reveal(d['y'][i].cuda(),d['meta'][i]['start']+40)
    scale=pred.core.generator.lib['stats']['scale'].cpu().numpy();save(dict(residual={k:np.asarray(v) for k,v in residual.items()},scale=scale),ROOT/'task_calibration.pt');print('TASK_CALIBRATED',flush=True)

def run():
    ck=load(ROOT/'task_calibration.pt');scale=ck['scale'];rows=[];issued=[];chosen=['base','pod_rw','geo_shift','lin_shift','memory_gated','local_shift'];groups={}
    for p in sorted((ROOT/'parts').glob('*.pt')):
        a=load(p);groups.setdefault(a['meta']['file'],[]).append(a)
    for name,data in groups.items():
        data.sort(key=lambda a:a['index']);history={k:[] for k in chosen};controllers={k:{'acmcp_block_h1':AcMCPBlock(np.abs(ck['residual'][k]).max(0)/scale),'acmcp_lr005':AcMCPBlock(np.abs(ck['residual'][k]).max(0)/scale,lr=.05),'acmcp_lr020':AcMCPBlock(np.abs(ck['residual'][k]).max(0)/scale,lr=.2),'pid_no_scorecaster':AcMCPBlock(np.abs(ck['residual'][k]).max(0)/scale,scorecast=False)} for k in chosen}
        for a in data:
            intervals={};crps={}
            for mode,methods in a['functions'].items():
                for k,v in methods.items():intervals[(k,'field_'+mode)]=np.quantile(v.numpy(),[.05,.95],axis=0)
            for k in chosen:
                point=a['point'][k].numpy();intervals[(k,'task_fixed')]=point+np.quantile(ck['residual'][k],[.05,.95],axis=0)
                if a['scored']:
                    intervals[(k,'rolling10')]=point+np.quantile(history[k][-10:],[.05,.95],axis=0)
                    for lag in (1,3,5):intervals[(k,f'spci_qrf_lag{lag}')]=point+np.stack([qrf_interval(np.asarray(history[k])[:,j],lag) for j in range(5)],1)
                    for label,state in controllers[k].items():intervals[(k,label)]=state.issue(point/scale)*scale
            # All interval endpoints are computed before consuming the current Y.
            y=a['truth'].numpy()
            if a['scored']:
                for (k,label),bounds in intervals.items():
                    stats=interval_metrics(bounds,y,scale)
                    for j,task in enumerate(TASKS):rows.append(dict(**a['meta'],index=a['index'],method=k,interval=label,task=task,**{m:float(v[j]) for m,v in stats.items()}))
                    issued.append(dict(**a['meta'],method=k,interval=label,lower=bounds[0],upper=bounds[1],truth=y))
            for k in chosen:
                e=y-a['point'][k].numpy();history[k].append(e)
                for state in controllers[k].values():state.reveal(e/scale)
        print('INTERVALS',name,flush=True)
    df=pd.DataFrame(rows);df.to_csv(ROOT/'windows.csv',index=False);metrics=['coverage','width','interval_score','infinite','reversed'];tr=df.groupby(['method','interval','task','file','group'])[metrics].mean().reset_index();tr.to_csv(ROOT/'trajectories.csv',index=False);tr.groupby(['method','interval','task'])[metrics].mean().to_csv(ROOT/'task_summary.csv');tr.groupby(['method','interval'])[metrics].mean().to_csv(ROOT/'summary.csv');save(issued,ROOT/'issued_intervals.pt');dump(ROOT/'provenance.json',dict(protocol_sha256=digest(Path(__file__).with_name('TEMPORAL_INTERVAL_PROTOCOL.md')),source_sha256=digest(Path(__file__)),references={p.name:digest(p) for p in Path(__file__).with_name('reference_sources').glob('*')},claim='Disclosed fixed-predictor SPCI QRF adaptation and AcMCP block-h1 Python recurrence, not original-package/full-frame-h20 benchmark.'))

if __name__=='__main__':
    setup();ROOT.mkdir(exist_ok=True,parents=True);parser=argparse.ArgumentParser();parser.add_argument('action',choices=['collect','calibrate','run']);args=parser.parse_args();{'collect':collect,'calibrate':calibrate_tasks,'run':run}[args.action]()
