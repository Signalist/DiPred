"""Common probability calibration of frozen causal signed-state corrections."""
from state import *
import state as latent_state
sys.path.insert(0,str(W/'lowcost_joint_20261117'))
import study as joint
import local_learn as local
from geometric import GeometricState
sys.path.insert(0,str(W.parent/'Shared'))
from online_residual_subspace import ResidualSubspaceStream
FOLDER=OUT/'probability';FINAL=W/'runs/lowcost_joint_20261117/conditional_scores/scorefit'
NAMES=['base','pod_rw','rolling_bias','geo_shift','lin_shift','memory_gated','local_shift','geo_member_warp','lin_member_warp','local_member_warp']
LEVELS=(torch.arange(32)+.5)/32

def configure_scores():
    old=joint.legacy_exp.functionals
    if getattr(old,'physical_coordinates',False):return
    def functions(z,p,n):
        r=old(z,p,n);a=z*n['std']+n['mean'];v=(a[:,1,:,1:-1,2:]-a[:,1,:,1:-1,:-2])/(2*joint.DX)+(a[:,0,:,2:,1:-1]-a[:,0,:,:-2,1:-1])/(2*joint.DX);r[:,2]=v.square().mean((-1,-2)).amax(-1);return r
    functions.physical_coordinates=True;joint.legacy_exp.functionals=functions

def diagonal_quantile(a):return torch.stack([torch.quantile(a[k].flatten(1),float(LEVELS[k]),dim=1) for k in range(32)])

class Provider:
    def __init__(self):
        joint.OUT=FINAL;self.core=joint.Predictor(method='ar__mixed');self.n=self.core.n;self.basis=load(OUT/'basis.pt');self.b=self.basis['basis'].cuda();self.models=load(OUT/'models.pt');self.selected=json.loads((OUT/'selection.json').read_text());self.geo={};self.states={};self.pending=None
        for name,folder in [('geo','geometric'),('lin','linear_geometry')]:
            root=OUT/folder;sel=json.loads((root/'selection.json').read_text());self.geo[name]=(load(root/'model.pt'),sel['models'][sel['chosen']]['config'])
        self.memory_shrink=json.loads((OUT/'control_selection.json').read_text())['memory_gated']['shrink'];ck=load(SOURCE/'local/warp_F_last.pt','cuda');self.local_cfg=ck['summary'];self.local_proj=ck['projection'];self.head=local.Head('warp').cuda().eval().requires_grad_(False);self.head.load_state_dict(ck['model'])
    def new_state(self):
        return dict(pod=State(self.selected['models']['rw_kalman']['config'],self.models),bias=State(self.selected['models']['rolling_bias']['config'],self.models),geo=GeometricState(self.geo['geo'][1],self.geo['geo'][0]),lin=GeometricState(self.geo['lin'][1],self.geo['lin'][0]),memory=ResidualSubspaceStream(gated=True),previous_truth=None,revealed=0,last_reveal=-1)
    @torch.no_grad()
    def issue(self,x,meta,seed):
        if self.pending is not None:raise RuntimeError('Reveal previous block first')
        name=meta['file'];state=self.states.setdefault(name,self.new_state());start=meta['start'];assert start+20>=state['last_reveal'];p,values,desc=self.core.issue(x);p=p[None];z=joint.couple(values,self.core.generator.sample('ar__mixed',desc,seed));points={'base':p};raw=torch.cat([desc[k] for k in ('X','P','F_last')],1).cpu().double();f=project(raw,self.basis['projection'])[0].numpy()
        for model,label in [('pod','pod_rw'),('bias','rolling_bias')]:
            c=state[model].issue(f,start);delta=(torch.tensor(c,device='cuda')@self.b).reshape_as(p).float()/self.n['std'];points[label]=p+delta
        coeff={}
        for model,label in [('geo','geo_shift'),('lin','lin_shift')]:
            coeff[model]=torch.tensor(state[model].issue(start),dtype=torch.float32,device='cuda')[None];points[label]=st.transform(p,coeff[model])
        physical_p=(p*self.n['std']+self.n['mean']).cpu();history=(x*self.n['std']+self.n['mean']).cpu() if state['previous_truth'] is None else state['previous_truth'];m,_=state['memory'].issue(history,physical_p);points['memory_gated']=p+(m-physical_p).cuda()*self.memory_shrink/self.n['std']
        lf=local.apply_projection(desc['F_last'].reshape(1,-1,4,8),self.local_proj);lc=self.head.coefficients(lf)*self.local_cfg['shrink']*p.new_tensor(st.LIMITS)[None,:,None,None];points['local_shift']=local.local_transform(p,lc)
        fields={k:z+(q-p) for k,q in points.items()};fields['geo_member_warp']=st.transform(z,coeff['geo'].expand(32,-1));fields['lin_member_warp']=st.transform(z,coeff['lin'].expand(32,-1));fields['local_member_warp']=local.local_transform(z,lc.expand(32,-1,-1,-1));points.update(geo_member_warp=points['geo_shift'],lin_member_warp=points['lin_shift'],local_member_warp=points['local_shift'])
        assert list(fields)==NAMES and all(torch.isfinite(v).all() for v in fields.values());self.pending=dict(meta=meta,state=state,p=p.detach().clone(),issued_reveals=state['revealed']);return p[0],fields,points
    def reveal(self,y,reveal_time):
        if self.pending is None or reveal_time<self.pending['meta']['start']+40:raise ValueError('Early/unissued full20 truth')
        state=self.pending['state'];p=self.pending['p'];y=y[None] if y.ndim==4 else y;r=((y-p)*self.n['std']).flatten().double();c=(r@self.b.T/D).cpu().numpy()
        state['pod'].reveal(c,reveal_time);state['bias'].reveal(c,reveal_time)
        with torch.enable_grad():nonlinear,_=st.optimize_oracle(p,y)
        linear,_=st.linear_oracle(p,y);state['geo'].reveal(nonlinear[0].cpu().numpy().astype(float),reveal_time);state['lin'].reveal(linear[0].cpu().numpy().astype(float),reveal_time);state['previous_truth']=(y*self.n['std']+self.n['mean']).detach().cpu();state['revealed']+=1;state['last_reveal']=reveal_time;self.pending=None

def fit():
    pred=Provider();d=load(SOURCE/'calibration.pt');errors={k:[] for k in NAMES};events=[]
    for ix in group_ids(d['meta']):
        for i in ix:
            p,fields,points=pred.issue(d['x'][i:i+1].cuda(),d['meta'][i],420000+i);events.append(dict(**d['meta'][i],event='issue'))
            y=d['y'][i].cuda()
            for k,z in fields.items():errors[k].append((y-z.sort(dim=0,stable=True).values)[...,::4,::4].cpu())
            pred.reveal(y,d['meta'][i]['start']+40);events.append(dict(**d['meta'][i],event='full20_reveal'))
        print('FIT_CALIBRATION',d['meta'][ix[0]]['file'],flush=True)
    offsets={}
    for k,parts in errors.items():offsets[k]=diagonal_quantile(torch.stack(parts,2).cuda()).cpu()
    save(dict(offsets=offsets,levels=LEVELS,events=events),FOLDER/'offsets.pt');dump(FOLDER/'fit.json',dict(methods=NAMES,calibration_windows=90,trajectories=10,procedure='32node x2component quantile offsets, spatial stride4, separate fit per changed distribution; identical procedure',protocol_sha256=digest(W/'online_signed_state_20261117/PROBABILITY_PROTOCOL.md'),frozen_models={str(p.relative_to(W)):digest(p) for p in [OUT/'models.pt',OUT/'selection.json',OUT/'geometric/model.pt',OUT/'geometric/selection.json',OUT/'linear_geometry/model.pt',OUT/'linear_geometry/selection.json',SOURCE/'local/warp_F_last.pt']}))

@torch.no_grad()
def metrics(z,y,p,point,pred,stats):
    score,func=joint.scores(z,y,p,pred.n,stats);q=torch.quantile(z,z.new_tensor([.05,.95]),dim=0);coverage=((y>=q[0])&(y<=q[1])).float();score.update(point_mse=float(st.mse(point,y[None])[0]),sample_mean_mse=float(st.mse(z.mean(0,keepdim=True),y[None])[0]),coverage90=float(coverage.mean()),coverage90_u=float(coverage[0].mean()),coverage90_v=float(coverage[1].mean()),width90=float(((q[1]-q[0])*pred.n['std'][0]).mean()))
    return score,func

def run(split):
    configure_scores();pred=Provider();d=load(SOURCE/f'{split}.pt');initial=load(FOLDER/'offsets.pt','cuda')['offsets'];states={};folder=FOLDER/split;folder.mkdir(parents=True,exist_ok=True);(folder/'parts').mkdir(exist_ok=True);rows=[];events=[];stats=pred.core.generator.lib['stats'];count=0
    for ix in group_ids(d['meta']):
        state={k:v.clone() for k,v in initial.items()}
        for i in ix:
            meta=d['meta'][i];p,raw,points=pred.issue(d['x'][i:i+1].cuda(),meta,420000+i);fields={'raw':raw,'fixed':{},'online':{}};nodes={}
            with torch.no_grad():
                for k,z in raw.items():
                    q,order=z.sort(dim=0,stable=True);nodes[k]=q
                    for mode,offset in [('fixed',initial[k]),('online',state[k])]:
                        calibrated=(q+offset[:,:,None,None,None]).sort(dim=0,stable=True).values;fields[mode][k]=torch.empty_like(z).scatter_(0,order,calibrated)
            events.append(dict(**meta,event='issue_all_methods_and_modes',previous_reveals=pred.pending['issued_reveals']));y=d['y'][i].cuda();sc={};func={}
            for mode,methods in fields.items():
                sc[mode]={};func[mode]={}
                for k,z in methods.items():
                    values,f=metrics(z,y,p,points[k],pred,stats);sc[mode][k]=values;func[mode][k]=f;rows.append(dict(**meta,method=k,mode=mode,**values))
            pointfunc=joint.legacy_exp.functionals(p[None],p,pred.n)[0].cpu();save(dict(meta=meta,scores=sc,functionals=func,point_functionals=pointfunc),folder/'parts'/f'{i:04d}.pt')
            if count in (0,4):
                save(dict(samples={mode:{k:fields[mode][k].cpu() for k in ('base','lin_shift','lin_member_warp')} for mode in fields},y=y.cpu(),p=p.cpu(),norm={k:v.cpu() for k,v in pred.n.items()},stats={k:v.cpu() for k,v in stats.items()},meta=meta),folder/f'audit_{i:04d}.pt')
            with torch.no_grad():
                for k,q in nodes.items():state[k]=.5*state[k]+.5*diagonal_quantile((y-q)[...,::4,::4])
            pred.reveal(y,meta['start']+40);events.append(dict(**meta,event='full20_reveal'));count+=1
            if count%20==0:print('PROBABILITY',split,count,len(d['meta']),flush=True)
    df=pd.DataFrame(rows);df.to_csv(folder/'windows.csv',index=False);cols=[k for k in df if k not in ('file','group','start','method','mode')];tr=df.groupby(['method','mode','file','group'])[cols].mean().reset_index();tr.to_csv(folder/'trajectories.csv',index=False);tr.groupby(['method','mode'])[cols].mean().to_csv(folder/'summary.csv');(folder/'events.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in events));print('DONE',split,flush=True)

if __name__=='__main__':
    setup();FOLDER.mkdir(parents=True,exist_ok=True);p=argparse.ArgumentParser();p.add_argument('action',choices=['fit','run']);p.add_argument('--split',default='development');a=p.parse_args();fit() if a.action=='fit' else run(a.split)
