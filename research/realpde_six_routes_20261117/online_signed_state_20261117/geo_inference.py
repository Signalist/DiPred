"""Actual seven-coefficient identification on already revealed physical blocks."""
from geometric import *

class GeometricPredictor:
    def __init__(self,method=None,linear=False):
        setup();self.linear=linear;folder=OUT/'linear_geometry' if linear else GEO;self.predictor=st.model();self.n=norm();self.model=load(folder/'model.pt');self.selection=json.loads((folder/'selection.json').read_text());self.method=self.selection['chosen'] if method is None else method;self.state=GeometricState(self.selection['models'][self.method]['config'],self.model);self.pending=None
    @torch.no_grad()
    def predict(self,observed,start,normalized=False):
        if observed.shape!=(1,2,20,32,64) or not torch.isfinite(observed).all():raise ValueError('Expected complete finite history')
        x=observed.cuda();x=x if normalized else (x-self.n['mean'])/self.n['std'];p=self.predictor.rollout(x,500000,sign=0);c=self.state.issue(start);out=st.transform(p,torch.tensor(c,device='cuda',dtype=torch.float32)[None]);self.pending=p.detach().clone();return out if normalized else out*self.n['std']+self.n['mean']
    def reveal(self,observed_future,time,normalized=False):
        if self.pending is None or self.state.pending is None or time<self.state.pending+40:raise ValueError('Early or unissued feedback')
        y=observed_future.cuda();y=y if normalized else (y-self.n['mean'])/self.n['std']
        if y.shape!=self.pending.shape or not torch.isfinite(y).all():raise ValueError('Expected full finite20-frame reveal')
        with torch.enable_grad():c,_=(st.linear_oracle if self.linear else st.optimize_oracle)(self.pending,y)
        self.state.reveal(c[0].cpu().numpy().astype(float),time);self.pending=None;return c
    def skip_feedback(self):self.state.skip_feedback();self.pending=None

def audit():
    data=load(SOURCE/'development.pt');direct_path=GEO/'development_direct_issued.pt';direct_replay=direct_path.exists()
    if direct_replay:
        direct=load(direct_path);issued=direct['issued'];stored=dict(coefficients=direct['coefficients'],meta=data['meta'])
    else:stored=feedback_coefficients('development');issued=load(GEO/'development_issued.pt')
    ix=group_ids(data['meta'])[0];records=[]
    for method in ('rolling','rw'):
        pred=GeometricPredictor(method,linear=IDENTIFIER=='linear');times=[];reveal_times=[];largest=0.;coeferror=0.;replaymse=[]
        for i in ix:
            start=data['meta'][i]['start'];begin=time.perf_counter();q=pred.predict(data['x'][i:i+1],start,True);torch.cuda.synchronize();times.append(1000*(time.perf_counter()-begin))
            with torch.no_grad():
                reference=st.transform(pred.pending if direct_replay else data['p'][i:i+1].cuda(),torch.tensor(issued[method][i:i+1],dtype=torch.float32,device='cuda'));largest=max(largest,float((q-reference).abs().max()));replaymse.append(abs(float(st.mse(q,data['y'][i:i+1].cuda())[0])-float(st.mse(reference,data['y'][i:i+1].cuda())[0])))
            begin=time.perf_counter();coefficient=pred.reveal(data['y'][i:i+1],start+40,True);torch.cuda.synchronize();reveal_times.append(1000*(time.perf_counter()-begin));coeferror=max(coeferror,float((coefficient.cpu()-stored['coefficients'][i:i+1]).abs().max()))
        records.append(dict(method=method,field_replay_max_error=largest,mean_mse_replay_error=float(np.mean(replaymse)),max_mse_replay_error=max(replaymse),identification_coefficient_max_error=coeferror,median_predict_ms=float(np.median(times[1:])),median_reveal_identification_ms=float(np.median(reveal_times[1:])),n=len(ix)))
    model=load(GEO/'model.pt');sel=json.loads((GEO/'selection.json').read_text());checks={}
    for name,entry in sel['models'].items():
        before=issue(stored,entry['config'],model);changed=dict(stored,coefficients=stored['coefficients'].clone());changed['coefficients'][ix[3]]+=10;after=issue(changed,entry['config'],model)
        assert np.array_equal(before[ix[:4]],after[ix[:4]]);outside=[i for g in group_ids(stored['meta'])[1:] for i in g];assert np.array_equal(before[outside],after[outside])
        state=GeometricState(entry['config'],model);state.issue(0)
        try:state.reveal(np.ones(7),39);raise AssertionError('Early reveal accepted')
        except ValueError:pass
        assert len(state.history)==0 and state.last is None;state.skip_feedback();state.issue(20);state.skip_feedback();assert len(state.history)==0
        checks[name]=dict(prefix_exact=True,trajectory_isolation_exact=True,early_reveal_rejected=True,skip_does_not_create_feedback=True)
    passed=all(r['max_mse_replay_error']<1e-8 and (not direct_replay or r['field_replay_max_error']<3e-5) for r in records);path=GEO/('inference_audit.json' if direct_replay else 'batch_cache_replay_diagnostic.json')
    dump(path,dict(passed=passed,identifier=IDENTIFIER,direct_stream_replay=direct_replay,actual_past_target_identification=True,rows=records,causal=checks,numerics_note='Final efficacy uses actual single-window inference and identification. For nonlinear80-step identification, the legacy batch cache had small floating-point differences; that failed diagnostic is retained separately.'))
    pd.DataFrame(records).to_csv(GEO/'timing.csv',index=False);print(records,flush=True)
    if direct_replay:assert passed

def stream(split):
    setup();d=load(SOURCE/f'{split}.pt');model=load(GEO/'model.pt');selection=json.loads((GEO/'selection.json').read_text());net=st.model();issued={k:np.zeros((len(d['meta']),7)) for k in selection['models']};coefficients=torch.zeros(len(d['meta']),7);rows=[];events=[];timings=[]
    for ix in group_ids(d['meta']):
        states={k:GeometricState(v['config'],model) for k,v in selection['models'].items()}
        for count,i in enumerate(ix):
            start=d['meta'][i]['start']
            with torch.no_grad():
                p=net.rollout(d['x'][i:i+1].cuda(),500000,sign=0);predictions={}
                for k,state in states.items():
                    c=state.issue(start);issued[k][i]=c;predictions[k]=st.transform(p,torch.tensor(c,dtype=torch.float32,device='cuda')[None])
            events.append(dict(**d['meta'][i],event='issue',previous_full_reveals=count));y=d['y'][i:i+1].cuda()
            with torch.no_grad():
                for k,q in predictions.items():rows.append(dict(**d['meta'][i],scenario='natural',method='geo_'+k,mse=float(st.mse(q,y)[0])))
            begin=time.perf_counter();c,_=(st.linear_oracle if IDENTIFIER=='linear' else st.optimize_oracle)(p,y);torch.cuda.synchronize();timings.append(1000*(time.perf_counter()-begin));coefficients[i]=c[0].cpu()
            for state in states.values():state.reveal(c[0].cpu().numpy().astype(float),start+40)
            events.append(dict(**d['meta'][i],event='full20_reveal',previous_full_reveals=count+1))
        print('DIRECT_STREAM',split,d['meta'][ix[0]]['file'],flush=True)
    save(dict(issued=issued,coefficients=coefficients,meta=d['meta'],events=events),GEO/f'{split}_direct_issued.pt');pd.DataFrame(rows).to_csv(GEO/f'{split}_direct_windows.csv',index=False);dump(GEO/f'{split}_direct_manifest.json',dict(whole_trajectory_direct_inference=True,full20_registration_after_issue=True,median_identification_ms=float(np.median(timings)),selection_sha256=digest(GEO/'selection.json'),windows=len(d['meta'])))

def rollout():
    from common import records,read
    rows=[];n=norm()
    for method in ('rolling','rw'):
        pred=GeometricPredictor(method,linear=IDENTIFIER=='linear')
        for rec in records('development'):
            raw=torch.cat([read(rec,s)[:,:20] for s in range(0,281,20)],1).cuda()[None];raw=(raw-n['mean'])/n['std']
            for mode,warm in [('cold_free',0),('warm_free',3)]:
                pred.state=GeometricState(pred.selection['models'][method]['config'],pred.model);pred.pending=None
                for j in range(warm):pred.predict(raw[:,:,j*20:(j+1)*20],j*20,True);pred.reveal(raw[:,:,(j+1)*20:(j+2)*20],j*20+40,True)
                h=raw[:,:,warm*20:(warm+1)*20]
                for block in range(10):
                    start=(warm+block)*20;q=pred.predict(h,start,True);pred.skip_feedback();y=raw[:,:,start+20:start+40];rows.append(dict(file=rec['file'],group=rec['group'],method='geo_'+method,mode=mode,block=block+1,mse=float(st.mse(q,y)[0]),real_reveals=warm));h=q.detach()
        print('GEO_ROLLOUT',method,flush=True)
    pd.DataFrame(rows).to_csv(GEO/'rollout_windows.csv',index=False)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['audit','rollout','stream']);p.add_argument('--split',default='development');p.add_argument('--linear',action='store_true');a=p.parse_args()
    if a.linear:
        import geometric
        GEO=OUT/'linear_geometry';IDENTIFIER='linear';geometric.GEO=GEO;geometric.IDENTIFIER=IDENTIFIER
    {'audit':audit,'rollout':rollout,'stream':lambda:stream(a.split)}[a.action]()
