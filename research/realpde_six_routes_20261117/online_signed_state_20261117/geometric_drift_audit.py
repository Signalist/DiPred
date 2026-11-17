"""Replay the actual first faulted trajectory, including signed feedback."""
from geometric_drift import *

def main():
    setup();rec=records('development')[0];net=st.model();n=norm();saved=load(ROOT/'coefficients.pt');lookup={(r['start'],r['scenario'],r['method']):r['issued'] for r in saved if r['file']==rec['file']};table=pd.read_csv(ROOT/'windows.csv');table=table[table.file==rec['file']].set_index(['start','scenario','method']);states={};models={};cfgs={};trace={};maxcoef=0.;maxmse=0.
    for kind,folder in [('nonlinear','geometric'),('linear','linear_geometry')]:
        models[kind]=load(OUT/folder/'model.pt');cfgs[kind]={k:v['config'] for k,v in json.loads((OUT/folder/'selection.json').read_text())['models'].items() if k!='zero'}
    for scenario in ('step','ramp'):
        states[scenario]={kind:{k:GeometricState(cfg,models[kind]) for k,cfg in family.items()} for kind,family in cfgs.items()};trace[scenario]={kind:[] for kind in cfgs}
    for j,start in enumerate(range(0,481,20)):
        raw=(read(rec,start)[None].cuda()-n['mean'])/n['std']
        with torch.no_grad():clean=net.rollout(raw[:,:,:20],500000,sign=0)
        for scenario in states:
            amount=0 if j<12 else (1 if scenario=='step' else min(1.,(j-11)/4));c=VECTOR*amount
            with torch.no_grad():
                p=clean if amount==0 else st.transform(clean,torch.tensor(c,dtype=torch.float32,device='cuda')[None]);issued={}
                for kind,family in states[scenario].items():
                    for name,state in family.items():
                        cc=state.issue(start);label=kind+'_'+name;maxcoef=max(maxcoef,float(abs(cc-lookup[(start,scenario,label)]).max()));issued[label]=st.transform(p,torch.tensor(cc,dtype=torch.float32,device='cuda')[None])
                for name,q in issued.items():maxmse=max(maxmse,abs(float(st.mse(q,raw[:,:,20:])[0])-float(table.loc[(start,scenario,name),'mse'])))
            for kind,identifier in [('nonlinear',st.optimize_oracle),('linear',st.linear_oracle)]:
                observed,_=identifier(p,raw[:,:,20:]);value=observed[0].cpu().numpy().astype(float);trace[scenario][kind].append(value)
                for state in states[scenario][kind].values():state.reveal(value,start+40)
    assert maxcoef==0 and maxmse<1e-12,(maxcoef,maxmse)
    for scenario,kinds in trace.items():
        for kind,values in kinds.items():
            for name,cfg in cfgs[kind].items():
                states2=[GeometricState(cfg,models[kind]) for _ in range(2)]
                for j,value in enumerate(values):
                    before=states2[0].issue(j*20);after=states2[1].issue(j*20)
                    if j<=15:assert np.array_equal(before,after)
                    states2[0].reveal(value,j*20+40);states2[1].reveal(value+(1 if j==15 else 0),j*20+40)
    events=[json.loads(x) for x in (ROOT/'events.jsonl').read_text().splitlines()];seen={}
    for a,b in zip(events[::2],events[1::2]):
        assert a['event']=='issue_all' and b['event']=='full20_reveal';assert a['file']==b['file'] and a['start']==b['start'];assert a['previous_reveals']==seen.get(a['file'],0);seen[a['file']]=a['previous_reveals']+1
    assert len(events)==600 and len(seen)==12 and set(seen.values())=={25}
    save(trace,ROOT/'audit_feedback_trace.pt');dump(ROOT/'audit.json',dict(passed=True,actual_single_trajectory_replay_windows=25,scenarios=2,identifiers=2,issued_coefficient_max_error=maxcoef,mse_replay_max_error=maxmse,future_feedback_preserves_issued_prefix=True,all_issue_before_full20_events=True,windows=300,source_sha256=digest(Path(__file__))));print('GEOMETRIC_DRIFT_AUDIT_PASSED',maxcoef,maxmse,flush=True)

if __name__=='__main__':main()
