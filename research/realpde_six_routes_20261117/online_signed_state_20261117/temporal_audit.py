"""Independent indexed h1 recurrence and saved-interval causal replay."""
from temporal_intervals import *
import math

def indexed_reference(errors,ki,lr=.1,scorecast=True):
    # Explicit 1-based row indices as in the R source, one tail/task at a time.
    count,dim=errors.shape;result=np.zeros((count+1,2,dim));csat=2/math.pi*(math.ceil(math.log(1000)*.01)-1/math.log(1000))
    for k in range(dim):
        for side,sign in enumerate((-1,1)):
            tracker=np.zeros(count+2);q=np.zeros(count+2);misses=[]
            for t in range(1,count+1):
                miss=float(sign*errors[t-1,k]>q[t]);misses.append(miss);recent=errors[max(0,t-10):t,k];rate=lr if t==1 else lr*(max(recent)-min(recent));tracker[t+1]=tracker[t]+rate*(miss-.05)
                integral=0.
                if t>1 and ki[k]!=0:
                    arg=(sum(misses)-t*.05)*math.log(t)/(csat*t);integral=ki[k]*(math.inf if arg>=math.pi/2 else -math.inf if arg<=-math.pi/2 else math.tan(arg))
                score=sign*sum(recent)/len(recent) if scorecast and t>=10 else 0.;q[t+1]=tracker[t+1]+integral+score;result[t,side,k]=q[t+1]
    return result

def main():
    rng=np.random.default_rng(42);errors=rng.normal(size=(65,5))*.2;errors[20:40]+=np.arange(5)[None]*.3;ki=np.array([.5,1.,2.,3.,5.]);largest=0.
    for lr,scorecast in ((.05,True),(.1,True),(.2,True),(.1,False)):
        expected=indexed_reference(errors,ki,lr,scorecast);state=AcMCPBlock(ki,lr,scorecast);actual=[state.qs.copy()]
        for e in errors:state.reveal(e);actual.append(state.qs.copy())
        actual=np.asarray(actual);assert np.array_equal(np.isinf(actual),np.isinf(expected));mask=np.isfinite(expected);err=float(np.max(abs(actual[mask]-expected[mask])));largest=max(largest,err);assert err<1e-12
    ck=load(ROOT/'task_calibration.pt');scale=ck['scale'];data=[load(p) for p in sorted((ROOT/'parts').glob('00_*.pt'))];saved=load(ROOT/'issued_intervals.pt');lookup={(a['file'],a['start'],a['method'],a['interval']):np.stack([a['lower'],a['upper']]) for a in saved};replay=0.;prefix=[]
    for perturb in (False,True):
        history=[];ki=np.abs(ck['residual']['base']).max(0)/scale;state=AcMCPBlock(ki);issued=[]
        for i,a in enumerate(data[:17]):
            point=a['point']['base'].numpy();bounds={}
            if i>=10:
                bounds['acmcp_block_h1']=state.issue(point/scale)*scale;bounds['rolling10']=point+np.quantile(history[-10:],[.05,.95],axis=0);bounds['spci_qrf_lag3']=point+np.stack([qrf_interval(np.asarray(history)[:,j],3) for j in range(5)],1)
                issued.append(bounds)
                if not perturb:
                    for label,b in bounds.items():
                        expected=lookup[(a['meta']['file'],a['meta']['start'],'base',label)];assert np.array_equal(np.isinf(b),np.isinf(expected));mask=np.isfinite(b);err=float(np.max(abs(b[mask]-expected[mask])));replay=max(replay,err);assert err<1e-10
            error=a['truth'].numpy()-point+(scale*.5 if perturb and i==13 else 0);history.append(error);state.reveal(error/scale)
        prefix.append(issued)
    for i in range(4):
        for label in prefix[0][i]:assert np.array_equal(prefix[0][i][label],prefix[1][i][label])
    assert any(not np.array_equal(prefix[0][4][k],prefix[1][4][k]) for k in prefix[0][4])
    events=[json.loads(v) for v in (ROOT/'events.jsonl').read_text().splitlines()];seen={}
    for a,b in zip(events[::2],events[1::2]):
        assert a['event']=='issue_all' and b['event']=='full20_reveal';assert a['start']==b['start'] and a['file']==b['file'];name=a['file'];assert a['previous_reveals']==seen.get(name,0);seen[name]=a['previous_reveals']+1;assert b['previous_reveals']==seen[name]
    assert len(seen)==12 and set(seen.values())=={41} and len(events)==984
    result=dict(passed=True,independently_indexed_acmcp_max_error=largest,saved_temporal_interval_replay_max_error=replay,future_residual_intervention_preserves_prefix=True,next_prediction_changes_after_intervention=True,common_full20_events_verified=True,trajectories=12,windows_per_trajectory=41,common_warmup=10,scored_per_trajectory=31,limitations='No R runtime was available; reference audit is an independently indexed Python translation of the downloaded R recurrence. QRF is disclosed leaf-weighted adaptation, not author-package identity.',source_sha256=digest(Path(__file__)));dump(ROOT/'audit.json',result);print(result,flush=True)

if __name__=='__main__':main()
