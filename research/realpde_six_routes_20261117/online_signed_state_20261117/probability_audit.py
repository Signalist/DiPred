"""Independent scores, prior-result replay, and issued-prefix interventions."""
from probability import *
import hashlib
import importlib.util
spec=importlib.util.spec_from_file_location('joint_audit',W/'lowcost_joint_20261117/audit.py');audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)

def numerical():
    rows=[]
    for split in ('calibration','development','late','future'):
        for path in sorted((FOLDER/split).glob('audit_*.pt')):
            a=load(path);saved=load(path.parent/'parts'/path.name.replace('audit_',''))['scores'];n={k:v.numpy().astype(float) for k,v in a['norm'].items()};y=a['y'].numpy().astype(float);p=a['p'].numpy().astype(float);scale=a['stats']['scale'].numpy().astype(float)
            for mode,methods in a['samples'].items():
                for method,value in methods.items():
                    z=value.numpy().astype(float);flat=z.reshape(32,-1);fy=audit.funcs(y[None],p,n)[0];f=audit.funcs(z,p,n)
                    es=np.sqrt(((flat-y.reshape(-1))**2).mean(1)).mean()-sum(np.sqrt(((flat[i]-flat)**2).mean(1)).sum() for i in range(32))/(2*32**2)
                    vs=np.mean([((np.sqrt(np.abs(np.diff(z,axis=dim+1))).mean(0)-np.sqrt(np.abs(np.diff(y,axis=dim))))**2).mean() for dim in (3,2,1,0)])
                    q=np.quantile(z,[.05,.95],axis=0);cov=((y>=q[0])&(y<=q[1])).mean();width=((q[1]-q[0])*n['std'][0]).mean();mean_mse=(((z.mean(0)-y)*n['std'][0])**2).mean()
                    vals=dict(es=es,vs=vs,functional_crps=(audit.crps(f,fy)/scale).mean(),coverage90=cov,width90=width,sample_mean_mse=mean_mse)
                    for metric,v in vals.items():rows.append(dict(split=split,snapshot=path.stem,mode=mode,method=method,metric=metric,error=abs(v-saved[mode][method][metric])))
    df=pd.DataFrame(rows);df.to_csv(FOLDER/'numerical_audit.csv',index=False);assert df.error.max()<3e-5;return float(df.error.max())

def replay():
    rows=[];causal={}
    for split in ('calibration','development','late','future'):
        d=pd.read_csv(FOLDER/split/'windows.csv');point=d[d['mode']=='raw'].set_index(['method','file','start']);cols=['file','start']
        for method,filename,old in [('base',f'{split}_windows.csv','base'),('pod_rw',f'{split}_windows.csv','rw_kalman'),('rolling_bias',f'{split}_windows.csv','rolling_bias'),('memory_gated',f'{split}_controls.csv','memory_gated'),('local_shift',f'{split}_controls.csv','local_warp'),('geo_shift',f'geometric/{split}_direct_windows.csv','geo_rolling'),('lin_shift',f'linear_geometry/{split}_direct_windows.csv','geo_rolling')]:
            ref=pd.read_csv(OUT/filename);ref=ref[(ref.scenario=='natural')&(ref.method==old)].set_index(cols);now=point.loc[method].reindex(ref.index)
            assert len(ref)>0 and now.point_mse.notna().all();err=float(np.max(abs(now.point_mse-ref.mse)));assert err<1e-8,(split,method,err);rows.append(dict(split=split,check='point',method=method,error=err))
        if split!='calibration':
            ref=pd.read_csv(W/f'runs/lowcost_joint_20261117/common_calibration/{split}/windows.csv');ref=ref[ref.method=='lowcost'].set_index(['mode']+cols);now=d[d.method=='base'].set_index(['mode']+cols).reindex(ref.index)
            metrics=[c for c in ref if c not in ('group','method')];err=float(np.max(abs(now[metrics].to_numpy()-ref[metrics].to_numpy())));assert err<3e-5;rows.append(dict(split=split,check='prior_joint_baseline',method='base',error=err))
        events=[json.loads(v) for v in (FOLDER/split/'events.jsonl').read_text().splitlines()];seen={}
        for issue,reveal in zip(events[::2],events[1::2]):
            assert issue['event']=='issue_all_methods_and_modes' and reveal['event']=='full20_reveal';assert all(issue[k]==reveal[k] for k in ('file','start','group'));name=issue['file'];assert issue['previous_reveals']==seen.get(name,0);seen[name]=seen.get(name,0)+1
        assert len(events)==2*len(d)//30;causal[split]=dict(windows=len(events)//2,trajectories=len(seen),issue_before_reveal=True)
    df=pd.DataFrame(rows);df.to_csv(FOLDER/'replay_audit.csv',index=False);offset=float((load(FOLDER/'offsets.pt')['offsets']['base']-load(W/'runs/lowcost_joint_20261117/common_calibration/offsets.pt')['offsets']).abs().max());assert offset==0;return dict(maximum_error=float(df.error.max()),baseline_offset_exact=True,events=causal)

def interventions():
    d=load(SOURCE/'development.pt');ids=group_ids(d['meta']);initial=load(FOLDER/'offsets.pt','cuda')['offsets'];results=[]
    def hashes(fields):return {k:hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest() for k,v in fields.items()}
    for perturb in (False,True):
        pred=Provider();state={k:v.clone() for k,v in initial.items()};out=[]
        for j,i in enumerate(ids[0][:6]):
            meta=d['meta'][i];_,fields,_=pred.issue(d['x'][i:i+1].cuda(),meta,420000+i);rawhash=hashes(fields);online={};nodes={}
            for k,z in fields.items():
                q,order=z.sort(dim=0,stable=True);nodes[k]=q;online[k]=torch.empty_like(z).scatter_(0,order,(q+state[k][:,:,None,None,None]).sort(dim=0,stable=True).values)
            out.append((rawhash,hashes(online)))
            if j==0:
                try:pred.reveal(d['y'][i].cuda(),meta['start']+39);raise AssertionError('Early truth accepted')
                except ValueError:pass
                assert pred.pending['state']['revealed']==0
            y=d['y'][i].cuda()+(.2 if perturb and j==3 else 0.)
            for k,q in nodes.items():state[k]=.5*state[k]+.5*diagonal_quantile((y-q)[...,::4,::4])
            pred.reveal(y,meta['start']+40)
        i=ids[1][0];_,fields,_=pred.issue(d['x'][i:i+1].cuda(),d['meta'][i],420000+i);out.append(hashes(fields));results.append(out);del pred;torch.cuda.empty_cache()
    assert results[0][:4]==results[1][:4] and results[0][-1]==results[1][-1];assert results[0][4]!=results[1][4]
    return dict(future_target_intervention_preserves_all_issued_prefixes=True,online_calibration_prefix_invariant=True,changed_feedback_affects_next_forecast=True,trajectory_isolation=True,early_reveal_rejected=True)

if __name__=='__main__':
    setup();r=replay();print('REPLAY',r,flush=True);n=numerical();print('NUMERICAL',n,flush=True);c=interventions();dump(FOLDER/'audit.json',dict(passed=True,replay=r,numerical_max_error=n,causal=c,source_sha256=digest(Path(__file__)),probability_source_sha256=digest(Path(__file__).with_name('probability.py'))));print('AUDIT_PASSED',flush=True)
