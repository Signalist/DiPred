"""Independent conditioning/MSE algebra and actual sparse-feedback replay."""
from active_stream import *

def numpy_checks():
    model=load(OUT/'model.pt');n={k:v.numpy().astype(float) for k,v in norm('cpu').items()};rows=[]
    for split in ('calibration','development','late','future'):
        for path in sorted((OUT/split).glob('audit_*.pt')):
            s=load(path);cfg=s['config'];b=model['basis'][:cfg['rank']].numpy();scale=s['scale'].numpy().astype(float);full=b*scale;at=full[:,:,5:].reshape(cfg['rank'],-1);h=full[:,:,4,SITES[:,0],SITES[:,1]].transpose(2,1,0);remainder=np.maximum(model['second'].numpy()-(b*b).sum(0),.01)*cfg['remainder'];noise=(remainder*scale**2)[:,4,SITES[:,0],SITES[:,1]].T;indices=s['indices'];hs=h[indices].reshape(-1,cfg['rank']);rs=noise[indices].reshape(-1);p=s['p'].numpy().astype(float);physical=p*n['std'][0]+n['mean'][0];obs=s['observed_physical'].numpy().astype(float);error=(obs-physical[:,4,SITES[indices,0],SITES[indices,1]].T).reshape(-1)
            # Observation-space Schur solve, independent of latent precision code.
            z=cfg['gain']*hs.T@np.linalg.solve(hs@hs.T+np.diag(rs),error);reconstructed=physical[:,5:]+(z@at).reshape(2,15,32,64);truth=s['truth'].numpy().astype(float)*n['std'][0]+n['mean'][0];mse=((reconstructed-truth[:,5:])**2).mean();ze=float(np.max(abs(z-s['z'].numpy())));me=abs(mse-s['analytic_mse']);assert ze<1e-5 and me<1e-9,(path,ze,me)
            g=at@at.T/at.shape[1];first=[]
            for j in range(128):first.append(float(np.trace(g@h[j].T@np.linalg.solve(h[j]@h[j].T+np.diag(noise[j]),h[j]))))
            if path.stem.endswith('voi'):assert int(np.argmax(first))==int(indices[0])
            rows.append(dict(split=split,snapshot=path.name,latent_error=ze,mse_error=me,first_voi_choice_matches=path.stem.endswith('voi')))
    df=pd.DataFrame(rows);df.to_csv(OUT/'numerical_audit.csv',index=False);assert len(df)==32;return dict(snapshots=len(df),latent_max_error=float(df.latent_error.max()),mse_max_error=float(df.mse_error.max()))

def request_checks():
    result={}
    for split in ('calibration','development','late','future'):
        req=load(OUT/split/'requests.pt');events=[json.loads(v) for v in (OUT/split/'events.jsonl').read_text().splitlines()];df=pd.read_csv(OUT/split/'windows.csv')
        for r in req:
            for label,indices in r['actions'].items():
                budget=int(label.rsplit('_',1)[1]);assert len(indices)==budget and len(set(indices))==budget and all(0<=v<128 for v in indices)
        for k,r in enumerate(req):
            ev=events[4*k:4*k+4];assert [e['event'] for e in ev]==['issue_forecast_and_all_requests','sparse_values_arrive','reissue_remaining15','future15_score_only'];assert [e['time']-r['meta']['start'] for e in ev]==[20,25,25,40]
        assert np.array_equal(df.scalar_measurements.values,2*df.budget.values);zero=df[df.budget==0];assert np.max(abs(zero.mse-zero.base_mse))==0;assert len(events)==4*len(req);result[split]=dict(windows=len(req),budget_masks_valid=True,chronology_valid=True,zero_observation_exact=True)
    train=set(m['file'] for m in load(OUT/'model.pt')['meta']);cal=set(m['file'] for m in load(OUT/'calibration.pt')['meta']);dev=set(m['file'] for m in load(OUT/'development.pt')['meta']);assert train.isdisjoint(cal) and train.isdisjoint(dev) and cal.isdisjoint(dev);return result

def direct():
    d=load(SOURCE/'development.pt');cache=load(OUT/'development.pt');table=pd.read_csv(OUT/'development/windows.csv');results=[];n=norm();indices=[0,4]
    for method in ('uniform','voi','variance','risk'):
        pred=SparseUpdatePredictor(method,8)
        for i in indices:
            meta=d['meta'][i];request=pred.predict(d['x'][i:i+1],meta['start'],meta['file'],True);idx=request['site_indices'];s=load(OUT/'development'/f'audit_{i:04d}_{method}.pt');assert np.array_equal(idx,s['indices']);pe=float((pred.pending['p'].cpu()-cache['p'][i]).abs().max());assert pe==0
            y=d['y'][i].cuda()*n['std'][0]+n['mean'][0];values=observation_rows(y)[idx]
            try:pred.update(values,meta['start']+24);raise AssertionError('Early values accepted')
            except ValueError:pass
            try:pred.update(y,meta['start']+25);raise AssertionError('Full target accepted')
            except ValueError:pass
            q=pred.update(values.cpu(),meta['start']+25);mse=float((q-y[:,5:]).square().mean());me=abs(mse-s['analytic_mse']);assert me<1e-9
            request2=pred.predict(d['x'][i:i+1],meta['start'],meta['file'],True);changed=y.clone();mask=torch.ones(128,dtype=torch.bool,device='cuda');mask[idx]=False;changed[:,4,SITES[mask.cpu().numpy(),0],SITES[mask.cpu().numpy(),1]]+=10;changed[:,5:]+=10;values2=observation_rows(changed)[idx];assert torch.equal(values,values2);q2=pred.update(values2,meta['start']+25);assert torch.equal(q,q2)
            pred.predict(d['x'][i:i+1],meta['start'],meta['file'],True);q3=pred.update(values+.001,meta['start']+25);assert not torch.equal(q,q3)
            results.append(dict(method=method,index=i,point_replay_exact=True,mask_match=True,mse_error=me,early_reveal_rejected=True,full_target_rejected=True,unobserved_truth_isolation=True,selected_sensor_changes_prediction=True))
        del pred;torch.cuda.empty_cache()
    pred=SparseUpdatePredictor('uniform',0);i=0;req=pred.predict((d['x'][i:i+1].cuda()*n['std']+n['mean']).cpu(),0,'physical',False);q=pred.update(np.empty((0,2)),25);assert torch.equal(q,req['prediction'][:,5:]);results.append(dict(cpu_physical_interface=True,zero_budget_exact=True));return results

if __name__=='__main__':
    setup();a=numpy_checks();print('NUMERICAL',a,flush=True);b=request_checks();print('REQUESTS',b,flush=True);c=direct();dump(OUT/'audit.json',dict(passed=True,numerical=a,requests=b,direct=c,protocol_hash_matches=json.loads((OUT/'selection.json').read_text())['protocol_sha256']==digest(Path(__file__).with_name('PROTOCOL.md')),source_sha256=digest(Path(__file__))));print('AUDIT_PASSED',flush=True)
