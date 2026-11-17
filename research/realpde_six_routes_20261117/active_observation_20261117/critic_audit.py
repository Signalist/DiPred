"""Public stream replay of the learned observation policy and training labels."""
from critic import *

@torch.no_grad()
def main():
    setup();selected=json.loads((CRITIC/'selection.json').read_text());assert selected['common_assimilation_selection_sha256']==digest(ROOT/'selection.json');weights=load(CRITIC/'model.pt');assert weights['weights'][str(selected['alpha'])].numel()==2432;train=set(m['file'] for m in weights['meta']);rows=[];n=norm()
    for split in ('calibration','development','late','future'):
        requests=load(CRITIC/split/'requests.pt');assert train.isdisjoint(r['meta']['file'] for r in requests);events=[json.loads(e) for e in (CRITIC/split/'events.jsonl').read_text().splitlines()];frame=pd.read_csv(CRITIC/split/'windows.csv');assert (frame.scalar_measurements==2*frame.budget).all();zero=frame[frame.budget==0];assert (zero.mse==zero.base_mse).all()
        for i,request in enumerate(requests):
            assert len(request['order'])==32 and len(set(request['order']))==32;assert [e['time']-request['meta']['start'] for e in events[3*i:3*i+3]]==[20,25,40]
        d=load(SOURCE/f'{split}.pt');predictor=make_critic_predictor(8)
        for i in (0,4):
            meta=d['meta'][i];issued=predictor.predict(d['x'][i:i+1],meta['start'],meta['file'],True);idx=issued['site_indices'];assert np.array_equal(idx,requests[i]['order'][:8]);y=d['y'][i].cuda()*n['std'][0]+n['mean'][0];observed=observation_rows(y)[idx];q=predictor.update(observed.cpu(),meta['start']+25);loss=float((q-y[:,5:]).square().mean());saved=load(CRITIC/split/f'audit_{i:04d}_critic.pt');error=abs(loss-saved['analytic_mse']);assert error<1e-9
            predictor.predict(d['x'][i:i+1],meta['start'],meta['file'],True);other=y.clone();mask=torch.ones(128,dtype=torch.bool,device='cuda');mask[idx]=False;other[:,4,SITES[mask.cpu().numpy(),0],SITES[mask.cpu().numpy(),1]]+=3;other[:,5:]-=4;q2=predictor.update(observation_rows(other)[idx],meta['start']+25);assert torch.equal(q,q2);rows.append(dict(split=split,index=i,request_exact=True,mse_replay_error=error,unselected_and_future_targets_isolated=True))
        del predictor;torch.cuda.empty_cache()
    # Labels have their own field reconstruction check, not just a regression fit.
    labels=load(CRITIC/'training_labels.pt');source=load(SOURCE/'train.pt');cache=load(BASE/'train.pt');assim=PredictiveAssimilator(json.loads((ROOT/'selection.json').read_text())['config']);maximum=0.
    for i in (0,400,818):
        p=cache['p'][i].cuda();scale=cache['scale'][i].cuda();y=source['y'][i].cuda()*n['std'][0]+n['mean'][0];physical=p*n['std'][0]+n['mean'][0];raw=float((physical[:,5:].double()-y[:,5:].double()).square().mean())
        for site in (2,50,100):
            assim.issue(p,scale,[site],0);q=assim.observe(observation_rows(y)[[site]].cpu(),25);actual=raw-float((q-y[:,5:]).square().mean());expected=float(labels['target'][i,site]*labels['unit'][i]);err=abs(actual-expected);maximum=max(maximum,err);assert err<1e-9
    dump(CRITIC/'audit.json',dict(passed=True,public_replay=rows,training_single_site_gain_reconstruction_max_error=maximum,learned_coefficients=2432,common_assimilator_hash_unchanged=True,all_budgets_and_timestamps_verified=True,train_eval_trajectory_disjoint=True,source_sha256=digest(Path(__file__))));print('CRITIC_AUDIT_PASSED',maximum,flush=True)

if __name__=='__main__':main()
