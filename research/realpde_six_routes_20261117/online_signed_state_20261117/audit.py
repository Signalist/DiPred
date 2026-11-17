"""Independent physical reconstruction and legal-feedback audit."""
from predict import *

def main():
    setup();models=load(OUT/'models.pt');basis=load(OUT/'basis.pt');b=basis['basis'].numpy();sd=basis['normalization']['std'].numpy();selection=json.loads((OUT/'selection.json').read_text());checks={};maximum=0.
    for split in ('calibration','development','late','future'):
        small=load(OUT/f'{split}.pt');source=load(SOURCE/f'{split}.pt');r=((source['y'].numpy()-source['p'].numpy())*sd).astype(np.float64);indices=[0,len(r)//2,len(r)-1]
        for scenario in ('natural','step','ramp'):
            d=shifted(small,scenario,basis);saved=load(OUT/f'{split}_{scenario}_issued.pt')['issued']
            for method,c in saved.items():
                analytical=d['base_mse']+(c**2).sum(1)-2*(c*d['c']).sum(1)
                for i in indices:
                    shifted_error=r[i].reshape(-1)-(d['drift_coefficient'][i]+c[i])@b;direct=np.mean(shifted_error**2);error=abs(direct-analytical[i]);maximum=max(maximum,error);assert error<1e-10,(split,scenario,method,error)
        checks[split]=dict(windows=len(r),physical_reconstruction_indices=indices)
    d=load(OUT/'development.pt');groups=group_ids(d['meta']);ix=groups[0];intervention=ix[3];prefix=ix[:4]
    causal={}
    for name,cfg in selection['models'].items():
        before,_,_=simulate(d,cfg['config'],models);after,_,_=simulate(d,cfg['config'],models,lambda i,c:c+1 if i==intervention else c)
        assert np.array_equal(before[prefix],after[prefix]);outside=[i for g in groups[1:] for i in g];assert np.array_equal(before[outside],after[outside])
        controller=State(cfg['config'],models);controller.issue(d['f'][0],0)
        try:controller.reveal(d['c'][0],39);raise AssertionError('Early reveal accepted')
        except ValueError:pass
        assert controller.last_revealed is None and len(controller.history)==0;controller.skip_feedback();controller.issue(d['f'][1],20);controller.skip_feedback();assert controller.last_revealed is None and len(controller.history)==0
        causal[name]=dict(future_intervention_preserves_prefix=True,trajectory_isolation=True,early_reveal_rejected=True,skip_does_not_create_feedback=True)
    # Real target-free predictor replay for an entire trajectory, including
    # both cold-start and updated states; targets enter only reveal().
    direct={};source=load(SOURCE/'development.pt');saved=load(OUT/'development_natural_issued.pt')['issued']
    for method in ('rolling_bias','rw_kalman','conditional_kalman','linear_ar','rnn'):
        pred=OnlinePredictor(method);error=0.
        for i in ix:
            meta=d['meta'][i];q=pred.predict(source['x'][i:i+1],meta['start'],True)
            expected=source['p'][i:i+1].numpy()+((saved[method][i]@b).reshape(1,2,20,32,64)/sd)
            error=max(error,float(np.max(np.abs(q.detach().cpu().numpy()-expected))));pred.reveal(source['y'][i:i+1],meta['start']+40,True)
        assert error<3e-5,(method,error);direct[method]=error;del pred;torch.cuda.empty_cache()
    dump(OUT/'audit.json',dict(passed=True,physical_mse_max_error=maximum,physical_splits=checks,causal_checks=causal,direct_normalized_field_max_error=direct,protocol_hash_matches=selection['protocol_sha256']==digest(W/'online_signed_state_20261117/PROTOCOL.md'),models_hash_matches=selection['models_sha256']==digest(OUT/'models.pt')))
    print('AUDIT_PASSED',maximum,direct,flush=True)

if __name__=='__main__':main()
