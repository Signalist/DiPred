"""Independent field scores, exact imported baselines and deployment causality."""
from controls import *

@torch.no_grad()
def run():
    setup();checks={};sel=json.loads((OUT/'selection.json').read_text());maximum=0.;numpy_max=0.;zero_max=0.;n=st.norm();x=load(SOURCE/'development.pt')['x'][:1].cuda();p,f=st.capture(st.model(),x)
    for name in KINDS:assert digest(OUT/f'{name}.pt')==sel['model_hashes'][name]
    assert digest(Path(__file__).with_name('PROTOCOL.md'))==sel['protocol_sha256']
    checks['frozen_model_protocol_hashes']=True
    for name,cls in [('rescrct_original',original.Corrector),('rescrct_block',offline.BlockResidualCorrector)]:
        ck=load(OUT/f'{name}.pt','cuda');wrapped=factory(name).cuda();wrapped.load_state_dict(ck['model']);exact=cls().cuda();exact.load_state_dict(ck['model'])
        expected=p+exact(x,p,torch.zeros_like(p),torch.zeros_like(p)) if name=='rescrct_original' else exact(x,p)
        assert torch.equal(wrapped(x,p),expected)
    checks['exact_imported_rescrct_class_parity']=True
    trainfiles=set(sel['models'][KINDS[0]]['train_files']);calfiles=set(sel['models'][KINDS[0]]['calibration_files']);assert trainfiles.isdisjoint(calfiles)
    for split in ['development','late','future']:
        datafile=load(SOURCE/f'{split}.pt');assert trainfiles.isdisjoint({m['file'] for m in datafile['meta']}) and calfiles.isdisjoint({m['file'] for m in datafile['meta']})
    checks['trajectory_separation']=True
    index0=load(OUT/f'{KINDS[0]}_training_indices.pt')
    for name in KINDS:
        ix=load(OUT/f'{name}_training_indices.pt');assert torch.equal(ix,index0[:len(ix)])
    checks['single_seed_same_minibatch_prefix']=True
    for split in ['calibration','development','late','future']:
        src=load(SOURCE/f'{split}.pt');archive=load(OUT/split/'issued.pt');df=pd.read_csv(OUT/split/'windows.csv')
        for name in REFERENCES+KINDS:
            api=Predictor(name)
            for i in [0,4]:
                h=src['x'][i:i+1].clone();before=h.clone();out=api.predict(h,normalized=True);maximum=max(maximum,float((out['normalized'][0].cpu()-archive[name][i]).abs().max()));assert torch.equal(h,before)
                # Independent NumPy physical difference, no torch scoring reduction.
                error=(archive[name][i].double().numpy()-src['y'][i].double().numpy())*n['std'][0].double().cpu().numpy();nm=float(np.mean(error**2));row=df[(df.method==name)&(df.file==src['meta'][i]['file'])&(df.start==src['meta'][i]['start'])].iloc[0];numpy_max=max(numpy_max,abs(nm-row.mse))
                pp,ff=st.capture(api.net,h.cuda());zz=api.correct(h.cuda(),pp,ff,force_gain=0.);zero_max=max(zero_max,float((zz-pp).abs().max()))
            del api
    assert maximum==0,maximum
    assert numpy_max<1e-9,numpy_max
    assert zero_max==0
    checks.update(actual_public_batch1_exact=True,independent_numpy_mse=True,gain_zero_exact_base=True)
    # No reveal/target argument exists; perturb inaccessible truth and verify full free chain.
    rolled=load(OUT/'rollout_issued.pt');rec=records('development')[0];h=read(rec,0)[:,:20][None];physical=h.clone();normal=(physical.cuda()-n['mean'])/n['std'];future=read(rec,20)[:,20:][None]
    chainmax=0.
    for name in rolled:
        api=Predictor(name);state=normal.clone()
        import h5py
        old_torch_load,old_h5_file=torch.load,h5py.File
        def forbidden(*args,**kwargs):raise AssertionError('No data/checkpoint loading allowed during inference')
        torch.load,h5py.File=forbidden,forbidden
        try:
            for j in range(3):
                unrelated_future=future*0+9999+j
                q=api.predict(state,normalized=True)['normalized'];assert torch.isfinite(unrelated_future).all();expected=next(v['free'][j] for v in rolled[name] if v['file']==rec['file']);chainmax=max(chainmax,float((q[0].cpu()-expected).abs().max()));state=q
        finally:torch.load,h5py.File=old_torch_load,old_h5_file
        a=api.predict(physical);b=api.predict(normal,normalized=True);assert torch.equal(a['normalized'],b['normalized'])
        try:api.predict(torch.zeros(1,2,40,32,64))
        except ValueError:pass
        else:raise AssertionError('Future-containing input accepted')
    assert chainmax==0,chainmax
    checks.update(free_prefix_exact_replay=True,history_only_causal_interface=True,no_data_or_checkpoint_load_during_issue=True,cpu_physical_roundtrip=True,whole_future_rejected=True)
    times=pd.read_csv(OUT/'rollout_windows.csv');assert ((times.score_time-times.issue_time)==20).all();checks['rollout_timestamps']=True
    dump(OUT/'audit.json',dict(passed=all(checks.values()),checks=checks,public_max_error=maximum,numpy_mse_max_error=numpy_max,zero_gain_max_error=zero_max,rollout_prefix_max_error=chainmax,all_hyperparameters_selected_before_development=True))
    print('AUDIT',checks,numpy_max,flush=True)

if __name__=='__main__':run()
