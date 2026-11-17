"""Independent NumPy identities, causal public inference, frozen-model replay."""
from task_stream import *

def numpy_tasks(a,dx,dy):
    u,v=a[:,0],a[:,1];curl=(v[:,:,1:-1,2:]-v[:,:,1:-1,:-2])/(2*dx)-(u[:,:,2:,1:-1]-u[:,:,:-2,1:-1])/(2*dy)
    centered=a-a.mean(2,keepdims=True)
    return np.stack([.5*(a*a).sum(1).mean((1,2,3)),.5*(centered*centered).sum(1).mean((1,2,3)),.5*(curl*curl).mean((1,2,3)),u[:,:,:,48:].mean((1,2,3))],1)

@torch.no_grad()
def run():
    setup();tasks=Tasks();ck=load(OUT/'models.pt','cuda');sel=json.loads((OUT/'selection.json').read_text());checks={};numeric=0.;replay=0.;intervention=0.
    assert digest(Path(__file__).with_name('PROTOCOL.md'))==sel['protocol_sha256']
    assert digest(OUT/'models.pt')==sel['model_sha256']
    checks['frozen_hashes']=True
    if 'simple_controls_addendum_sha256' in sel:
        assert digest(Path(__file__).with_name('SIMPLE_CONTROL_ADDENDUM.md'))==sel['simple_controls_addendum_sha256']
        original=load(OUT/'initial_models.pt','cuda')
        assert ck['chosen']==original['chosen']
        assert torch.equal(ck['models'][ck['chosen']]['coef'],original['models'][original['chosen']]['coef'])
        for key in ['mean','sd','basis','scale']:
            assert torch.equal(ck['models'][ck['chosen']]['projection'][key],original['models'][original['chosen']]['projection'][key])
        checks['original_primary_unchanged_after_additional_controls']=True
    assert set(sel['fit_files']).isdisjoint(sel['calibration_files'])
    for split in ['development','late','future']:
        data=load(OUT/f'{split}.pt');assert set(sel['fit_files']+sel['calibration_files']).isdisjoint({m['file'] for m in data['meta']})
    checks['trajectory_separation']=True
    xx=torch.arange(64,dtype=torch.float64,device='cuda')*tasks.dx;yy=torch.arange(32,dtype=torch.float64,device='cuda')*tasks.dy
    linear=torch.zeros((1,2,20,32,64),device='cuda',dtype=torch.float64);linear[:,0]=-yy[None,:,None];linear[:,1]=xx[None,None,:]
    assert float((tasks.curl(linear)-2).abs().max())<1e-12
    checks['signed_coordinate_stencil']=True
    for split in ['calibration','development','late','future']:
        p,y=physical_fields(split);d=load(OUT/f'{split}.pt','cuda')
        for i in [0,4]:
            pp=p[i:i+1];r=y[i:i+1]-pp;l,q=tasks.expansion(pp,r)
            np_p=numpy_tasks(pp.cpu().numpy(),tasks.dx,tasks.dy);np_y=numpy_tasks(y[i:i+1].cpu().numpy(),tasks.dx,tasks.dy);np_q=numpy_tasks(r.cpu().numpy(),tasks.dx,tasks.dy);np_q[:,3]=0
            # Symmetric polarization, independently avoiding analytic Torch products.
            np_l=(numpy_tasks((pp+r).cpu().numpy(),tasks.dx,tasks.dy)-numpy_tasks((pp-r).cpu().numpy(),tasks.dx,tasks.dy))/2
            for a,b in [(d['jp'][i:i+1].cpu().numpy(),np_p),(d['jy'][i:i+1].cpu().numpy(),np_y),(l.cpu().numpy(),np_l),(q.cpu().numpy(),np_q)]:numeric=max(numeric,float(np.abs(a-b).max()))
            assert np.all(np.abs(np_y-np_p)<=tasks.bound(d['jp'][i:i+1],d['rho'][i:i+1]).cpu().numpy()+1e-10)
        altered=dict(d);altered.update(jy=torch.full_like(d['jy'],123),linear=torch.full_like(d['linear'],-999),quad=torch.zeros_like(d['quad']),rho=torch.full_like(d['rho'],9999))
        for name,model in ck['models'].items():
            a=prediction(d,model,tasks,p);b=prediction(altered,model,tasks,p);intervention=max(intervention,float((a-b).abs().max()))
        del p,y
    assert numeric<1e-10,numeric
    assert intervention==0
    checks.update(independent_numpy_tasks_and_taylor=True,oracle_cauchy_bound=True,future_target_intervention=True)
    # All operational methods on the same actual window; selected one on all segments.
    source=load(SOURCE/'development.pt');cached=load(OUT/'development'/'predictions.pt')
    for name in ck['models']:
        api=TaskErrorPredictor(name);out=api.predict(source['x'][:1],normalized=True)
        replay=max(replay,float((out['task_estimates'].cpu()-cached[name][0]).abs().max()))
        del api
    api=TaskErrorPredictor()
    for split in ['calibration','development','late','future']:
        source=load(SOURCE/f'{split}.pt');cache=load(OUT/split/'predictions.pt')
        for i in [0,4]:
            history=source['x'][i:i+1];before=history.clone();out=api.predict(history,normalized=True)
            replay=max(replay,float((out['task_estimates'].cpu()-cache[api.method][i]).abs().max()))
            assert torch.equal(before,history)
    assert replay<1e-8,replay
    checks['actual_batch1_public_replay']=True
    # Physical input is compared against explicit roundtrip-normalized input.
    history=source['x'][:1];n=st.norm('cpu');physical=history*n['std']+n['mean'];a=api.predict(physical);b=api.predict((physical-n['mean'])/n['std'],normalized=True)
    assert torch.equal(a['task_estimates'],b['task_estimates'])
    checks['cpu_physical_input']=True
    try:api.predict(torch.zeros(1,2,40,32,64))
    except ValueError:checks['whole_future_rejected']=True
    assert all(checks.values())
    dump(OUT/'audit.json',dict(passed=True,checks=checks,numpy_max_error=numeric,public_replay_max_error=replay,target_intervention_max_error=intervention,selected=api.method,protocol_sha256=sel['protocol_sha256']))
    print('AUDIT',numeric,replay,checks,flush=True)

if __name__=='__main__':run()
