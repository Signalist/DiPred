"""Protocol-bound numeric, gradient, identity and publication-boundary tests."""
from core import *
import tempfile,h5py


def main():
    configure();check();OUT.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'PREFLIGHT.json').exists()
    start=time.monotonic();model,norm=load_model();base=state_hash(model);checks={};details=[]
    torch.manual_seed(112)
    p=torch.randn(5,2,3,2,2,dtype=torch.double,requires_grad=True);y=torch.randn(2,3,2,2,dtype=torch.double)
    fast,first,pair=crps_parts(p,y)
    explicit=(p-y).abs().mean()-.5*(p[:,None]-p[None,:]).abs().mean()
    g1=torch.autograd.grad(fast,p,retain_graph=True)[0];g2=torch.autograd.grad(explicit,p,retain_graph=True)[0]
    assert abs(float(fast-explicit))<1e-10 and float((g1-g2).abs().max())<1e-10
    assert torch.autograd.gradcheck(lambda x:crps_parts(x,y)[0],(p,),eps=1e-6,atol=1e-5)
    checks['explicit_crps_value_error']=abs(float(fast-explicit));checks['explicit_crps_gradient_error']=float((g1-g2).abs().max());checks['gradcheck']=True
    selected=[e for e in entries('train') if e['start']==400 and e['repeat']==0 and e['trajectory'] in [r['file'] for r in CFG['records']['train'][:2]]]
    for entry in selected:
        b=prior.load_bundle(entry)
        assert torch.equal(prior.frames(entry['trajectory'],400,420),b['history'])
        noisehashes=[h for c in b['calls'].values() for h in c['noise_sha256']];assert len(noisehashes)==len(set(noisehashes))
        for k in CFG['eval_k']:
            h,mask,spread,z=prepare(b,k,norm)
            assert h.shape==(32,2,20,32,64)
            assert int(mask.sum())==k*32*64 and torch.count_nonzero(spread[:,:,:20-k])==0
            if k:
                expected=b['paths']['A'][:,:,:k].cuda().std(0,unbiased=False)[None]/norm['std']
                assert torch.equal(expected,spread[:,:,-k:])
            else:assert torch.count_nonzero(mask)==torch.count_nonzero(spread)==0
            target=b['paths']['B' if k==0 else 'S'+str(k)]
            initial=[]
            for arm in CFG['arms']:
                adapter=Adapter(arm,42).cuda();initial.append(state_hash(adapter))
                with torch.no_grad():pred=decode(sample(model,adapter,h,mask,spread,z),norm)
                error=float((pred.cpu()-target).abs().max());assert error<=CFG['replay_atol']
                details.append(dict(trajectory=entry['trajectory'],k=k,arm=arm,replay_maxabs=error,params=sum(v.numel() for v in adapter.parameters())))
            assert len(set(initial))==1
    adapter=Adapter('C_unc',42).cuda();h,mask,spread,z=prepare(b,5,norm)
    # Actual32-member objective and all four contributions to adapter output gradients.
    truth=(truth_for(entry,5)-norm['mean'][0])/norm['std'][0]
    trace=[];prediction=sample(model,adapter,h,mask,spread,z,trace)
    loss,first,pair=crps_parts(prediction,truth)
    parameters=tuple(adapter.parameters())
    gp=torch.autograd.grad(pair,parameters,retain_graph=True)
    gf=torch.autograd.grad(first,parameters,retain_graph=True)
    gl=torch.autograd.grad(loss,parameters,retain_graph=True)
    errors=[float((a-b+c).abs().max()) for a,b,c in zip(gl,gf,gp)]
    assert all(torch.allclose(a,b-c,rtol=1e-4,atol=1e-7) for a,b,c in zip(gl,gf,gp))
    pairnorm=float(sum(v.square().sum() for v in gp).sqrt());assert pairnorm>0
    loss.backward();gradients=[float(v.grad.square().mean().sqrt()) for v in trace];assert len(gradients)==4 and min(gradients)>0
    assert all(p.grad is None and not p.requires_grad for p in model.parameters())
    checks.update(pair_term_adapter_gradient_norm=pairnorm,gradient_decomposition_maxabs=max(errors),four_step_adapter_gradient_rms=gradients)
    # Nonzero fixed diagnostic parameters prove member-specific path change; no fitting.
    with torch.no_grad():adapter.out.weight.fill_(.001);adapter.out.bias.zero_()
    with torch.no_grad():p1=sample(model,adapter,h,mask,spread,z);p0=prior.sampler(model,h,z)
    shift=p1-p0;noncommon=float((shift-shift.mean(0)).abs().max());assert noncommon>1e-6
    checks['nonzero_adapter_noncommon_shift_maxabs']=noncommon
    # Synthetic data file: keep observed history unchanged, poison all future labels.
    with tempfile.TemporaryDirectory() as tmp:
        tmp=Path(tmp)
        for name,poison in [('a.h5',0),('b.h5',1000)]:
            with h5py.File(tmp/name,'w') as f:
                for c in 'uv':
                    arr=np.zeros((460,64,128),dtype='float32');arr[420:]=poison;f.create_dataset(c,data=arr)
        h1=prior.frames('a.h5',400,420,tmp).cuda();h2=prior.frames('b.h5',400,420,tmp).cuda()
        parent=b['paths']['A'].cuda();c1=conditions(h1,parent,5,norm);c2=conditions(h2,parent,5,norm)
        assert all(torch.equal(x,y) for x,y in zip(c1,c2))
        with torch.no_grad():v1=sample(model,adapter,*c1,z);v2=sample(model,adapter,*c2,z)
        assert torch.equal(v1,v2)
    checks['future_poison_input_output_maxabs']=0
    for k in CFG['k']:
        physical=native.physical_indices(400,k)
        assert physical['child'][:20-k]==physical['direct'][k:]
        tagged=torch.arange(32*2*20*32*64).reshape(32,2,20,32,64).float()
        mixed=native.restart_history(b['history'],tagged,k)
        assert torch.equal(mixed[:,:,-k:],tagged[:,:,:k])
    checks['physical_time_member_order']=True
    checks['backbone_unchanged']=state_hash(model)==base;assert checks['backbone_unchanged']
    # Every initial model shares state for each seed, not merely equal parameter counts.
    checks['seed_initialization']={str(seed):[state_hash(Adapter(arm,seed)) for arm in CFG['arms']] for seed in CFG['seeds']}
    assert all(len(set(v))==1 for v in checks['seed_initialization'].values())
    csvwrite(OUT/'preflight_replay.csv',details)
    dump(OUT/'PREFLIGHT.json',dict(passed=True,checks=checks,seconds=time.monotonic()-start,code=code_hashes(),config_sha256=digest(DELIVERY/'config.json'),checkpoint_hashes=CFG['frozen'],source_caches=selected,utc=time.time()))
    print('PASS full CRPS values/gradients/pair term; four steps;24 zero replays; provenance/std slots; future poison; member/time identity; frozen backbone',flush=True)

if __name__=='__main__':main()
