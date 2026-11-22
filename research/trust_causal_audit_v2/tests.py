"""A-G end-to-end gates and intentionally broken minimal regressions.

No optimizer is constructed. A failure is written as a failure, never a green
placeholder. CUDA is required to reproduce the original cache's actual noise.
"""
import copy
import time
import torch
from .common import OUT, MEMBERS, dump, load_backbone, state_hash, tensor_hash, code_hashes
from .data import Key, Issued, CacheIndex, window_catalog, fit_scale, raw_risk, prepare_inputs, model_inputs, batch_noise
from .model import TrustDiffusion, SpatialRouter, routing, neighbors, geometry, native_sample, crps_parts, energy_score


class MemoryReader:
    def __init__(self, arrays):
        self.arrays = arrays
        self.reads = []
    def __call__(self,name,begin,end):
        self.reads.append((name,begin,end))
        return self.arrays[name][:,begin:end].clone()


def must_detect(callback):
    try:
        callback()
    except (AssertionError, ValueError, IndexError):
        return True
    raise AssertionError('Injected defect escaped the regression test')


def run():
    started = time.perf_counter()
    torch.manual_seed(731)
    native,norm = load_backbone()
    base_hash = state_hash(native)
    models = {name:TrustDiffusion(copy.deepcopy(native),name=='G_trust')
              for name in ('G_context','G_trust')}
    params = {name:sum(p.numel() for p in m.router.parameters()) for name,m in models.items()}
    assert params['G_context']==params['G_trust'] and params['G_trust']<=10000
    assert state_hash(models['G_context'].router)==state_hash(models['G_trust'].router)
    norm_cpu = {k:v.cpu() for k,v in norm.items()}
    key = Key('train','synthetic.h5',420)
    physical = torch.randn(2,500,32,64)*norm_cpu['std']+norm_cpu['mean']
    hist = physical[:,420:440].clone()
    prev = Issued(Key('train',key.trajectory_id,400),
        hist[None]+torch.randn(32,2,20,32,64)*norm_cpu['std']*.2,
        (400,420),(420,440),MEMBERS,0,'synthetic-issued-native').validate()
    R = raw_risk(hist,prev,norm_cpu['std'])
    scale = fit_scale([R], 'train')
    reader_a = MemoryReader({key.trajectory_id:physical.clone()})
    changed = physical.clone();changed[:,440:460] = torch.nan
    reader_b = MemoryReader({key.trajectory_id:changed})
    prepared_a = prepare_inputs(reader_a,key,prev,norm_cpu,scale)
    prepared_b = prepare_inputs(reader_b,key,prev,norm_cpu,scale)
    assert reader_a.reads==reader_b.reads==[(key.trajectory_id,420,440)]
    z,noise_ids = batch_noise(key,[key.trajectory_id])
    inputs_a,inputs_b = model_inputs(prepared_a,z),model_inputs(prepared_b,z)
    assert torch.equal(prepared_a['risk'],prepared_b['risk'])
    assert all(torch.equal(inputs_a[k],inputs_b[k]) for k in inputs_a)
    prediction_hashes = {}
    with torch.no_grad():
        for name,model in models.items():
            a = model.sample(**inputs_a)
            b = model.sample(**inputs_b)
            assert torch.equal(a,b), name
            prediction_hashes[name]=tensor_hash(a)
    # Regression G1: current future labels accidentally used by risk builder.
    def leaked_risk(reader):
        future = reader(key.trajectory_id,440,460)
        return (torch.log1p(((prev.paths.mean(0)-future)/norm_cpu['std']).square().mean((0,1))) / scale['log1p_p99']).clamp(0,1)
    leak_detected = must_detect(lambda: torch.testing.assert_close(leaked_risk(reader_a),leaked_risk(reader_b),equal_nan=False,atol=0,rtol=0))
    # B: change revealed past, keep future labels byte-identical.
    alternate = physical.clone();alternate[:,:,8:16,15:25] += norm_cpu['std']*.7
    alternate[:,440:] = physical[:,440:]
    reader_c = MemoryReader({key.trajectory_id:alternate})
    c = prepare_inputs(reader_c,key,prev,norm_cpu,scale)
    expected = ((prev.paths.mean(0)-alternate[:,420:440])/norm_cpu['std']).square().mean((0,1))[None]
    assert torch.equal(c['raw_risk'],expected)
    assert not torch.equal(c['risk'],prepared_a['risk'])
    assert torch.equal(alternate[:,440:460],physical[:,440:460])

    # C/G2/G3: the short middle trajectory shifts later indices if lookup is
    # positional. Give each file a unique value, and vary lengths by window.
    files = ['first.h5','21600_5.h5','after_short.h5']
    lengths = dict(zip(files,[520,492,520]))
    cat = window_catalog('train',files,lengths,(420,440,460))
    assert [sum(r['key'].start==s for r in cat) for s in (420,440,460)]==[3,3,2]
    ids = {name:float(11+i*7) for i,name in enumerate(files)}
    def fake_cache(start):
        p = torch.stack([torch.full((32,2,20,32,64),ids[name]+start/10000) for name in files])
        payload = dict(identity=dict(split='train',start=start,files=files,members=list(MEMBERS)),paths=p)
        return CacheIndex(payload,'train',start,files,'synthetic')
    cache = {s:fake_cache(s) for s in (400,420,440)}
    for row in cat:
        k=row['key'];p=cache[k.start-20].lookup(Key(k.split,k.trajectory_id,k.start-20))
        assert p.cache_row==files.index(k.trajectory_id)
        assert torch.equal(p.paths,torch.full_like(p.paths,ids[k.trajectory_id]+(k.start-20)/10000))
    def legacy_valid():
        actual = {s:[r['key'].trajectory_id for r in cat if r['key'].start==s] for s in (420,440,460)}
        for s in (420,440,460):
            assert [name for name in files if lengths[name]>=460+40] == actual[s]
    last_valid_detected = must_detect(legacy_valid)
    def legacy_position():
        final = [r for r in cat if r['key'].start==460]
        for position,row in enumerate(final):
            k=row['key'];wrong=cache[440].paths[position]
            expected = cache[440].lookup(Key('train',k.trajectory_id,440)).paths
            assert torch.equal(wrong,expected)
    position_detected = must_detect(legacy_position)
    # Reject metadata identity mismatch as well as checking file/values above.
    identity_rejected = must_detect(lambda: cache[400].lookup(Key('development',files[0],400)))

    # D: batch reconstruction is deterministic, distinct members stay distinct,
    # and disabled wrapper executes all four steps rather than bypassing itself.
    z_again,noise_ids_again = batch_noise(key,[key.trajectory_id])
    assert torch.equal(z,z_again) and noise_ids==noise_ids_again
    assert len({i['noise_tensor_hash'] for i in noise_ids})==32
    disabled_errors={}
    with torch.no_grad():
        expected=native_sample(native,inputs_a['history'],z)
        for name,model in models.items():
            actual=model.sample(**inputs_a,enabled=False)
            err=float((actual-expected).abs().max());assert err==0.,err
            disabled_errors[name]=err

    # E: controlled routing law, legal finite boundaries, optional obstacles,
    # and independent ensemble-member computation.
    f=torch.randn(2,48,5,7,dtype=torch.float64)
    risk=torch.zeros(2,1,5,7,dtype=torch.float64)
    _,mask=geometry(f)
    assert mask[0,:,0,0].sum()==3 and mask[0,:,0,3].sum()==5 and mask[0,:,2,3].sum()==8
    logits=torch.zeros(2,8,5,7,dtype=torch.float64);q=torch.zeros(2,1,5,7,dtype=torch.float64)
    a0,g0=routing(logits,q,risk,mask,True)
    rj=risk.clone();rj[:,:,2,4]=1
    a1,_=routing(logits,q,rj,mask,True)
    # Neighbor direction (0,+1) is slot 4 of the eight valid directions.
    assert (a1[:,4,2,3]<a0[:,4,2,3]).all()
    ri=risk.clone();ri[:,:,2,3]=1
    _,g1=routing(logits,q,ri,mask,True)
    assert (g1[:,:,2,3]>g0[:,:,2,3]).all()
    assert (g1>=0).all() and (g1<=.25).all()
    assert (a1>=0).all() and torch.allclose(a1.sum(1),torch.ones(2,5,7,dtype=torch.float64),atol=1e-14,rtol=0)
    assert torch.equal(a1[~mask],torch.zeros_like(a1[~mask]))
    obstacle=torch.ones(2,1,5,7,dtype=torch.bool);obstacle[:,:,2,4]=False
    _,omask=geometry(f,obstacle)
    oa,og=routing(logits,q,risk,omask,True)
    assert not omask[:,4,2,3].any() and (og[:,:,2,4]==0).all()
    assert (oa[:,:,2,4]==0).all()
    isolated=torch.zeros_like(obstacle);isolated[:,:,2,3]=True
    _,imask=geometry(f,isolated);ia,ig=routing(logits,q,risk,imask,True)
    assert torch.isfinite(ia).all() and ia.sum()==0 and ig.sum()==0
    router=SpatialRouter(True).double()
    hh=torch.randn(2,2,20,5,7,dtype=torch.float64);pp=torch.randn_like(hh)
    tt=torch.tensor([99,99]);rr=torch.rand(2,1,5,7,dtype=torch.float64)
    trace=[];before=router(f,hh,pp,tt,rr,trace=trace)
    changed_f=f.clone();changed_f[0]+=100
    after=router(changed_f,hh,pp,tt,rr)
    assert torch.equal(before[1],after[1])
    assert torch.equal(before[1],router(f[1:],hh[1:],pp[1:],tt[1:],rr[1:])[0])
    # Member permutation is only a diagnostic of equivariance, never inference
    # matching/sorting. Each full path retains its input row identity.
    perm=torch.tensor([1,0]);equiv=router(f[perm],hh[perm],pp[perm],tt[perm],rr[perm])
    assert torch.equal(equiv,before[perm])
    ctx=SpatialRouter(False).double();ctxtrace=[]
    ctx(f,hh,pp,tt,rr,trace=ctxtrace)
    assert torch.equal(ctxtrace[0]['learned_inputs'],trace[0]['learned_inputs'])
    assert torch.equal(router(f,hh,pp,tt,torch.zeros_like(rr)),ctx(f,hh,pp,tt,torch.zeros_like(rr)))
    # Full history and previous mean actually affect both learned heads.
    assert not torch.equal(router(f,hh+1,pp,tt,rr),before)
    assert not torch.equal(ctx(f,hh,pp+1,tt,rr),ctx(f,hh,pp,tt,rr))

    # F: exact full32 CRPS value, two analytic gradients, finite differences;
    # channel standardization changes ES correctly, without unit conflation.
    torch.manual_seed(83)
    p=torch.randn(32,2,2,2,3,dtype=torch.float64,requires_grad=True)
    y=torch.randn(2,2,2,3,dtype=torch.float64)
    loss,first,pair=crps_parts(p,y)
    explicit=(p-y).abs().mean()-.5*(p[:,None]-p[None,:]).abs().mean()
    grad=torch.autograd.grad(loss,p,retain_graph=True)[0]
    eg=torch.autograd.grad(explicit,p,retain_graph=True)[0]
    assert abs(float(loss-explicit))<1e-12 and float((grad-eg).abs().max())<1e-12
    assert torch.autograd.gradcheck(lambda x:crps_parts(x,y)[0],(p,),eps=1e-6,atol=1e-7,rtol=1e-5)
    std=torch.tensor([.1,.02],dtype=torch.float64).reshape(2,1,1,1)
    raw_es=energy_score(p,y);std_es=energy_score(p,y,std)
    flat=p.flatten(1);yt=y.flatten()
    raw_explicit=torch.linalg.vector_norm(flat-yt,dim=1).mean()/flat.shape[1]**.5
    raw_explicit-=.5*torch.cdist(flat,flat,compute_mode='donot_use_mm_for_euclid_dist').mean()/flat.shape[1]**.5
    assert abs(float(raw_es-raw_explicit))<1e-12
    assert abs(float(std_es-energy_score(p/std,y/std)))<1e-12
    assert abs(float(raw_es-std_es))>1e-3
    torch.testing.assert_close(energy_score(7*p,7*y),7*raw_es,atol=1e-12,rtol=1e-12)
    torch.testing.assert_close(energy_score(7*p,7*y,7*std),std_es,atol=1e-12,rtol=1e-12)
    # Four-call VJPs: retain the actual loss gradient at each routing output and
    # recompute each call with its frozen input detached. This isolates that
    # call's direct parameter contribution from earlier recursive calls.
    perstep={}
    for name,model in models.items():
        hook_features=[]
        handle=model.native.net.first.register_forward_hook(lambda m,args,out:hook_features.append(out.detach()))
        trace=[]
        pred=model.sample(**inputs_a,trace=trace)
        target=torch.randn_like(pred[0])*.2
        loss,_,pair=crps_parts(pred,target)
        parameters=list(model.router.parameters())
        total=torch.autograd.grad(loss,parameters,retain_graph=True)
        pair_grad=torch.autograd.grad(pair,parameters,retain_graph=True)
        assert sum(float(g.abs().sum()) for g in total)>0
        assert sum(float(g.abs().sum()) for g in pair_grad)>0
        details=[]
        for i,call in enumerate(trace):
            downstream=torch.autograd.grad(loss,call['output'],retain_graph=True)[0].detach()
            label=torch.full((32,),call['label'],device='cuda',dtype=torch.long)
            local=model.router(hook_features[i],inputs_a['history'],inputs_a['previous_mean'],label,inputs_a['risk'])
            contributions=torch.autograd.grad(local,parameters,grad_outputs=downstream)
            norm_by_head={k:float(g.double().square().sum().sqrt()) for (k,_),g in zip(model.router.named_parameters(),contributions)}
            assert all(v>0 and torch.isfinite(torch.tensor(v)) for v in norm_by_head.values())
            details.append(dict(label=call['label'],direct_parameter_gradient_norms=norm_by_head))
        assert [x['label'] for x in details]==[99,66,33,0]
        handle.remove();perstep[name]=details
        assert state_hash(model.native)==base_hash and all(p.grad is None for p in model.native.parameters())
    assert state_hash(native)==base_hash
    results=dict(passed=True,optimizer_updates=0,optimizer_instances=0,training_runs=0,
        parameters=params,identical_parameter_initialization=True,
        A=dict(passed=True,reads_before_mutation=[420,440],raw_and_model_inputs_equal=True,issued_equal=True,prediction_hashes=prediction_hashes),
        B=dict(passed=True,past_risk_changes=True,future_labels_unchanged=True),
        C=dict(passed=True,per_window_counts=[3,3,2],middle_short_trajectory_kept_in_earlier_windows=True),
        D=dict(passed=True,full32_all20_four_steps_disabled_errors=disabled_errors,noise_identity_fields=list(noise_ids[0])),
        E=dict(passed=True,lower_neighbor_prior_with_higher_neighbor_risk=True,higher_gate_with_higher_relative_local_risk=True,
               gate_bound=.25,boundary_neighbor_counts=[3,5,8],obstacle_and_isolated_cell_checked=True,member_independence=True,same_learned_information=True),
        F=dict(passed=True,full32_crps_value_error=abs(float(crps_parts(p,y)[0]-explicit)),
               full32_crps_gradient_error=float((grad-eg).abs().max()),finite_difference_gradcheck=True,
               energy_original=float(raw_es),energy_standardized=float(std_es),energy_units_verified=True,
               four_step_direct_gradients=perstep),
        G=dict(passed=True,future_leak_mutant_rejected=leak_detected,last_valid_mutant_rejected=last_valid_detected,
               positional_cache_mutant_rejected=position_detected,wrong_split_rejected=identity_rejected),
        backbone_state_sha256_before=base_hash,backbone_state_sha256_after=state_hash(native),
        code_hashes=code_hashes(),seconds=time.perf_counter()-started)
    dump(OUT/'PREFLIGHT.json',results)
    return results


if __name__=='__main__':
    import sys
    try:
        result=run();print('A-G PASS; zero optimizer updates; parameters',result['parameters'],flush=True)
    except Exception as exc:
        dump(OUT/'PREFLIGHT_FAILURE.json',dict(passed=False,error=repr(exc)))
        raise
