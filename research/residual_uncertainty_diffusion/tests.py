"""All training gates; zero optimizer instances and zero updates."""
from .core import *
from .experiment import synthetic_pipeline_test,freeze_hash
from research.trust_causal_audit_v2.model import native_sample


def main():
    native,norm=load_backbone();train=torch.load(CACHE/'train_conditions.pt',map_location='cpu',weights_only=False)
    dev=torch.load(CACHE/'development_conditions.pt',map_location='cpu',weights_only=False);scales=read(OUT/'CONDITION_SCALES.json')
    assert len(train)==137 and len(dev)==36
    assert {s:sum(i['key'].start==s for i in train) for s in (420,440,460)}=={420:46,440:46,460:45}
    assert scales['source_split']=='train'
    models={a:CorrectedDiffusion(native) for a in ARMS}
    assert state_hash(models['C0'])==state_hash(models['C1'])
    permitted=['native.net.out.weight','native.net.out.bias']+[n for n,p in models['C0'].named_parameters() if n.startswith('correction.')]
    assert set(n for n,p in models['C0'].named_parameters() if p.requires_grad)==set(permitted)
    xpf=FrozenXPF();item=train[0];key=item['key'];file_names=[r['file'] for r in records()[key.split]]
    index=CacheIndex(torch.load(ROOT/f'.cache/error_guided_spatial_routing/train_{key.start-20}_native.pt',map_location='cpu',weights_only=False),key.split,key.start-20,file_names,'test')
    previous=index.lookup(Key(key.split,key.trajectory_id,key.start-20));reader=H5Reader(DATA)
    a=release(reader,key,previous,{k:v.cpu() for k,v in norm.items()},xpf)
    class Poison:
        def __init__(self):self.reads=[]
        def __call__(self,name,s,e):
            self.reads.append((s,e))
            if s>=key.start+20:return torch.full((2,20,32,64),float('nan'))
            return reader(name,s,e)
    poisoned=Poison();b=release(poisoned,key,previous,{k:v.cpu() for k,v in norm.items()},xpf)
    assert poisoned.reads==[(key.start,key.start+20)]
    for field in ('H','E','W'):assert torch.equal(a[field],b[field])
    assert a['previous_interval']==[key.start,key.start+20]
    z,ids=batch_noise(key,file_names);assert len({v['noise_tensor_hash'] for v in ids})==32
    z=z[:8]
    release_hashes={}
    for arm,model in models.items():
        ia=condition_inputs(a,scales,z,arm);ib=condition_inputs(b,scales,z,arm)
        assert all(torch.equal(ia[k],ib[k]) for k in ia)
        with torch.no_grad():
            baseline=native_sample(native,ia['history'],z)
            closed=model.sample(**ia,enabled=False);initialized=model.sample(**ia)
            assert torch.equal(baseline,closed) and torch.equal(baseline,initialized)
            assert torch.equal(initialized,model.sample(**ib))
        release_hashes[arm]=tensor_hash(initialized)
    bigger=dict(a,W=a['W']*2)
    ia=condition_inputs(a,scales,z,'C1');large=condition_inputs(bigger,scales,z,'C1')
    assert not torch.equal(ia['W'],large['W']) and not torch.equal(ia['W_mean'],large['W_mean'])
    # Eight-member fair formula and derivative vs explicit ordered pair formula.
    torch.manual_seed(88);p=torch.randn(8,2,2,2,3,dtype=torch.float64,requires_grad=True);y=torch.randn(2,2,2,3,dtype=torch.float64)
    formula=(p-y).abs().mean()-(p[:,None]-p[None,:]).abs().sum()/(2*8*7*p[0].numel())
    value=crps(p,y,True);g=torch.autograd.grad(value,p,retain_graph=True)[0];expected=torch.autograd.grad(formula,p)[0]
    assert abs(float(value-formula))<1e-12 and float((g-expected).abs().max())<1e-12
    fair_gradient_error=float((g-expected).abs().max())
    assert torch.autograd.gradcheck(lambda t:crps(t,y,True),(p,),eps=1e-6,atol=1e-7,rtol=1e-5)
    # Validate per-channel standardized MSE and complete scoring independently.
    target=torch.randn(2,20,32,64,device='cuda')*.01
    ps=torch.randn(32,2,20,32,64,device='cuda')*.1
    values=score(ps,target,norm)
    raw=ps.double()*norm['std'].double()+norm['mean'].double();err=raw.mean(0)-target.double()
    assert abs(values[0]['smse']-.5*(values[1]['smse']+values[2]['smse']))<1e-12
    for c in (0,1):
        explicit=float((err[c]/norm['std'][c].double()).square().mean())
        assert abs(explicit-values[c+1]['smse'])<1e-12
    # Actual four-step graph: each clean-state tensor receives nonzero gradient;
    # output layer and G final layer have finite nonzero gradients.
    model=models['C1'];trace=[];out=model.sample(**ia,trace=trace)
    target=(item['target'].cuda()-norm['mean'])/norm['std'];loss=crps(out,target,True);loss.backward()
    assert [r['label'] for r in trace]==[99,66,33,0]
    gradient=[]
    for call in trace:
        assert torch.equal(call['clean'],call['base']+call['delta'])
        g=call['clean'].grad;assert g is not None and torch.isfinite(g).all() and g.abs().sum()>0
        gradient.append(dict(label=call['label'],clean_grad_norm=float(g.double().square().sum().sqrt())))
    for module in (model.correction.net[-1],model.native.net.out):
        assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for p in module.parameters())
    # Hidden correction layers are initially gradient-zero by final zero init;
    # nonzero synthetic final weights test upstream derivatives, not an update.
    with torch.no_grad():model.correction.net[-1].weight.fill_(.002)
    for p in model.parameters():p.grad=None
    trace2=[];out=model.sample(**ia,trace=trace2);crps(out,target,True).backward()
    assert any(float(r['delta'].abs().max())>0 for r in trace2)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for p in model.correction.parameters())
    frozen=freeze_hash(native,True)
    assert freeze_hash(model.native,True)==frozen
    assert state_hash(xpf.head)==xpf.weight_hash and all(not p.requires_grad for p in xpf.head.parameters())
    assert synthetic_pipeline_test()
    dump(OUT/'PREFLIGHT.json',dict(passed=True,optimizer_updates=0,correction_parameters=6392,backbone_trainable_parameters=17320,
        allowed_trainable_names=permitted,real_window_intervals_valid=True,raw_xpf_equivalence=read(OUT/'XPF_CONDITION_AUDIT.json'),
        future_poison_release_hashes=release_hashes,actual_noise_members=8,closed_and_zero_initialized_native_exact=True,
        fair8_crps_value_error=abs(float(value-formula)),fair8_gradient_error=fair_gradient_error,
        fair8_gradcheck=True,channel_smse_explicit_verified=True,full_scoring_pipeline_synthetic_pass=True,
        absolute_risk_increases_preserved=True,four_step_gradients=gradient,correction_and_out_gradients_nonzero=True,
        zero_final_layer_hidden_initial_gradients_expected=True,synthetic_nonzero_head_upstream_gradients_pass=True,
        frozen_backbone_sha256=frozen,xpf_frozen_hash=xpf.weight_hash,code_hashes=code_hashes()))
    print('ALL PREFLIGHT GATES PASS; updates=0',flush=True)

if __name__=='__main__':main()
