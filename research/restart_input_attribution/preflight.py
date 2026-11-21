"""Fixed train-only numerical checks; no tolerance search or outcome selection."""
from core import *
import tempfile


def poisoning_test(model,norm,b,k):
    # Synthetic HDF5 has identical H/prefix but poisoned later truth. Production reader is exercised.
    h=b['history'];a=b['paths']['A'];start=400
    with tempfile.TemporaryDirectory() as tmp:
        data=Path(tmp)
        base=np.zeros((440,64,128),dtype=np.float32)
        for name,poison in [('clean.h5',0),('poison.h5',100000)]:
            with h5py.File(data/name,'w') as f:
                for c in 'uv':
                    x=base.copy();x[start+20+k:]=poison;f.create_dataset(c,data=x)
        prefix1=oracle_prefix('clean.h5',start,k,data);prefix2=oracle_prefix('poison.h5',start,k,data)
        x=inputs(h,a,prefix1,k);y=inputs(h,a,prefix2,k)
        assert all(torch.equal(v,w) for v,w in zip(x,y))
        try:inputs(h,a,torch.zeros(2,20,32,64),k)
        except AssertionError:pass
        else:raise AssertionError('full future incorrectly accepted')
        z=noise_for(b,k,x[2].cuda())[:1]
        differences=[]
        with torch.no_grad():
            for v,w in zip(x,y):
                v=v[None] if v.ndim==4 else v[:1];w=w[None] if w.ndim==4 else w[:1]
                differences.append(float((sampler(model,normalize(v.cuda(),norm),z)-sampler(model,normalize(w.cuda(),norm),z)).abs().max()))
        assert max(differences)==0
        return differences


def main():
    configure();check();OUT.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'PREFLIGHT.json').exists(),'Preflight evidence must not be overwritten'
    before=time.monotonic();model,norm=model_load();replay=[];fd=[];directions=[];inc=[]
    selected=[e for e in entries('train') if e['trajectory'] in [r['file'] for r in CFG['records']['train'][:2]] and e['start']==400 and e['repeat']==0]
    assert len(selected)==2
    labels=[]
    hook=model.net.register_forward_pre_hook(lambda module,args:labels.append(int(args[1][0])))
    for entry in selected:
        b=load_bundle(entry)
        for k in CFG['k']:
            r,m,s=inputs(b['history'],b['paths']['A'],oracle_prefix(entry['trajectory'],400,k),k)
            s=s.cuda();r=r.cuda();hn=normalize(s,norm);rn=normalize(r[None].expand(32,-1,-1,-1,-1),norm);z=noise_for(b,k,hn)
            with torch.no_grad():p=decode(sampler(model,hn,z),norm)
            err=float((p.cpu()-b['paths']['S'+str(k)]).abs().max())
            replay.append(dict(trajectory=entry['trajectory'],k=k,maxabs=err,passed=err<=CFG['replay_atol']))
            for member in CFG['preflight_members']:
                x=rn[member:member+1];e=hn[member:member+1]-x;e=e/rms(e);noise=z[member:member+1]
                fn=lambda v:sampler(model,v,noise)[:,:,:20-k]
                _,j=torch.autograd.functional.jvp(fn,x,e,strict=True)
                assert torch.isfinite(j).all() and rms(j)>0
                small=[]
                for eps in CFG['epsilon']:
                    with torch.no_grad():v=(fn(x+eps*e)-fn(x-eps*e))/(2*eps)
                    row=dict(trajectory=entry['trajectory'],k=k,member=member,epsilon=eps,**tangent_metrics(j[0],v[0],e[0],k))
                    fd.append(row);small.append(row)
                passed=any(v['relative_error']<=CFG['fd_relative_limit'] and v['cosine']>=CFG['fd_cosine_limit'] for v in small[1:]) and min(v['relative_error'] for v in small[1:])<=small[0]['relative_error']*1.25+.001
                directions.append(dict(trajectory=entry['trajectory'],k=k,member=member,passed=passed,jvp_rms=rms(j)))
                # Inspect actual first-layer increment channels and their directional derivative.
                captured=[]
                def capture(module,args):captured.append(args[0]);return None
                handle=model.net.first.register_forward_pre_hook(capture)
                xx=x.clone().requires_grad_(True);model.net(noise,torch.tensor([99],device='cuda'),xx);handle.remove()
                weight=torch.linspace(-1,1,40,device='cuda').reshape(1,40,1,1)
                grad=torch.autograd.grad((captured[0][:,80:]*weight).sum(),xx)[0]
                manual=torch.autograd.grad(((torch.cat([torch.zeros_like(xx[:,:,:1]),xx.diff(dim=2)],2)/model.net.scale).reshape(1,40,32,64)*weight).sum(),xx)[0]
                assert torch.equal(grad,manual) and rms(grad)>0
                inc.append(dict(trajectory=entry['trajectory'],k=k,member=member,gradient_rms=rms(grad),maxabs=float((grad-manual).abs().max())))
            print('PREFLIGHT',entry['trajectory'],k,'replay',err,'FD',directions[-2:],flush=True)
    hook.remove()
    # Every full sampler call must use99,66,33,0; direct increment checks add a99 separately.
    # A separate explicit trace isolates one full call for the order assertion.
    trace=[];hook=model.net.register_forward_pre_hook(lambda module,args:trace.append(int(args[1][0])))
    with torch.no_grad():sampler(model,hn[:1],z[:1])
    hook.remove();assert trace==[99,66,33,0]
    poison=poisoning_test(model,norm,b,10)
    std=norm['std'].cpu().reshape(2,1,1,1)
    rr,ee=response_rows(b['paths']['A'],b['paths']['B'],b['paths']['C'],std)
    ss=signed_rows(b['paths']['A'],b['paths']['C'],b['history'],std)
    assert max(abs(r['identity_error']) for r in rr+ss)<1e-12
    csvwrite(OUT/'preflight_replay.csv',replay);csvwrite(OUT/'preflight_fd.csv',fd);csvwrite(OUT/'preflight_directions.csv',directions);csvwrite(OUT/'preflight_increment_gradients.csv',inc)
    passed=all(v['passed'] for v in replay+directions)
    dump(OUT/'PREFLIGHT.json',dict(passed=passed,replay_passed=all(v['passed'] for v in replay),jvp_passed=all(v['passed'] for v in directions),directions=len(directions),finite_differences=len(fd),poisoning_maxabs=poison,full_labels=trace,increment_gradient_checked=True,identities_checked=True,seconds=time.monotonic()-before,config_sha256=digest(DELIVERY/'config.json'),code=code_hashes(),preflight_sources=selected,utc=time.time(),cost=dict(replay_paths=192,fd_paths=72,jvp_directions=12,jvp_reverse_sweeps=24,increment_net_calls=12,label_trace_paths=1,poisoning_paths=6)))
    print('PREFLIGHT_COMPLETE',passed,'seconds',time.monotonic()-before,flush=True)

if __name__=='__main__':main()
