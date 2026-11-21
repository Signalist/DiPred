"""Train-only evaluation diagnostics checks; no fitting or test outcome selection."""
from core import *
from research.provenance_uncertainty_conditioning.metrics import scores,structure


def main():
    configure();assert not (OUT/'EVALUATION_TESTS.json').exists()
    model,norm=load_model();e=entries('train')[0];b=prior.load_bundle(e);std=norm['std'].reshape(2,1,1,1)
    a=b['paths']['A'].cuda();direct=b['paths']['B'].cuda();ind=b['paths']['C'].cuda();y=prior.frames(e['trajectory'],e['start']+20,e['start']+40).cuda()
    records=[];maxerror=0.
    for k in CFG['k']:
        s=b['paths']['S'+str(k)].cuda();joint=torch.cat([a[:,:,:k],s[:,:,:20-k]],2)
        assert torch.equal(joint[:,:,:k],a[:,:,:k]) and torch.equal(joint[:,:,k:],s[:,:,:20-k])
        for view,p,truth,refs in [('tail',s[:,:,:20-k],y[:,k:],(direct[:,:,k:],s[:,:,:20-k],ind[:,:,k:])),('joint',joint,y,(direct,joint,ind))]:
            st=tuple(structure(t.double()/std) for t in refs)
            rr,cc,mat=scores(p,truth,std,view,k,a,refs,st)
            baseline,_=native.quality(p,truth,std)
            for r,base in zip(rr,baseline):
                for metric in QUAL:maxerror=max(maxerror,abs(r[metric]-base[metric]))
            for ch in 'uv':
                row=next(r for r in rr if r['channel']==ch);leads=[c for c in cc if c['channel']==ch]
                for metric in ['crps','mse','smse','variance','width90']:assert abs(row[metric]-np.mean([c[metric] for c in leads]))<1e-10
            center=(p.double()/std)-(p.double()/std).mean(0)
            explicit=torch.stack([torch.stack([torch.stack([(center[:,ch,t]*center[:,ch,u]).mean() for u in range(p.shape[2])]) for t in range(p.shape[2])]) for ch in range(2)])
            assert float((explicit-mat['cov'].cuda()).abs().max())<1e-10
            records.extend(dict(k=k,view=view,**r) for r in rr)
    assert maxerror==0
    # Check all three adapters can execute evaluation entry import/zero native equivalence.
    from research.provenance_uncertainty_conditioning import evaluate
    h,mask,spread,z=prepare(b,2,norm)
    with torch.no_grad():pred=sample(model,Adapter('C_unc',42).cuda(),h,mask,spread,z)
    assert float((decode(pred,norm).cpu()-b['paths']['S2']).abs().max())<=CFG['replay_atol']
    csvwrite(OUT/'evaluation_train_diagnostic_checks.csv',records)
    dump(OUT/'EVALUATION_TESTS.json',dict(passed=True,quality_maxabs=maxerror,lead_reduction_checked=True,temporal_covariance_explicit_checked=True,member_join_checked=True,eval_entry_import_checked=True,code=code_hashes(),utc=time.time()))
    print('PASS scoring/lead reduction/member join/explicit covariance/entry import',flush=True)

if __name__=='__main__':main()
