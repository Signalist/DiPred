"""Requirement-specific audit of empirical positive benefit, not paper novelty."""
from physics import *
import pandas as pd
from common import DATA,records,read
SOURCE=Path(__file__).resolve().parent

@torch.no_grad()
def main():
    setup();seed();checks={}
    attachment=W.parent.parent/'.app/.codex-runtime/home/attachments/15f47df4-35f6-4993-8615-e5e9164b56b4/pasted-text-1.txt'
    assert attachment.exists();checks['reference_sha256']=digest(attachment)
    assert str(DATA).endswith('OLD/REALPDE/realpdebench-competition/data/train_real')
    checks['dataset']=str(DATA);checks['five_pde_used']=False
    manifests=json.loads(MANIFEST.read_text())['files']
    for a in ('train','calibration','development'):
        for b in manifests:
            if a!=b:
                assert {v['file'] for v in manifests[a]}.isdisjoint({v['file'] for v in manifests[b]})
                assert {v['group'] for v in manifests[a]}.isdisjoint({v['group'] for v in manifests[b]})
    libaudit=json.loads((BASE/'library_audit.json').read_text());assert libaudit['checkpoint_sha256']==digest(SHARED)
    heads={}
    for k in range(1,50):
        path=W/'runs/xpf16_fullpercent_20261111/models'/f'XPF16-Q{k:02d}'/'model.pt';ck=load(path)
        assert ck['config']['seed']==42 and ck['config']['k']==k and ck['parameters']==34004
        heads[path.parent.name]=digest(path)
    checks['training_seed']=42;checks['frozen_head_hashes']=heads;checks['checkpoint_sha256']=digest(SHARED)
    checks['split_isolation']='original trajectory and Re partitions disjoint; later windows share development identities'
    # Direct raw-data replay checks the provenance of the reused normalized cache.
    n=norm('cpu');data_errors=[]
    for split in ('train','calibration','development'):
        d=load(W/'runs/f_risk_query_20261116/data'/f'{split}.pt')
        rec={r['file']:r for r in records(split)}
        for i in (0,len(d['meta'])-1):
            m=d['meta'][i];raw=read(rec[m['file']],m['start'])
            x=(raw[:,:20]-n['mean'][0])/n['std'][0];y=(raw[:,20:]-n['mean'][0])/n['std'][0]
            error=max(float((x-d['x'][i]).abs().max()),float((y-d['y'][i]).abs().max()));assert error<1e-6
            data_errors.append(dict(split=split,index=i,max_error=error))
    checks['raw_cache_replay']=data_errors
    audits={}
    for stage in ('','sensitivity','localized','physics','advection','helmholtz_filter'):
        p=BASE/stage/'audit.json';v=json.loads(p.read_text());assert v['passed'];assert v['numerical_max_error']<3e-5;audits[stage or 'transport']=v['numerical_max_error']
    checks['independent_audits_max_error']=audits
    protocol=json.loads((BASE/'future_protocol.json').read_text())
    for stage,h in protocol['selection_hashes'].items():assert digest(BASE/stage/'selection.json')==h
    expected={'calibration':90,'development':108,'late':120,'future':72};evidence=[]
    for split,count in expected.items():
        path=BASE/'physics'/split;parts=sorted((path/'parts').glob('*.pt'));assert len(parts)==count
        for p in parts:
            part=load(p);assert max(part['marginal_error'].values())==0
        d=pd.read_csv(path/'summary.csv').set_index('method');b=d.loc['ecc32'];a=d.loc['smooth_ecc_b50_i1']
        # Positive evidence is for the fixed joint correction, never universal superiority.
        assert a.es<b.es and a.vs<b.vs and a.functional_crps<b.functional_crps
        if split!='calibration':
            paired=pd.read_csv(path/'paired.csv');s=paired[(paired.method=='smooth_ecc_b50_i1')&(paired.baseline=='ecc32')].set_index('metric')
            assert s.loc['crps_speed_max','ci_low']>0
            groups=pd.read_csv(path/'per_re.csv').set_index(['method','group']);assert (groups.loc['smooth_ecc_b50_i1','crps_speed_max']<groups.loc['ecc32','crps_speed_max']).all()
        evidence.append(dict(split=split,windows=count,es_gain_percent=100*(b.es-a.es)/b.es,vs_gain_percent=100*(b.vs-a.vs)/b.vs,functional_gain_percent=100*(b.functional_crps-a.functional_crps)/b.functional_crps,speed_gain_percent=100*(b.crps_speed_max-a.crps_speed_max)/b.crps_speed_max,mvpe_gain_percent=100*(b.crps_mvpe-a.crps_mvpe)/b.crps_mvpe))
    checks['positive_evidence']=evidence
    infer=json.loads((BASE/'inference_audit.json').read_text());assert max(infer['replay_max_error'].values())<2e-5;checks['inference']=infer
    # Algebraic audit of the nonperiodic divergence/adjoint/projection, independently
    # from the scoring assertions. This does not turn failed physics models positive.
    a=torch.randn(2,2,20,32,64,device='cuda');q=torch.randn(2,20,30,62,device='cuda');h=Hodge()
    lhs=(divergence(a)*q).sum();rhs=(a*adjoint(q)).sum();ae=float((lhs-rhs).abs()/lhs.abs().clamp_min(1e-8));b=h(a)
    de=float(divergence(b).square().mean().sqrt()/divergence(a).square().mean().sqrt());ie=float((h(b)-b).abs().max());assert ae<1e-5 and de<1e-5 and ie<1e-5
    yy,xx=torch.meshgrid(torch.arange(32,device='cuda'),torch.arange(64,device='cuda'),indexing='ij');linear=torch.stack((xx*DX,-yy*DX))[None,:,None].expand(1,2,20,32,64)
    ge=float((divergence(linear)/DX-2).abs().max());assert ge<1e-5
    mechanism=dict(adjoint_relative_error=ae,hodge_divergence_ratio=de,idempotence_max_error=ie,physical_coordinate_linear_field_error=ge)
    dump(BASE/'mechanism_audit.json',mechanism);checks['mechanism']=mechanism
    for f in ('REPORT.md','positive_gains.csv','positive_paired.csv','route_outcomes.csv','figures/gains.png','figures/gains.pdf','figures/tradeoffs.png'):
        assert (BASE/f).exists() and (BASE/f).stat().st_size>0
    checks['source_hashes']={p.name:digest(p) for p in SOURCE.glob('*.py')}
    dump(BASE/'completion_audit.json',dict(passed=True,scope='RealPDE small-data exploration until a verified positive joint-distribution benefit; not proof of ICML novelty or all-route failure',checks=checks,limits=['same development trajectories used repeatedly','MVPE and some TKE/group regressions retained','physics and sensitivity mechanisms have not established overall unique superiority','DLL not reproduced','no claim of independent new-domain generalization']))
    print('COMPLETION_AUDIT_PASSED',json.dumps(evidence,indent=2),flush=True)

if __name__=='__main__':main()
