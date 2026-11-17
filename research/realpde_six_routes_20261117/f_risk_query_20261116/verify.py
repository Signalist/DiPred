"""Independent NumPy score checks and actual frozen-model replay."""
from common import *
from risk import RiskHead,features,ARMS
from query import QueryHead,mask,inputs,target_queries,contain_pairs
import pandas as pd

def np_crps(q,y,t):
    e=y[None]-q;p=t.reshape(-1,*([1]*y.ndim));loss=np.maximum(p*e,(p-1)*e)
    return 2*(np.trapz(loss,t,axis=0)+t[0]*(loss[0]+np.maximum(q[0]-y,0))/2+(1-t[-1])*(loss[-1]+np.maximum(y-q[-1],0))/2)
@torch.no_grad()
def main():
    setup();audit={};n=norm('cpu');std=n['std'].flatten().numpy()
    old=json.loads((OUT/'data_audit.json').read_text())
    for p,h in old['sources'].items():assert digest(p)==h
    allmeta={s:load(OUT/'data'/f'{s}.pt')['meta'] for s in ('train','calibration','development')}
    for a in allmeta:
        for b in allmeta:
            if a!=b:assert {r['file'] for r in allmeta[a]}.isdisjoint({r['file'] for r in allmeta[b]})
    assert len(allmeta['train'])==819 and len(allmeta['development'])==108
    audit['splits']={s:dict(windows=len(rows),trajectories=len(set(r['file'] for r in rows)),groups=sorted(set(r['group'] for r in rows))) for s,rows in allmeta.items()}
    audit['source_hashes_unchanged']=True
    snapshots=load(OUT/'risk/main/snapshots.pt');maximum=0.
    for v in snapshots['snapshots']:
        qq=v['q'].numpy().astype('float64');y=v['y'].numpy();s=np_crps(qq,y,snapshots['levels'].numpy())*std[:,None,None,None]
        error=float(np.max(np.abs(s-v['crps'].numpy())));maximum=max(maximum,error);assert error<2e-6
    audit['numpy_risk_crps_max_error']=maximum
    d=cache('development');m=model();p,f=feature(m,d['x'][:12],'diffusion');error=float((p-d['p'][:12]).abs().max());assert error<2e-5;audit['actual_four_call_point_replay_max_error']=error
    data={k:(v[:2] if torch.is_tensor(v) else v) for k,v in d.items()};data['y']=torch.full_like(data['y'],float('nan'))
    zero=torch.zeros(2,48,32,64,device='cuda');hashes=[]
    for arm in ARMS:
        ck=load(OUT/'risk/main'/f'{arm}.pt');assert ck['seed']==42 and ck['steps']==2000;hashes.append((ck['initial_sha256'],ck['schedule_sha256']))
        h=RiskHead(ck['width']).cuda().eval();h.load_state_dict(ck['model']);ii=torch.arange(2,device='cuda');q=h.quantiles(*features(data,arm,ii,zero),torch.tensor([.05,.5,.95],device='cuda'))
        assert torch.equal(q[1],data['p']) and torch.isfinite(q).all();assert (q[1:]>=q[:-1]).all()
    assert len(set(hashes))==1;audit['all_risk_centers_identical_and_no_future_dependency']=True;audit['matched_initializations_and_training_schedules']=True
    qp=load(OUT/'query/predictions.pt');qrows=pd.read_csv(OUT/'query/per_query.csv');qmax=0.;bmax=0.
    for arm,v in qp['predictions'].items():
        q=v['quantiles'].numpy().astype('float64');y=qp['targets'].numpy();score=np_crps(q,y,qp['levels'].numpy())*np.tile(std,2)
        for stat in range(2):
            for c in range(2):
                expected=qrows[(qrows.method==arm)&(qrows.statistic==('mean','max')[stat])&(qrows.channel==c)].crps.to_numpy().reshape(108,160)
                er=float(np.max(np.abs(score[:,:,c+stat*2]-expected)));qmax=max(qmax,er);assert er<2e-6
        event=v['event'].numpy();truth=y[:,:,None,2:]>qp['thresholds'].numpy()[None,None]
        for c in range(2):
            block=qrows[(qrows.method==arm)&(qrows.statistic=='max')&(qrows.channel==c)]
            for t in range(3):
                er=float(np.max(np.abs((event[:,:,t,c]-truth[:,:,t,c])**2-block[f'brier{t}'].to_numpy().reshape(108,160))));bmax=max(bmax,er);assert er<1e-6
        assert (event[:,:,1:]<=event[:,:,:-1]+1e-6).all()
    audit['numpy_query_crps_max_error']=qmax;audit['numpy_query_brier_max_error']=bmax
    ep=load(OUT/'query/ensemble_predictions.pt');erows=pd.read_csv(OUT/'query/ensemble_per_query.csv');emax=0.
    for arm,v in ep.items():
        # Independent O(K^2) definition on a small aligned snapshot.
        s=v['samples'][:,:2,:3].numpy().astype('float64');y=qp['targets'][:2,:3].numpy();pair=np.abs(s[:,None]-s[None,:]).mean((0,1));score=(np.abs(s-y).mean(0)-.5*pair)*np.tile(std,2)
        for i in range(2):
            meta=qp['meta'][i]
            for j in range(3):
                for st in range(2):
                    for c in range(2):
                        row=erows[(erows.method==arm)&(erows.file==meta['file'])&(erows.start==meta['start'])&(erows['query']==j)&(erows.channel==c)&(erows.statistic==('mean','max')[st])]
                        er=abs(float(row.crps.iloc[0])-score[i,j,c+st*2]);emax=max(emax,er);assert er<2e-6
        aa,bb=zip(*contain_pairs(qp['queries']));event=v['event'];assert (event[:,aa]<=event[:,bb]+1e-6).all()
    audit['numpy_pairwise_ensemble_crps_max_error']=float(emax);audit['ensemble_inclusion_consistent']=True
    # Query aggregate matches a direct field slice; corrupt future cannot affect query inputs.
    q=qp['queries'][7];a,b,c,e=q['rect'];t,u=q['time'];truth=d['y'][:2,:,t:u,a:b,c:e];targets=target_queries(d['y'][:2],[q])[:,0]
    assert torch.allclose(targets[:,:2],truth.mean((2,3,4))) and torch.equal(targets[:,2:],truth.amax((2,3,4)))
    z1=inputs(data,[q]);data['y']=torch.randn_like(data['y']);z2=inputs(data,[q]);assert torch.equal(z1,z2);audit['query_labels_and_no_future_dependency']=True
    for ck in list((OUT/'risk').glob('*/*.pt'))+list((OUT/'query').glob('*.pt')):
        if ck.name in ('progress.pt','snapshots.pt') or ck.parent.name=='query' and ck.stem not in ('X','P','F','XPF'):continue
        value=load(ck);assert value['seed']==42
    symmetric=load(OUT/'risk/symmetric/F.pt')
    assert symmetric['symmetric'] is True
    h=RiskHead(symmetric['width'],True).cuda().eval();h.load_state_dict(symmetric['model'])
    q=h.quantiles(*features(data,'F',torch.arange(2,device='cuda')),torch.arange(1,100,device='cuda')/100)
    err=(q+q.flip(0)-2*data['p'][None]).abs();symmetry_error=float(err.max())
    scale=q.abs()+q.flip(0).abs()+2*data['p'][None].abs()+1
    # Complementary float32 probabilities differ by up to one ULP; logit amplifies tails.
    assert (err<=8*torch.finfo(torch.float32).eps*scale).all()
    t=torch.arange(1,100,device='cuda',dtype=torch.float64)/100
    w=h.weights(*features(data,'F',torch.arange(2,device='cuda'))).double()
    q64=data['p'][None].double()+torch.einsum('bckthw,qk->qbcthw',w,.5*(bases(t)-bases(1-t)))
    error64=float((q64+q64.flip(0)-2*data['p'][None].double()).abs().max());assert error64<2e-13
    audit['symmetric_float64_construction_error']=error64
    audit['symmetric_float32_relative_epsilon_error']=float((err/scale).max()/torch.finfo(torch.float32).eps)
    audit['symmetric_mean_and_median_control_max_error']=symmetry_error
    qs=load(OUT/'query/train.pt')['queries'];keys=lambda values:{tuple(q['rect']+q['time']) for q in values}
    assert len(keys(qs))==48
    assert keys(qs).isdisjoint(keys([q for q in qp['queries'] if q['category']!='seen']))
    audit['heldout_query_geometries_disjoint']=True
    qtrain=load(OUT/'query/train.pt')['y'].numpy()[:,:,2:].reshape(-1,2)
    expected=np.quantile(qtrain,[.5,.75,.9],axis=0);actual=load(OUT/'query/statistics.pt')['thresholds'].numpy()
    assert np.allclose(expected,actual,atol=1e-6);audit['event_thresholds_recomputed_from_train_only']=True
    audit['all_new_training_single_seed_42']=True
    audit['passed']=True;dump(OUT/'verification.json',audit);print(json.dumps(audit,indent=2),flush=True)
if __name__=='__main__':main()
