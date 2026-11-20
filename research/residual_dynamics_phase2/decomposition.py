"""Separate common forecast bias from native member deviations; no deployment claim."""
from common import *
from mechanisms import fit_linear,simple_predict,errors


def corr(x,y):
    a=x.movedim(-3,0).reshape(2,-1).double()
    b=y.movedim(-3,0).reshape(2,-1).double()
    a=a-a.mean(1,keepdim=True); b=b-b.mean(1,keepdim=True)
    return ((a*b).sum(1)/(a.square().sum(1)*b.square().sum(1)).sqrt().clamp_min(1e-30)).cpu().numpy()


def offset_pair(x,y,dy,dx):
    h,w=x.shape[-2:]
    ya,yb=max(dy,0),min(h+dy,h)
    xa,xb=max(dx,0),min(w+dx,w)
    return x[...,ya:yb,xa:xb],y[...,ya-dy:yb-dy,xa-dx:xb-dx]


@torch.no_grad()
def run():
    assert read(OUT/'alignment_audit.json')['passed']
    statistics=[];energy=[];predictions=[];models={};shuffle=[]
    for split in ('train','development','test'):
        e=residual(split)
        # Cached subtraction is float32; numerical reconstruction audited in
        # float64 using original native P and original raw Y in audit.py.
        b,d=components(e)
        maxerr=float((e-(b+d)).abs().max())
        assert maxerr<1e-7
        for name,data in [('bias',b),('delta',d)]:
            if split=='train':
                models[name]={k:fit_linear(data,k) for k in
                              ('A2_scaled','A3_linear','symmetric_linear3x3','linear3x3')}
            for i,record in enumerate(protocol()['records'][split]):
                v=data[i]
                power=v.double().square().mean((0,1,2,4,5)).cpu().numpy()
                for c,ch in enumerate(('u','v')):
                    energy.append(dict(split=split,trajectory=record['file'],Re=record['group'],
                        component=name,channel=ch,mean_square=float(power[c]),reconstruction_max_error=maxerr))
                for lag in (1,2,3,4,5,10,19):
                    value=corr(v[:,:,:-lag],v[:,:,lag:])
                    for c,ch in enumerate(('u','v')):
                        statistics.append(dict(split=split,trajectory=record['file'],Re=record['group'],
                            component=name,channel=ch,statistic='temporal',lag=lag,dy=0,dx=0,correlation=float(value[c])))
                for radius in (1,2,3,4,6,8):
                    for dy,dx in [(0,radius),(radius,0)]:
                        a,z=offset_pair(v,v,dy,dx);value=corr(a,z)
                        for c,ch in enumerate(('u','v')):
                            statistics.append(dict(split=split,trajectory=record['file'],Re=record['group'],
                                component=name,channel=ch,statistic='spatial',lag=0,dy=dy,dx=dx,correlation=float(value[c])))
                for dy in (-2,-1,0,1,2):
                    for dx in (-2,-1,0,1,2):
                        a,z=offset_pair(v[:,:,:-1],v[:,:,1:],dy,dx);value=corr(a,z)
                        for c,ch in enumerate(('u','v')):
                            statistics.append(dict(split=split,trajectory=record['file'],Re=record['group'],
                                component=name,channel=ch,statistic='lag1_offset',lag=1,dy=dy,dx=dx,correlation=float(value[c])))
            if split!='train':
                methods={'A1_persistence':dict(kind='A1_persistence'),**models[name]}
                for method,ck in methods.items():
                    values=errors(data,lambda x,l,ck=ck:simple_predict(x,l,ck))
                    for i,r in enumerate(protocol()['records'][split]):
                        for t in range(19):
                            for c,ch in enumerate(('u','v')):
                                predictions.append(dict(split=split,trajectory=r['file'],Re=r['group'],
                                    component=name,method=method,channel=ch,lead=t+2,mse=float(values[i,t,c])))
                baseline=errors(data,lambda x,l:x)
                shuffled=[errors(data,lambda x,l:x,shuffle_seed=seed) for seed in range(7100,7110)]
                if name=='bias':
                    assert all(np.array_equal(baseline,z) for z in shuffled)
                    shuffled=baseline.copy()
                else:shuffled=np.mean(shuffled,0)
                for i,r in enumerate(protocol()['records'][split]):
                    shuffle.append(dict(split=split,trajectory=r['file'],Re=r['group'],component=name,
                        aligned_mse=float(baseline[i].mean()),shuffled_mse=float(shuffled[i].mean())))
            print('DECOMPOSITION',split,name,flush=True)
        del e,b,d
    csvwrite(OUT/'component_energy.csv',energy)
    csvwrite(OUT/'component_correlations.csv',statistics)
    csvwrite(OUT/'component_correlations_by_Re.csv',aggregate(statistics,
        ['split','Re','component','channel','statistic','lag','dy','dx'],['correlation']))
    csvwrite(OUT/'component_prediction_by_trajectory_lead_channel.csv',predictions)
    csvwrite(OUT/'component_prediction_summary.csv',aggregate(predictions,['split','component','method'],['mse']))
    csvwrite(OUT/'component_member_shuffle.csv',shuffle)
    torch.save(models,EXTRA/'component_linear.pt')
    lengths=[]
    for r in aggregate(statistics,['split','Re','component','channel','statistic','lag','dy','dx'],['correlation']):
        if r['statistic']=='spatial':
            lengths.append(r)
    groups={}
    for r in lengths:
        key=tuple(r[k] for k in ('split','Re','component','channel'))+('x' if r['dx'] else 'y',)
        groups.setdefault(key,[]).append(r)
    estimates=[]
    for key,rows in groups.items():
        reached=[max(r['dx'],r['dy']) for r in rows if r['correlation']<=np.exp(-1)]
        estimates.append(dict(zip(['split','Re','component','channel','axis'],key),
            first_observed_radius_below_1_over_e=min(reached) if reached else None,
            right_censored_above8=not bool(reached),units='grid cells; sampled radii1,2,3,4,6,8'))
    dump(OUT/'correlation_lengths.json',estimates)
    grouped=aggregate(predictions,['split','component','method','trajectory'],['mse'])
    lookup={(r['split'],r['component'],r['method'],r['trajectory']):r['mse'] for r in grouped}
    comparisons=[]
    for split in ('development','test'):
        for comp in ('bias','delta'):
            for a,b in [('A2_scaled','A1_persistence'),('linear3x3','symmetric_linear3x3')]:
                diff=[lookup[split,comp,a,r['file']]-lookup[split,comp,b,r['file']] for r in protocol()['records'][split]]
                comparisons.append(dict(split=split,component=comp,contrast=a+' minus '+b,**bootstrap(diff)))
    dump(OUT/'component_bootstrap.json',comparisons)
    dump(OUT/'component_information_boundary.json',dict(
        bias='P ensemble mean minus Y; future bias is an offline label, never a causal input',
        delta='issued native P minus ensemble mean; already available, prediction is diagnostic only',
        correlation='centered Pearson pooled over window/member/time/space per channel; spatial nonstationarity remains',
        shuffle='10 permutations of previous whole member fields, each window/time separately; target stays unchanged',
        linear_parameters=dict(A2_scaled=2,A3_linear=4,symmetric_linear3x3=12,linear3x3=36),
        coefficients_fitted_on='train only; sampled262144 train points except exact A2 all-data moments'))


if __name__=='__main__':
    import sys
    sys.path.insert(0,str(REPO))
    from reliability.backbone import configure
    configure();run()
