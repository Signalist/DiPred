"""Supplementary negative control: preserve every marginal and break native time coupling."""
from common import *
from causal import native
from metrics import scores
from mechanisms import fit_linear,simple_predict,errors


def run():
    fit=fit_linear(residual('train'),'symmetric_linear3x3')
    rows=[]; linear=[]
    threshold=read(OUT/'causal_training_protocol.json')['event_threshold']
    for split in ('development','test'):
        fields=aux(split); issued=native(split)
        err=errors(residual(split),lambda x,l:simple_predict(x,l,fit))
        for i,r in enumerate(protocol()['records'][split]):
            linear.append(dict(split=split,trajectory=r['file'],symmetric_linear_mse=float(err[i].mean())))
            for w in range(2):
                p=issued[i,w]; y=fields['truth'][i,w]
                score0,_,_=scores(p,y,fields['normalization']['std'].squeeze(0),threshold)
                for seed in range(7100,7110):
                    rng=torch.Generator(device=p.device).manual_seed(seed+100*i+10000*w)
                    q=torch.stack([p[torch.randperm(len(p),device=p.device,generator=rng),:,t] for t in range(20)],2)
                    assert torch.equal(q.sort(0).values,p.sort(0).values)
                    s,_,_=scores(q,y,fields['normalization']['std'].squeeze(0),threshold)
                    assert abs(s['crps_standardized']-score0['crps_standardized'])<1e-12
                    rows.append(dict(split=split,trajectory=r['file'],window=w,seed=seed,**{
                        k:s[k]-score0[k] for k in ['crps_standardized','energy_standardized','temporal_acf_rmse','event_brier']}))
        print('COUPLING',split,flush=True)
    csvwrite(OUT/'coupling_permutation.csv',rows)
    csvwrite(OUT/'symmetric_linear_control.csv',linear)
    tr=aggregate(rows,['split','trajectory'],['crps_standardized','energy_standardized','temporal_acf_rmse','event_brier'])
    dump(OUT/'coupling_bootstrap.json',{k:bootstrap([r[k] for r in tr if r['split']=='test']) for k in
        ['crps_standardized','energy_standardized','temporal_acf_rmse','event_brier']})


if __name__=='__main__':
    from reliability.backbone import configure
    configure();run()
