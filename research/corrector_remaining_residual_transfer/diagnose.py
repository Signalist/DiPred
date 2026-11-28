"""Fold-out, track-weighted cross-window associations on physical residuals."""
from collections import defaultdict

import torch

from research.corrector_residual_optimization.fit import track_weights
from research.corrector_simple_baseline.fit import table


def correlations(x, y, weights):
    """Use samples × grid × leads arrays to form all lead-pair moments."""
    a, b = (v.reshape(-1, v.shape[-1]) for v in (x,y))
    w = weights[:,None].expand(x.shape[:2]).reshape(-1,1)/x.shape[1]
    mean_a, mean_b = (v.T @ w for v in (a,b))
    square_a, square_b = (v.square().T @ w for v in (a,b))
    inner = a.T @ (b*w)
    means = mean_a @ mean_b.T
    variance_a = (square_a-mean_a.square()).clamp_min(0)
    variance_b = (square_b-mean_b.square()).clamp_min(0)
    pearson = (inner-means)/(variance_a @ variance_b.T).sqrt().clamp_min(1e-30)
    cosine = inner/(square_a @ square_b.T).sqrt().clamp_min(1e-30)
    assert all(torch.isfinite(v).all() for v in (inner,pearson,cosine))
    return torch.stack((inner,pearson,cosine),dim=-1).tolist()


@torch.inference_mode()
def stage_zero(out, view, basis):
    basis = basis.to(view.target)
    previous = view.historical.double()
    remaining = view.true-previous
    fields = dict(R=view.true,P=previous,E=remaining,Rn=view.target,En=view.target-view.current.double(),
                  S3=view.current.double(),observed=view.data.x.double(),native=view.data.p.double(),
                  last=view.data.x[:,:,-1:].double().expand_as(view.target))
    pairs = (("R","Rn"),("R","En"),("P","Rn"),("P","En"),("E","Rn"),("E","En"),
             ("E","observed"),("E","native"),("E","last"),("E","S3"))
    groups = defaultdict(list)
    for i, case in enumerate(view.data.cases):
        groups[str(case["re"])].append(i)
    groups["all"] = list(range(len(view.data.cases)))
    weights = track_weights(view.data)
    times, spatial, global_rows, modes, energies = [], [], [], [], []
    for re, ids in groups.items():
        w = weights[ids]
        w /= w.sum()
        for channel,name in enumerate(("u","v")):
            for first,second in pairs:
                x,y = (fields[k][ids,channel] for k in (first,second))
                identity = dict(source=first,target=second,channel=name,re=re)
                a,b = (v.flatten(-2).transpose(1,2).contiguous() for v in (x,y))
                for mode, arrays, rows in ((False,(a,b),times),
                                          (True,(a@basis,b@basis),modes)) if re=="all" else ((False,(a,b),times),):
                    m = correlations(*arrays,w)
                    for i,line in enumerate(m,1):
                        for j,(inner,pearson,cosine) in enumerate(line,1):
                            point = dict(source_index=i,target_index=j)
                            if not mode:
                                point.update(lag=20+j-i,same_lead=i==j,boundary=i==20 and j==1)
                            rows.append(dict(identity,**point,inner=inner,pearson=pearson,cosine=cosine))
                w4 = (w/20)[:,None,None,None]
                means = [(v*w4).sum((0,1)) for v in (x,y)]
                squares = [(v.square()*w4).sum((0,1)) for v in (x,y)]
                inner = (x*y*w4).sum((0,1))
                variance = [(q-m.square()).clamp_min(0) for q,m in zip(squares,means)]
                corr = (inner-means[0]*means[1])/(variance[0]*variance[1]).sqrt().clamp_min(1e-30)
                cosine = inner/(squares[0]*squares[1]).sqrt().clamp_min(1e-30)
                if re=="all":
                    for i,line in enumerate(torch.stack((inner,corr,cosine),dim=-1).tolist()):
                        for j,(c,p,d) in enumerate(line):
                            spatial.append(dict(identity,y=i,x=j,inner=c,pearson=p,cosine=d))
                c = inner.mean()
                ma,mb = [v.mean() for v in means]
                qa,qb = [v.mean() for v in squares]
                global_rows.append(dict(identity,inner=float(c),pearson=float(
                    (c-ma*mb)/((qa-ma.square())*(qb-mb.square())).clamp_min(1e-30).sqrt()),
                    cosine=float(c/(qa*qb).clamp_min(1e-30).sqrt()),source_rms=float(qa.sqrt()),target_rms=float(qb.sqrt())))
                u,v = [z.square().mean((1,2,3)) for z in (x,y)]
                for label,start,end in (("energy",u,v),("rms",u.sqrt(),v.sqrt())):
                    xu=start-(w*start).sum()
                    yu=end-(w*end).sum()
                    dot=(xu*yu*w).sum()
                    denominator=((xu.square()*w).sum()*(yu.square()*w).sum()).sqrt().clamp_min(1e-30)
                    energies.append(dict(identity,quantity=label,source_mean=float((w*start).sum()),
                                         target_mean=float((w*end).sum()),pearson=float(dot/denominator)))
    for name,rows in (("residual_transfer_correlations",global_rows),("temporal_correlations",times),
                      ("spatial_correlations",spatial),("transfer_modes",modes),("amplitude_transfer",energies)):
        table(out/f"{name}.csv",rows)
