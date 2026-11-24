"""Physical residual decomposition and equal-original-trajectory weighting."""
from collections import defaultdict
from statistics import mean

import torch

from research.corrector_predictability.alpha import trajectory_indices


def weighted(values,cases):
    return torch.stack([values[ix].mean(0) for ix in trajectory_indices(cases).values()]).mean(0)


def errors(r,p,y,std):
    r,p,y = (z.to(torch.float64) for z in (r,p,y))
    truth = y-p
    q = p+r
    error = q-y
    native = truth.square()
    square = error.square()
    dot = r*truth
    energy = r.square()
    axes = (1,2,3,4)
    c,e,b,m = (v.mean(axes) for v in (dot,energy,native,square))
    g = 2*c-e
    identity_error = float((b-m-g).abs().max())
    assert identity_error<1e-15
    stats = dict(native_mse=b,mse=m,mae=error.abs().mean(axes),
                 standardized_mse=(error/std).square().mean(axes),
                 mse_u=square[:,0].mean((1,2,3)),mse_v=square[:,1].mean((1,2,3)),
                 C=c,E=e,G=g,cosine=c/(b*e).sqrt().clamp_min(1e-30))
    lead_axes = (1,3,4)
    lc,le,lb,lm = (v.mean(lead_axes) for v in (dot,energy,native,square))
    lg = 2*lc-le
    assert (lb-lm-lg).abs().max()<1e-15
    leads = dict(native_mse=lb,mse=lm,C=lc,E=le,G=lg)
    return stats,leads,identity_error


def correction_rate(row):
    return dict(row, correction_pct=100*(1-row["mse"]/row["native_mse"]))


def groups(rows,fields):
    result = defaultdict(list)
    for row in rows:
        result[tuple(row[k] for k in fields)].append(row)
    return result


def average(rows,keys,measurements):
    return [dict(zip(keys,k),**{m:mean(v[m] for v in cohort) for m in measurements})
            for k,cohort in groups(rows,keys).items()]
