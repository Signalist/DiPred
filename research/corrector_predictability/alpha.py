"""Analytic shrinkage fitted to development and separately labeled test oracles."""
from collections import defaultdict

import numpy as np
import torch


def moments(prediction, truth, axes=(1, 2, 3, 4)):
    r, t = prediction.double(), truth.double()
    return (r*t).mean(axes), r.square().mean(axes), t.square().mean(axes)


def optimum(c, e, bounded=True):
    value = float(c / e) if e else 0.
    return min(1., max(0., value)) if bounded else value


def fit_alpha(pred, truth):
    c, e, _ = moments(pred, truth)
    return optimum(c.mean(), e.mean())


def perturbation(c, e, alpha):
    return alpha*alpha*e - 2*alpha*c


def grid(pred, truth, arm):
    c, e, p = (float(i.mean()) for i in moments(pred, truth))
    chosen = optimum(c, e)
    return [dict(arm=arm, mode=mode, alpha=a, native_mse=p,
                 mse=p+perturbation(c,e,a), improvement_pct=-100*perturbation(c,e,a)/p,
                 inner=c, energy=e, twice_inner=2*c)
            for mode, a in [("development_fit", chosen)] +
            [("fixed", x) for x in (0., .05, .1, .2, .5, 1.)]]


def oracles(pred, truth, cases, arm):
    """Read true targets only after the study freeze and report each upper bound."""
    c, e, p = moments(pred, truth)
    oracle = []
    for policy, groups in (
        ("global", [("all", c, e, p)]),
        ("lead", [(str(i+1), cv[:,i], ev[:,i], pv[:,i]) for cv,ev,pv in [moments(pred,truth,(1,3,4))]
                  for i in range(20)]),
        ("channel", [(name, cv[:,i], ev[:,i], pv[:,i]) for cv,ev,pv in [moments(pred,truth,(2,3,4))]
                     for i,name in enumerate(("u","v"))]),
        ("trajectory", [(name, c[ix], e[ix], p[ix]) for name,ix in trajectory_indices(cases).items()]),
        ("window", [(f'{r["file"]}/{r["start"]}', c[i:i+1], e[i:i+1], p[i:i+1])
                    for i,r in enumerate(cases)]),
    ):
        for bound in (True, False):
            cohort = []
            for group, cs, es, ps in groups:
                a, b, base = (float(x.mean()) for x in (cs, es, ps))
                scale = optimum(a, b, bound)
                mse = base+perturbation(a,b,scale)
                cohort.append(dict(arm=arm, scope=policy, group=group, bound="0..1" if bound else "unrestricted",
                                   alpha=scale, native_mse=base, mse=mse, improvement_pct=100*(1-mse/base),
                                   inner=a, energy=b, twice_inner=2*a, case_count=len(cs)))
            oracle.extend(cohort)
            if len(cohort) > 1:
                base = np.average([r["native_mse"] for r in cohort], weights=[r["case_count"] for r in cohort])
                mse = np.average([r["mse"] for r in cohort], weights=[r["case_count"] for r in cohort])
                oracle.append(dict(arm=arm, scope=policy, group="ALL", bound=cohort[0]["bound"],
                                   alpha="", native_mse=base, mse=mse, improvement_pct=100*(1-mse/base),
                                   inner="", energy="", twice_inner="", case_count=len(cases)))
    return oracle


def trajectory_indices(cases):
    tracks = defaultdict(list)
    for i, c in enumerate(cases):
        tracks[c["file"]].append(i)
    return dict(sorted(tracks.items()))
