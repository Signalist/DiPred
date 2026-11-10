"""Marginal error-density losses sharing the existing ordered90% bounds.

The midpoint is an error-distribution location, not a correction to the fixed
DiPred point forecast. These losses do not model joint spatiotemporal density.
"""

import math

import torch
from scipy.stats import norm,t


Q95 = {"density_gaussian":float(norm.ppf(.95)),"density_student4":float(t.ppf(.95,4))}


def density_loss(lower,upper,residual,kind):
    midpoint = .5*(lower+upper)
    radius = (.5*(upper-lower)).clamp_min(1e-4)
    scale = radius/Q95[kind]
    z = (residual-midpoint)/scale
    if kind=="density_gaussian":
        return (scale.log()+.5*z.square()+.5*math.log(2*math.pi)).mean()
    if kind!="density_student4":
        raise ValueError("Choose Gaussian or fixeddf4 Student-t.")
    df = 4
    constant = .5*math.log(df*math.pi)+math.lgamma(df/2)-math.lgamma((df+1)/2)
    return (scale.log()+.5*(df+1)*torch.log1p(z.square()/df)+constant).mean()
