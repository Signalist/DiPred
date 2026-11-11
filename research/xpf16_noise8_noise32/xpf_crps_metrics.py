"""Marginal CRPS: explicit complete distributions, never intervals alone."""

import math
import torch


Z95 = 1.6448536269514722


def gaussian_crps(observation, location, scale):
    scale = scale.clamp_min(1e-10)
    z = (observation - location) / scale
    return scale * (z * torch.erf(z / math.sqrt(2))
                    + math.sqrt(2 / math.pi) * torch.exp(-.5 * z.square())
                    - 1 / math.sqrt(math.pi))


def laplace_crps(observation, location, scale):
    scale = scale.clamp_min(1e-10)
    distance = (observation - location).abs()
    return distance + scale * torch.exp(-distance / scale) - .75 * scale


def interval_completions(lower, upper, observation):
    """Diagnostic distributions whose 5/95% quantiles equal issued bounds.

    Both are assumptions imposed for scoring, not distributions originally
    learned by the two-quantile head. Use both to expose sensitivity.
    """
    midpoint = .5 * (lower + upper)
    radius = (.5 * (upper - lower)).clamp_min(1e-10)
    return dict(gaussian=gaussian_crps(observation, midpoint, radius / Z95),
                laplace=laplace_crps(observation, midpoint, radius / math.log(10)))


def ensemble_crps(samples, observation):
    """Exact empirical-CDF CRPS and iid-member fair population estimator.

    Sorted coefficients avoid a quadratic pairwise tensor. These two estimands
    differ at finite K; fair is not called the empirical distribution's CRPS.
    """
    count = len(samples)
    if count < 2:
        raise ValueError('At least two members are needed for fair CRPS.')
    ordered = samples.sort(dim=0).values
    weights = 2 * torch.arange(count, device=samples.device, dtype=samples.dtype) - count + 1
    weights = weights.reshape(count, *([1] * observation.ndim))
    half_pair_sum = (ordered * weights).sum(dim=0)
    absolute = (samples - observation[None]).abs().mean(dim=0)
    return dict(empirical=absolute - half_pair_sum / (count * count),
                fair=absolute - half_pair_sum / (count * (count - 1)))
