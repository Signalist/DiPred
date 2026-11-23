"""Frozen forecast interventions and complete-path score diagnostics."""

import torch
from research.corrector_window_study.metrics import NAMES, scores
from research.corrector_window_study.model import empirical_crps

METRICS = (*NAMES, "increment5_crps")
MARGINAL = tuple(k for k in NAMES if k not in ("energy", "time_mean_crps"))


def evaluate(paths, target, scale):
    values, channels, leads = scores(paths, target, scale)
    separation = 5
    error = (paths - target[:, None]) / scale[:, None]
    increments = error[:, :, :, separation:] - error[:, :, :, :-separation]
    value = empirical_crps(increments, torch.zeros_like(increments[:, 0])).mean((1, 2, 3, 4))
    values["increment5_crps"] = value.cpu().tolist()
    values["crps_u"] = [v[0] for v in channels]
    values["crps_v"] = [v[1] for v in channels]
    for whole, center, spread in zip(values["member_mse"],values["mean_mse"],values["ensemble_variance"]):
        assert abs(whole - center - spread) <= 2e-7 * max(whole, 1e-5) + 2e-10
    return values


def centers(p, q):
    p, q = p.double(), q.double()
    p_mean, q_mean = p.mean(1, keepdim=True), q.mean(1, keepdim=True)
    p_d, q_d = p-p_mean, q-q_mean
    only_mean, only_center = q_mean+p_d, p_mean+q_d
    torch.testing.assert_close(only_mean+only_center-p, q, atol=1e-12, rtol=0)
    torch.testing.assert_close(only_center.mean(1, keepdim=True), p_mean, atol=1e-12, rtol=0)
    torch.testing.assert_close(only_mean.mean(1, keepdim=True), q_mean, atol=1e-12, rtol=0)
    return only_mean, only_center


def ordered(paths):
    return paths.sort(dim=1, stable=True)


def rank_interventions(p, q):
    p_sorted, p_idx = p
    q_sorted, q_idx = q
    q_native = torch.empty_like(q_sorted).scatter_(1, p_idx, q_sorted)
    p_corrector = torch.empty_like(p_sorted).scatter_(1, q_idx, p_sorted)
    assert torch.equal(q_native.sort(1).values, q_sorted)
    assert torch.equal(p_corrector.sort(1).values, p_sorted)
    return q_native, p_corrector


def compare_marginals(factual, permuted):
    max_error = 0.
    for metric in MARGINAL:
        for a,b in zip(factual[metric],permuted[metric]):
            error = abs(a-b)
            max_error = max(max_error,error)
            assert error <= 1e-7 + abs(a)*2e-7, (metric,a,b)
        # Canonicalize only permutation-invariant metrics after checking real scores.
        permuted[metric] = factual[metric]
    return max_error


def tie_rates(values):
    eq = values.diff(dim=1) == 0
    return eq.float().mean((1, 2, 3, 4, 5)).tolist()
