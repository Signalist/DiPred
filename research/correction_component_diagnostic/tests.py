"""Small synthetic checks that run before any real path scoring."""

from itertools import combinations
from types import SimpleNamespace

import torch
from torch import nn

from research.residual_uncertainty_diffusion.core import CorrectedDiffusion, crps
from research.trust_causal_audit_v2.model import native_sample
from .sampler import LABELS, sample
from .tables import (
    AMPLITUDE, COMPONENT, CONDITION, DECOMPOSITION,
    decompose, fd_status, save_scores, score_rows,
)


class ToyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.up2 = nn.Identity()

    def forward(self, x, label, history):
        return self.up2(0.3 * x + history)


def toy_correct(x, history, e, w, w_mean, feature, label):
    return 0.04 + x * 0.13 + feature.square() * 0.02


def subset_test():
    p = torch.tensor([-1.91, -0.74, 0.14, 0.66, 1.89], dtype=torch.float64)
    truth = torch.tensor(0.21, dtype=torch.float64)
    full = p.clone().requires_grad_()
    subset = p.clone().requires_grad_()
    m = 3
    M = len(p)
    values = []
    for indices in combinations(range(M), m):
        selected = subset[list(indices)]
        first = (selected - truth).abs().mean()
        half = (selected[:, None] - selected[None, :]).abs().sum() / (2 * m * (m - 1))
        values.append(first - ((M - 1) / M) * half)
    average = torch.stack(values).mean()
    total = crps(full, truth)
    torch.testing.assert_close(average, total, atol=1e-14, rtol=0)
    average.backward()
    total.backward()
    torch.testing.assert_close(subset.grad, full.grad, atol=1e-14, rtol=0)
    return dict(M=M, m=m, subsets=len(values), value_error=float((average - total).abs()),
                gradient_maxabs=float((subset.grad - full.grad).abs().max()), coefficient_M32_m8=31 / 32)


def sampler_test():
    native = SimpleNamespace(net=ToyNet(), abar=torch.tensor([.9, .65, .3, 0.], dtype=torch.float64))
    z = torch.tensor([-.5, .1, .7, 1.5], dtype=torch.float64)
    inputs = dict(history=torch.full_like(z, .1), noise=z, E=z, W=z, W_mean=z)
    truth = torch.tensor(-.71, dtype=torch.float64)
    evidence = []
    for endpoint in (0., 1.):
        amplitude = torch.full((4,), endpoint, dtype=z.dtype, requires_grad=True)
        seen = []
        paths = sample(native, inputs, toy_correct, amplitude, lambda label, *args: seen.append(label))
        assert tuple(seen) == LABELS
        for fair in (False, True):
            loss = crps(paths, truth, fair)
            loss.backward(retain_graph=not fair)
            gradients = amplitude.grad.clone()
            assert (gradients.abs() > 1e-6).all()
            amplitude.grad.zero_()
            directions = torch.cat((torch.eye(4, dtype=z.dtype), torch.ones((1, 4), dtype=z.dtype)))
            slopes = {}
            with torch.no_grad():
                for h in (0.01, 0.005):
                    slopes[h] = torch.stack([
                        (crps(sample(native, inputs, toy_correct, amplitude + h * d), truth, fair)
                         - crps(sample(native, inputs, toy_correct, amplitude - h * d), truth, fair)) / (2 * h)
                        for d in directions
                    ])
            analytical = directions @ gradients
            assert all(fd_status(a, b, c) == "supported" for a, b, c in
                       zip(analytical.tolist(), slopes[0.01].tolist(), slopes[0.005].tolist()))
            evidence.append(dict(endpoint=endpoint, fair=fair, gradients=gradients.tolist(),
                                 maxabs_fd_error=max(float((x - analytical).abs().max()) for x in slopes.values())))
    # Removing another member does not change this member's trajectory.
    separate = {k: value[2:3] for k, value in inputs.items()}
    torch.testing.assert_close(sample(native, separate, toy_correct)[0], sample(native, inputs, toy_correct)[2])
    return evidence


@torch.no_grad()
def legacy_sampler_parity():
    class LabelAwareToy(ToyNet):
        def forward(self, x, label, history):
            return self.up2(0.3 * x + history + 0.001 * label.to(x.dtype))

    result = {}
    for dtype in (torch.float32, torch.float64):
        native = SimpleNamespace(net=LabelAwareToy(), abar=torch.tensor([.9, .65, .3, 0.], dtype=dtype))
        noise = torch.tensor([-.5, .1, .7, 1.5], dtype=dtype)
        inputs = dict(history=torch.full_like(noise, .1), noise=noise, E=noise, W=noise, W_mean=noise)
        old_wrapper = SimpleNamespace(native=native, correction=toy_correct)
        old_n = native_sample(native, inputs["history"], noise)
        old_j = CorrectedDiffusion.sample(old_wrapper, **inputs)
        new_n = sample(native, inputs)
        new_j = sample(native, inputs, toy_correct)
        zeros = sample(native, inputs, toy_correct, noise.new_zeros(4))
        ones = sample(native, inputs, toy_correct, noise.new_ones(4))
        assert torch.equal(old_n, new_n) and torch.equal(old_n, zeros)
        assert torch.equal(old_j, new_j) and torch.equal(old_j, ones)
        assert not torch.equal(old_n, old_j)
        result[str(dtype)] = dict(native_exact=True, joint_exact=True, zero_endpoint_exact=True,
                                  unit_endpoint_exact=True)
    return dict(cpu_synthetic_only=True, real_forward_groups=0, results=result)


def run_synthetic(destination):
    subset = subset_test()
    smooth = sampler_test()
    norm = dict(mean=torch.tensor([10., -7.]).reshape(2, 1, 1, 1),
                std=torch.tensor([2., 4.]).reshape(2, 1, 1, 1))
    paths = torch.arange(48, dtype=torch.float64).reshape(3, 2, 2, 2, 2) / 10
    truth = norm["mean"] + norm["std"] * (paths.mean(0) + torch.tensor([1., 2.]).reshape(2, 1, 1, 1))
    meta = dict(split="synthetic", Re="1", start=420, trajectory_id="synthetic")
    rows = score_rows(meta, "C1", "N", paths, truth, norm)
    torch.testing.assert_close(
        torch.tensor([(r["mse"], r["smse"]) for r in rows], dtype=torch.float64),
        torch.tensor([(34., 2.5), (4., 1.), (64., 4.)], dtype=torch.float64),
        atol=1e-12, rtol=0,
    )
    decomposed = decompose(paths, 1.1 * paths + .2)
    families = {
        "COMPONENT": (dict(N=paths, O=paths+.2, G=paths*.9, J=paths*.9+.2), COMPONENT),
        "DECOMPOSITION": (decomposed, DECOMPOSITION),
        "AMPLITUDE": (dict(N=paths, O=paths+.1, J=paths+.2, half=paths+.15), AMPLITUDE),
        "CONDITION": (dict(J=paths, E_off=paths+.1, W_off=paths+.2), CONDITION),
    }
    for name, (states, contrasts) in families.items():
        rows = [row for s, p in states.items() for row in score_rows(meta, "C1", s, p, truth, norm)]
        save_scores(destination, name, rows, contrasts)
    return dict(passed=True, subset=subset, smooth_sampler=smooth,
                legacy_sampler_parity=legacy_sampler_parity(),
                channel_mse=True, synthetic_backward_calls=6, synthetic_autograd_grad_calls=0)
