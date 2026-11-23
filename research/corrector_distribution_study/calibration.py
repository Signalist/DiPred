"""Fit positive member-shared scales with the exact ordered empirical CRPS."""

import numpy as np
import torch
from scipy.optimize import minimize

from research.corrector_window_study.model import empirical_crps


STATIC = ("identity", "S0", "S1", "S2")
BOUNDS = (.25, 4.)


def static_basis(family, cases, device, dtype=torch.float32):
    size = {"S0": 1, "S1": 2, "S2": 6}[family]
    basis = torch.zeros(2, 20, size, device=device, dtype=dtype)
    if family == "S0":
        basis.fill_(1.)
    elif family == "S1":
        for channel in range(2):
            basis[channel, :, channel] = 1.
    else:
        lead = np.arange(1, 21)
        for channel in range(2):
            for node in range(3):
                basis[channel, :, 3*channel+node] = torch.as_tensor(
                    np.interp(lead, (1, 10, 20), np.eye(3)[node]), device=device,
                )
    return basis.expand(cases, -1, -1, -1)


def scales(basis, coefficients):
    values = torch.as_tensor(coefficients, dtype=basis.dtype, device=basis.device)
    result = (basis * values).sum(-1)
    assert torch.isfinite(result).all() and (result > 0).all()
    return result


def apply_scale(paths, alpha):
    center = paths.mean(1, keepdim=True)
    return paths + (alpha[:, None, :, :, None, None] - 1.) * (paths - center)


class TrainingCRPS:
    """Sufficient arrays sorted once. Positive alpha scales the pair half-term."""

    def __init__(self, data, scale, batch=8):
        centered, offset, dispersion = [], [], []
        for start in range(0, len(data.cases), batch):
            indices = list(range(start, min(start+batch, len(data.cases))))
            native = data.release(indices)[2]
            center = native.mean(1)
            d = ((native - center[:, None]) / scale[:, None]).sort(1).values
            truth = (data.target(indices) - center) / scale
            m = d.shape[1]
            coef = (2*torch.arange(m, device=d.device, dtype=d.dtype)+1-m) / m**2
            pair = (d * coef[None, :, None, None, None, None]).sum(1).mean((-1, -2))
            centered.append(d)
            offset.append(truth)
            dispersion.append(pair)
        self.centered = torch.cat(centered)
        self.offset = torch.cat(offset)
        self.dispersion = torch.cat(dispersion)

    def value_gradient(self, alpha):
        value, gradients = [], []
        for d, target, pair, a in zip(self.centered.split(16), self.offset.split(16),
                                      self.dispersion.split(16), alpha.split(16)):
            error = d * a[:, None, :, :, None, None] - target[:, None]
            value.append(error.abs().mean((1, 4, 5)) - a*pair)
            gradients.append((error.sign()*d).mean((1, 4, 5)) - pair)
        return torch.cat(value).double().mean(), torch.cat(gradients) / alpha.numel()

    def fit(self, basis):
        assert basis.shape[:3] == self.dispersion.shape
        torch.testing.assert_close(basis.sum(-1), torch.ones_like(self.dispersion, dtype=basis.dtype), atol=1e-7, rtol=0)
        assert (basis >= 0).all()
        evaluations = 0
        def at(parameter):
            nonlocal evaluations
            evaluations += 1
            value, gradient = self.value_gradient(scales(basis, parameter).float())
            derivative = (gradient[..., None]*basis).sum((0, 1, 2))
            return float(value), derivative.double().cpu().numpy()

        start = np.ones(basis.shape[-1])
        result = minimize(at, start, jac=True, method="L-BFGS-B",
                          bounds=[BOUNDS]*len(start),
                          options=dict(maxiter=300, maxfun=280, ftol=1e-11, gtol=2e-7))
        assert evaluations <= 300 and result.nit <= 300 and np.isfinite(result.fun)
        projected = result.jac.copy()
        projected[(result.x <= BOUNDS[0]+1e-7) & (projected > 0)] = 0.
        projected[(result.x >= BOUNDS[1]-1e-7) & (projected < 0)] = 0.
        return dict(coefficients=result.x.tolist(), train_crps=float(result.fun),
                    updates=int(result.nit), evaluations=evaluations, converged=bool(result.success),
                    projected_gradient_max=float(np.abs(projected).max()),
                    termination=str(result.message), range=list(BOUNDS))


def synthetic_check():
    rng = torch.Generator().manual_seed(23)
    p = torch.rand(2, 32, 2, 20, 2, 3, dtype=torch.float64, generator=rng)
    y = torch.rand(2, 2, 20, 2, 3, dtype=p.dtype, generator=rng)
    centered = (p-p.mean(1, keepdim=True)).sort(1).values
    obj = TrainingCRPS.__new__(TrainingCRPS)
    obj.centered = centered
    obj.offset = y-p.mean(1)
    obj.dispersion = (centered * ((2*torch.arange(32, dtype=p.dtype)+1-32)/32**2)
                       [None,:,None,None,None,None]).sum(1).mean((-1,-2))
    a = p.new_full((2, 2, 20), 1.4).requires_grad_()
    pred = apply_scale(p, a)
    loss = empirical_crps(pred, y).mean()
    (direct,) = torch.autograd.grad(loss, a)
    val, grad = obj.value_gradient(a.detach())
    torch.testing.assert_close(loss, val, atol=1e-12, rtol=0)
    torch.testing.assert_close(grad, direct, atol=1e-12, rtol=0)
    torch.testing.assert_close(pred.mean(1), p.mean(1), atol=1e-14, rtol=0)
    assert torch.equal(apply_scale(p, torch.ones_like(a)), p)
    assert torch.equal(pred.argsort(1), p.argsort(1))
    return dict(crps_value_and_gradient=True, mean_identity_order=True)
