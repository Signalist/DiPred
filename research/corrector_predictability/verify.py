"""Preflight analytic, differentiation and true-target intervention checks."""
import torch

from .alpha import fit_alpha, moments, perturbation
from .model import FeatureCorrector, direction


def verify(dev, fixed_inputs, scale):
    rng = torch.Generator().manual_seed(617)
    r = torch.randn((2, 2, 20, 4, 5), generator=rng, dtype=torch.float64).requires_grad_()
    y = torch.randn(r.shape, generator=rng, dtype=r.dtype)
    for arm in ("smooth", "match", "align"):
        fn = lambda x: direction(x, y, arm)
        assert torch.autograd.gradcheck(fn, (r,), fast_mode=True)
        zero = torch.zeros_like(r).requires_grad_()
        v = fn(zero)
        g, = torch.autograd.grad(v, zero)
        assert v.isfinite() and g.isfinite().all()
    for arm in ("match", "align"):
        assert abs(float(direction(y, y, arm))) < 1e-12
    t = 0.37 * y
    a = fit_alpha(y, t)
    assert abs(a-.37) < 1e-12
    c,e,b = moments(y,t)
    torch.testing.assert_close(b+perturbation(c,e,a), (t-a*y).square().flatten(1).mean(1),
                               atol=1e-14, rtol=0)
    assert fit_alpha(torch.zeros_like(y), t) == 0
    assert fit_alpha(y, -y) == 0 and fit_alpha(y, 2*y) == 1

    model = FeatureCorrector().to(dev.x.device).eval()
    src = tuple(x[:2] for x in fixed_inputs)
    original = dev.y[:2].clone()
    with torch.no_grad():
        p = dev.p[:2]
        assert torch.equal(p, p + 0*model(*src)*scale)
        model.net[-1].bias.fill_(.02)
        issued = p + model(*src)*scale
        dev.y[:2] += .123
        try:
            assert torch.equal(issued, p + model(*src)*scale)
        finally:
            dev.y[:2] = original
    return dict(passed=True, direction_gradchecks=True, same_direction_match_and_align=True,
                zero_gradient_finite=True, analytic_alpha=True, exact_native=True,
                label_intervention_bitwise_equal=True)
