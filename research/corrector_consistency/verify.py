"""Synthetic checks and information-boundary tests on a real development case."""
import torch

from .model import ARMS, PointCorrector, neighbors, spatial, temporal, loss


def close(x, y):
    torch.testing.assert_close(x, torch.full_like(x, y), atol=1e-6, rtol=0)


def verify(data, mean, units, scale):
    # A one-edge grid exposes exact pair weighting and boundary behavior.
    field = torch.zeros(1, 2, 20, 1, 2)
    field[:, 0, :, 0, 0] = 1
    field[:, 0, :, 0, 1] = 3
    close(spatial(field, field)["direction"], 0)
    close(spatial(field, field)["magnitude"], 4)
    close(spatial(field, field)["valid_pairs"], 1)
    reverse = field.sign().clone()
    reverse[:, :, :, :, 1] *= -1
    close(spatial(reverse, field)["direction"], 2)
    close(spatial(reverse, field)["magnitude"], 0)
    zero = torch.zeros_like(field)
    close(spatial(zero, field)["direction"], 1)
    close(spatial(zero, zero)["direction"], 0)
    close(spatial(zero, field)["zero"], 1)
    assert sum(a.shape[-2] * a.shape[-1] for a, _ in neighbors(torch.zeros(1, 2, 3))) == 11
    t = torch.arange(20.0).view(1, 1, 20, 1, 1)
    d1, d2 = temporal(torch.ones_like(t))
    close(d1, 0)
    close(d2, 0)
    close(temporal(t)[1], 0)
    close(temporal(t.square())[1], 2)
    close(loss(field, field, "F"), 0)

    rng = torch.Generator().manual_seed(333)
    target = torch.randn(2, 2, 20, 3, 4, generator=rng, dtype=torch.float64)
    pred = torch.randn(target.shape, generator=rng, dtype=torch.float64).requires_grad_()
    # Independently check the isolated added terms for gradient correctness.
    for arm in ARMS:
        fn = lambda p: loss(p, target, arm) - loss(p, target, "A")
        assert torch.autograd.gradcheck(fn, (pred,), fast_mode=True)
        grad, = torch.autograd.grad(fn(pred), pred)
        assert torch.isfinite(grad).all()
        if arm != "A":
            assert grad.abs().max() > 0
        z = torch.zeros_like(pred, requires_grad=True)
        v = loss(z, target, arm)
        (gradient,) = torch.autograd.grad(v, z)
        assert torch.isfinite(v) and torch.isfinite(gradient).all()

    device = data.x.device
    model = PointCorrector().to(device)
    x, p, target = data.x[:1], data.p[:1], data.y[:1]
    forecast = lambda: model((x-mean)/units, (p-mean)/units) * scale
    assert torch.equal(p + forecast(), p)
    with torch.no_grad():
        model.net[-1].bias.fill_(0.01)
        original = forecast()
        saved = target.clone()
        target.add_(0.25)
        changed = forecast()
        target.copy_(saved)
        assert torch.equal(original, changed) and (original != 0).all()
    return dict(passed=True, valid_neighborhood_count=True, same_direction=True,
                opposite_direction=True, zero_finite_no_mask_escape=True, time_polynomials=True,
                matching_identity=True, all_term_gradchecks=True, native_exact_at_step_0=True,
                nonzero_forecast_bitwise_equal_after_target_intervention=True)
