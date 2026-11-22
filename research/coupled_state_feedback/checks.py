"""One controlled set of analytical and real-training-sample prerequisites."""
import math

import torch
from .core import Forecast, objective, scores


def formulas():
    p = torch.tensor([[.2, -.3, .7], [1., .1, -.5], [-.4, 1.2, .3]], dtype=torch.double, requires_grad=True)
    y = torch.tensor([.4, -.1, .8], dtype=torch.double)
    m = len(p)
    crps = sum((v-y).abs().mean() for v in p)/m
    crps -= sum((p[i]-p[j]).abs().mean() for i in range(m) for j in range(m) if i != j)/(2*m*(m-1))
    es = sum(torch.linalg.vector_norm(v-y)/math.sqrt(3) for v in p)/m
    es -= sum(torch.linalg.vector_norm(p[i]-p[j])/math.sqrt(3) for i in range(m) for j in range(m) if i != j)/(2*m*(m-1))
    a, b = scores(p, y)
    assert torch.allclose(a, crps, atol=1e-14) and torch.allclose(b, es, atol=1e-14)
    loss = objective(p, y)
    g = torch.autograd.grad(loss, p)[0]
    q = p.detach()[[2, 0, 1]].requires_grad_(True)
    qloss = objective(q, y)
    qg = torch.autograd.grad(qloss, q)[0]
    assert torch.allclose(loss, qloss, atol=1e-14) and torch.allclose(g[[2, 0, 1]], qg, atol=1e-14)
    assert torch.autograd.gradcheck(lambda v: objective(v, y), (p,), eps=1e-6, atol=1e-6)
    # Norm gradients remain finite even when members coincide exactly.
    identical = p.detach()[:1].expand(3, -1).clone().requires_grad_(True)
    assert torch.isfinite(torch.autograd.grad(objective(identical, y), identical)[0]).all()
    return dict(explicit_crps=float(a), explicit_es=float(b), gradcheck=True,
                permutation_loss_and_gradient=True, coincident_members_finite=True)


def norm(parameters):
    return sum(float(p.grad.abs().sum()) for p in parameters if p.grad is not None)


def real_checks(native, h, y, z):
    result = formulas()
    c, o, f = (Forecast(native, arm) for arm in ('C', 'O', 'F'))
    initial = c.trainable_state()
    assert all(torch.equal(c.trainable_state()[k], o.trainable_state()[k]) for k in initial)
    original_h = h.clone()
    with torch.no_grad():
        ref = Forecast(native, 'N')(h, z)
        errors = {model.arm: float((model(h, z)-ref).abs().max()) for model in (f, o, c)}
        assert max(errors.values()) <= 2e-6
    result['zero_feedback_max_abs_standardized'] = errors
    opt = c.optimizer()
    pred = c(h, z)
    objective(pred, y).backward()
    projection_grad = norm(c.feedback.parameters())
    first_g_grad = norm(c.update.parameters())
    assert projection_grad > 0 and first_g_grad == 0
    opt.step()
    opt.zero_grad(set_to_none=True)
    trace = []
    pred = c(h, z, trace)
    loss = objective(pred, y)
    # r after first G is downstream of p3 and has no sampler bypass.
    dloss_dr = torch.autograd.grad(loss, trace[0]['r'], retain_graph=True)[0]
    feedback_to_p3 = torch.autograd.grad(trace[0]['r'], trace[0]['p'], grad_outputs=dloss_dr, retain_graph=True)[0]
    assert float(feedback_to_p3.abs().sum()) > 0
    loss.backward()
    assert norm(c.update.parameters()) > 0
    result['gradients'] = dict(initial_feedback=projection_grad, initial_G=first_g_grad,
        after_projection_G=norm(c.update.parameters()), final_loss_via_R_to_p3=float(feedback_to_p3.abs().sum()))
    frozen_before = {k: p.detach().clone() for k, p in c.named_parameters() if not p.requires_grad}
    opt.step()
    changed = {k: float((v-initial[k]).abs().max()) for k, v in c.trainable_state().items()}
    assert any(v > 0 for k, v in changed.items() if k.startswith('base.net.up2.'))
    assert all(v > 0 for k, v in changed.items() if k.startswith('base.net.out.'))
    assert all(torch.equal(p, frozen_before[k]) for k, p in c.named_parameters() if not p.requires_grad)
    result['smoke_updates'] = dict(up2=True, out=True, G=True, feedback=True, frozen_unchanged=True,
                                  discarded_optimizer_steps=2)
    with torch.no_grad():
        p = trace[0]['p'].detach()
        r = torch.zeros((len(h), 8, 20, 32, 64), device=h.device)
        perturb = torch.zeros_like(p)
        perturb[:, :, 9] = .1
        rc = c.update(r, p, None, h, 3, True)
        rc2 = c.update(r, p+perturb, None, h, 3, True)
        ro = o.update(r, p, None, h, 3, False)
        ro2 = o.update(r, p+perturb, p*.9, h, 3, False)
        adjacent = float((rc2-rc)[:, :, [8, 10]].abs().max())
        assert adjacent > 0 and torch.equal(ro, ro2)
        labels = torch.full((len(h),), 66, device=h.device, dtype=torch.long)
        downstream = float((c.denoise(z, labels, h, rc)-c.denoise(z, labels, h, rc2)).abs().max())
        assert downstream > 0
        p1 = c(h, z)
        _ = objective(p1, y+1.)  # Truth is only consumed by scoring.
        p2 = c(h, z)
        assert torch.equal(p1, p2) and torch.equal(h, original_h)
        separated = torch.cat([c(h[i:i+1], z[i:i+1]) for i in range(len(h))])
        assert torch.allclose(p1, separated, atol=2e-6, rtol=0)
        zero_trace = []
        c(h, z, zero_trace)
        assert torch.equal(zero_trace[0]['r_input'], r)
        assert zero_trace[-1]['r'] is zero_trace[-1]['r_input']
    result['interventions'] = dict(C_state_delta=float((rc2-rc).abs().max()),
        adjacent_physical_frames_delta=adjacent, O_masked_delta=float((ro2-ro).abs().max()),
        fixed_x_h_output_delta=downstream, truth_invariance=True, history_unchanged=True,
        repeat_state_reset=True, members_batch_independence_max=float((p1-separated).abs().max()),
        G_calls=3, backbone_calls=4)
    # Exercise exact gradient checkpointing equivalence without changing M or loss.
    a = c(h, z)
    b = c(h, z, checkpointing=True)
    assert torch.equal(a, b)
    ga = torch.autograd.grad(objective(a, y), tuple(p for p in c.parameters() if p.requires_grad))
    gb = torch.autograd.grad(objective(b, y), tuple(p for p in c.parameters() if p.requires_grad))
    assert all(torch.allclose(a, b, atol=1e-6, rtol=1e-5) for a, b in zip(ga, gb))
    result['checkpointing_output_gradient_equivalence'] = True
    return result
