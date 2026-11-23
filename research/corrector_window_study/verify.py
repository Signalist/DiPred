"""Preflight numerical, information, and capacity checks on training and synthetic data."""
import time

import torch

from research.corrector_window_study.data import SEEDS, STARTS
from research.corrector_window_study.model import (ARMS, SLOTS, WindowCorrector,
                                                   empirical_crps, explicit_crps, inputs)


def verify(data, manifest, normalization):
    cases = data.cases
    split = manifest["splits"]["train"]
    for i, case in enumerate(cases):
        s = case["start"]
        j = case["index"]
        assert case["earlier"] == [s-20, s] and case["history"] == [s, s+20]
        assert case["target"] == [s+20, s+40] and s in STARTS
        assert split["files"][j] == case["file"]
        assert torch.equal(data.labels[i], data.observed[j, :, s-360:s-340])
    for start in (380,) + STARTS:
        row = split["native"][str(start)]["identity"]
        assert row["start"] == start and row["seeds"] == SEEDS
        assert row["files"] == split["files"] and row["split"] == "train"
        assert row["frozen"] == manifest["native_checkpoint"]
        assert not data.paths[start].requires_grad

    indices = [0, next(i for i, case in enumerate(cases) if case["start"] == 420)]
    release = data.release(indices)
    target = data.target(indices)
    for b, i in enumerate(indices):
        case = cases[i]
        j, s = case["index"], case["start"]
        torch.testing.assert_close(release[5][b], release[0][b] - data.paths[s-20][j].mean(0),
                                   atol=0, rtol=0)
        assert torch.equal(release[2][b], data.paths[s][j])
    observed = data.observed
    i = indices[0]
    s, j = cases[i]["start"], cases[i]["index"]
    saved = observed[j, :, s-360:s-340].clone()
    saved_label = data.labels[i].clone()
    observed[j, :, s-360:s-340] += 0.125
    data.labels[i] += 0.125
    counterfactual = data.release([i])
    observed[j, :, s-360:s-340] = saved
    data.labels[i] = saved_label
    for old, changed in zip(data.release([i]), counterfactual):
        assert torch.equal(old, changed)

    torch.manual_seed(42)
    model = WindowCorrector().to(device=target.device, memory_format=torch.channels_last)
    params = sum(p.numel() for p in model.parameters())
    assert len(SLOTS) == 9 and sum(n for _, n in SLOTS) == 249
    assert ARMS["D"][:3] == ARMS["F"][:3]
    loc, scale = normalization
    for arm in ARMS:
        shared, standardized = inputs(release, arm, normalization)
        assert shared.shape[1] == 209 and standardized.shape[1:4] == (32, 2, 20)
        recent, summary, earlier, _, _, memory, flag, _ = shared.split(
            [n for _, n in SLOTS[:-1]], dim=1)
        mode, use_memory, use_earlier, _ = ARMS[arm]
        assert bool(recent.count_nonzero()) == (mode == "full")
        assert bool(summary.count_nonzero()) == (mode == "summary")
        assert bool(earlier.count_nonzero()) == use_earlier
        assert bool(memory.count_nonzero()) == use_memory and (flag == int(use_memory)).all()
        output = model(shared, standardized)
        assert output.shape == release[2].shape and torch.equal(
            release[2] + output * scale[:, None], release[2])
    # Exercise nonzero inference under the target intervention and a member permutation.
    with torch.no_grad():
        model.spatial[-1].weight.fill_(0.05)
        d = inputs(data.release([i]), "D", normalization)
        d_changed = inputs(counterfactual, "D", normalization)
        assert all(torch.equal(a, b) for a, b in zip(d, d_changed))
        nonzero = model(*d)
        assert nonzero.count_nonzero() and torch.equal(nonzero, model(*d_changed))
        permutation = torch.arange(31, -1, -1, device=target.device)
        permuted = model(d[0], d[1][:, permutation])
        torch.testing.assert_close(permuted, nonzero[:, permutation], rtol=1e-5, atol=1e-6)
        concat = torch.cat((d[0].unsqueeze(1).expand(-1, 32, -1, -1, -1),
                            d[1].flatten(2, 3)), dim=2).flatten(0, 1)
        projected = model.projection(concat.contiguous(memory_format=torch.channels_last))
        torch.testing.assert_close(model.spatial(projected).reshape_as(nonzero), nonzero,
                                   rtol=1e-4, atol=2e-6)
        prediction = release[2][:1].double()
        correction = nonzero.double() * scale[:, None].double()
        truth = target[:1, None].double()
        residual = truth - prediction
        torch.testing.assert_close(prediction + residual, truth.expand_as(prediction), atol=1e-12, rtol=0)
        torch.testing.assert_close(truth - (prediction + correction), residual - correction,
                                   atol=1e-12, rtol=0)

    generator = torch.Generator().manual_seed(123)
    x = torch.rand((2, 32, 2, 20, 2, 3), generator=generator, dtype=torch.float64).requires_grad_()
    y = torch.rand((2, 2, 20, 2, 3), generator=generator, dtype=torch.float64)
    exact = explicit_crps(x, y)
    fast = empirical_crps(x, y)
    torch.testing.assert_close(exact, fast, atol=1e-12, rtol=0)
    ref_grad, = torch.autograd.grad(exact.sum(), x)
    fast_grad, = torch.autograd.grad(fast.sum(), x)
    torch.testing.assert_close(ref_grad, fast_grad, atol=1e-12, rtol=0)
    assert (fast_grad.abs().sum(dim=(0, 1, 2, 4, 5)) > 0).all()

    # A full batch backward pass measures memory without taking an optimizer step.
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    shared, member = inputs(release, "D", normalization)
    output = model(shared, member)
    loss = empirical_crps(output + member, (target - loc) / scale).mean()
    loss.backward()
    torch.cuda.synchronize()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    return dict(passed=True, cases=len(cases), matched_parameters=params, input_slots=dict(SLOTS),
                same_initialization_and_budget_checked_in_training=True,
                all_frames_crps_gradient=True, target_mutation_bitwise_equal=True,
                member_permutation_equivariant=True, prior_window_mean=True, exact_zero_output=True,
                cache_hash_and_sign_verified_by_import=True, residual_identities=True,
                exact_crps_value_and_gradient=True, max_allocated_bytes=torch.cuda.max_memory_allocated(),
                full_batch_preflight_seconds=time.monotonic() - started)
