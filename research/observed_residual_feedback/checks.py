"""Pre-training checks on synthetic identities and one real training history."""
import time

import torch

from research.observed_residual_feedback.core import (
    FIELD, Forecast, Memory, Publication, field_metrics, score_components, score_u,
)


def numerical_checks():
    paths = torch.tensor([[0., 1.], [2., 4.], [3., -1.]], dtype=torch.double, requires_grad=True)
    truth = torch.tensor([1., 2.], dtype=torch.double)
    point, pairs, first, es_pairs = score_components(paths, truth)
    for denominator in (6, 9):
        direct_l1 = (paths[:, None] - paths[None, :]).abs().sum((0, 1)) / (2 * denominator)
        direct_rms = (paths[:, None] - paths[None, :]).square().mean(-1).sqrt().sum() / (2 * denominator)
        assert torch.allclose(pairs / denominator, direct_l1)
        assert torch.allclose(es_pairs / denominator, direct_rms)
        assert torch.allclose(point, (paths - truth).abs().mean(0))
        assert torch.allclose(first, (paths - truth).square().mean(-1).sqrt().mean())
    assert torch.autograd.gradcheck(score_u, (paths, truth))
    scale = torch.tensor([2., 5.], dtype=torch.double).reshape(2, 1, 1, 1)
    mean = torch.tensor([7., -3.], dtype=torch.double).reshape_as(scale)
    data = torch.tensor([[1., 2.], [3., 6.]], dtype=torch.double).reshape(2, 2, 1, 1, 1)
    target = torch.zeros_like(data[0])
    for channel, expected in enumerate([(16., 4., 3.6), (400., 100., 18.)]):
        sl = slice(channel, channel + 1)
        values = field_metrics(data[:, sl], target[sl], scale[sl])
        for metric, wanted in zip(('mse', 'variance', 'width90'), expected):
            assert abs(values[metric] - wanted) < 1e-12
    correction = data / 2
    residual = target - data
    assert torch.equal(target.expand_as(data), data + residual)
    assert torch.equal(residual - correction, target - (data + correction))
    target_phys, backbone_phys = target * scale + mean, data * scale + mean
    assert torch.equal(target_phys.expand_as(data), backbone_phys + residual * scale)
    assert torch.equal((residual - correction) * scale,
                       target_phys - (backbone_phys + correction * scale))


def publication_check():
    events = []

    def toy(history, noise, memory):
        events.append('issued')
        return noise + history + memory.residual, noise + memory.correction

    history = torch.zeros(FIELD)
    noise = torch.ones(2, *FIELD)
    streams = [Publication(history), Publication(history)]
    try:
        streams[0].reveal(history)
    except RuntimeError:
        pass
    else:
        raise AssertionError('Reveal before issue must fail')
    a, correction = streams[0].issue(toy, noise)
    b, _ = streams[1].issue(toy, noise)
    try:
        streams[0].issue(toy, noise)
    except RuntimeError:
        pass
    else:
        raise AssertionError('Double issue must fail')
    assert torch.equal(a, b) and all(not stream.memory.available for stream in streams)
    for stream, truth in zip(streams, (a.mean(0), b.mean(0) + 2)):
        events.append('revealed')
        stream.reveal(truth)
    # A genuine zero residual is still marked available, and R uses the backbone.
    assert streams[0].memory.available and not streams[0].memory.residual.count_nonzero()
    assert torch.equal(streams[0].memory.correction, correction.mean(0))
    assert torch.equal(streams[1].memory.residual, torch.full(FIELD, 2.))
    after = [stream.issue(toy, noise)[0] for stream in streams]
    assert not torch.equal(*after)
    assert events == ['issued', 'issued', 'revealed', 'revealed', 'issued', 'issued']


def preflight():
    from research.observed_residual_feedback.run import (
        OUT, Observations, dump, frozen_tensors, load, noise_for, read, sha, trainable,
    )
    assert not (OUT / 'PREFLIGHT.json').exists()
    started = time.monotonic()
    numerical_checks()
    publication_check()
    native, norm, settings = load()
    entry = read(OUT / 'schedule.json')['train'][0]
    obs = Observations(entry, norm, settings['data_root'])
    history = obs.read_window(0)
    seed = entry['noise_seeds'][0]
    noise = noise_for(history, seed)
    history_batch = history[None].expand_as(noise).contiguous()
    cold = Memory.cold(history)
    warm = Memory(torch.ones_like(history), cold.correction, True)
    zero = Memory(cold.residual, cold.correction, True)
    gradients = {}
    native_error = {}
    with torch.no_grad():
        reference = native.rollout(history_batch, seed=seed, sign=1)
    for arm in ('L', 'J'):
        model = Forecast(native, arm)
        frozen = frozen_tensors(model)
        with torch.no_grad():
            backbone, correction = model(history, noise, cold)
            native_error[arm] = float((backbone - reference).abs().max())
            assert native_error[arm] == 0 and not correction.count_nonzero()
            # Probe opened output routes in temporary models, with no optimizer.
            model.projection.weight.fill_(.01)
            model.head[-1].weight.fill_(.01)
            backbone_warm, head_warm = model(history, noise, warm)
            backbone_zero, _ = model(history, noise, zero)
            assert torch.equal(backbone_warm, backbone_zero) == (arm == 'L')
            fixed = backbone_warm.reshape(len(noise), 40, 32, 64)
            fields = [fixed, history_batch.reshape(len(noise), 40, 32, 64)]
            head_zero = model.head(torch.cat([*fields, model.encode(history, zero).expand(len(noise), -1, -1, -1)], 1))
            assert not torch.equal(head_warm, head_zero.reshape_as(head_warm))
        backbone, correction = model(history, noise, warm)
        truth = obs.read_window(1)
        # Verify the head reaches the current backbone independently of the backbone loss.
        head_gradient = torch.autograd.grad(correction.square().mean(), backbone, retain_graph=True)[0]
        assert head_gradient.abs().sum() > 0
        from research.observed_residual_feedback.core import objective
        objective(backbone, correction, truth).backward()
        gradients[arm] = {}
        for prefix in ('base.net.up2', 'base.net.out', 'encoder', 'projection', 'head'):
            norm_sum = sum(float(p.grad.abs().sum()) for name, p in trainable(model).items()
                           if name.startswith(prefix) and p.grad is not None)
            assert norm_sum > 0
            gradients[arm][prefix] = norm_sum
        assert frozen == frozen_tensors(model)
        assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
    torch.cuda.synchronize()
    dump(OUT / 'PREFLIGHT.json', dict(passed=True, config_sha256=sha(OUT / 'FROZEN.json'),
        seconds=time.monotonic() - started, native_zero_init_max_abs=native_error,
        L_masks_backbone_slots=True, J_backbone_uses_residual=True, both_heads_use_residual=True,
        gradients=gradients, head_backprop_to_current_backbone=True, frozen_tensors_unchanged=True,
        publication_order_and_future_label_pollution=True, scale_and_U_V_checks=True))
