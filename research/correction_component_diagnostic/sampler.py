"""Frozen component switches on the original four-step state update."""

import torch


LABELS = (99, 66, 33, 0)


def sample(native, inputs, correction=None, amplitudes=None, trace=None):
    history = inputs["history"]
    x = inputs["noise"].clone()
    for index, label_value in enumerate(LABELS):
        step = 3 - index
        label = torch.full((len(x),), label_value, device=x.device, dtype=torch.long)
        if correction is None:
            clean = native.net(x, label, history)
        else:
            features = []
            hook = native.net.up2.register_forward_hook(
                lambda module, args, result: features.append(result)
            )
            try:
                base = native.net(x, label, history)
            finally:
                hook.remove()
            assert len(features) == 1
            delta = correction(
                x, history, inputs["E"], inputs["W"], inputs["W_mean"],
                features[0], label,
            )
            clean = base + (delta if amplitudes is None else amplitudes[index] * delta)
            if trace is not None:
                trace(label_value, x, features[0], delta, clean)
        alpha = native.abar[step]
        eps = (x - alpha.sqrt() * clean) / (1 - alpha).sqrt()
        prev = native.abar[step - 1] if step else x.new_tensor(1.)
        x = prev.sqrt() * clean + (1 - prev).sqrt() * eps
    return x
