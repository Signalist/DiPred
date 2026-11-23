"""Match the previous J model while changing only its historical input slots."""
from typing import NamedTuple

import torch
import torch.nn.functional as F

from research.observed_residual_feedback.core import Forecast, Publication, objective
from research.observed_residual_feedback.run import noise_for, trainable


ARMS = ('C0', 'CH', 'CS')


class ConditionSlots(NamedTuple):
    first: torch.Tensor
    second: torch.Tensor
    available: bool


class ControlForecast(Forecast):
    def __init__(self, native):
        super().__init__(native, 'J')

    def encode(self, history, slots, masked=False):
        assert not masked
        available = history.new_full((1, 32, 64), float(slots.available))
        inputs = torch.cat([item.reshape(-1, 32, 64)
                            for item in (history, slots.first, slots.second, available)])
        return self.encoder(inputs[None])


def spatial_mean(field):
    # Treat channels and physical frames as independent planes.
    return F.avg_pool2d(F.pad(field, (1, 1, 1, 1), mode='replicate'), 3, stride=1)


def represent(arm, observed, older_history):
    zero = torch.zeros_like(observed.residual)
    if arm == 'C0':
        return ConditionSlots(zero, zero, observed.available)
    if arm == 'CH':
        assert (older_history is not None) == observed.available
        return ConditionSlots(older_history if observed.available else zero, zero, observed.available)
    assert arm == 'CS'
    return ConditionSlots(spatial_mean(observed.residual), spatial_mean(observed.correction),
                  observed.available)


class ControlPublication(Publication):
    """Maintain true issued residuals independently of how the model consumes them."""

    def __init__(self, history, arm):
        super().__init__(history)
        self.older_history = None
        self.arm = arm

    def encoded_memory(self):
        return represent(self.arm, self.memory, self.older_history)

    def issue(self, model, noise):
        return super().issue(
            lambda history, noise, _observed: model(history, noise, self.encoded_memory()), noise)

    def reveal(self, truth):
        earlier = self.history.detach()
        super().reveal(truth)
        self.older_history = earlier


def fit_update(arm, model, optimizer, observation):
    """The original J update, using the representation-specific publication state."""
    optimizer.zero_grad(set_to_none=True)
    publication = ControlPublication(observation.read_window(0), arm)
    losses = []
    for n, seed in enumerate(observation.entry['noise_seeds'], 1):
        noise = noise_for(publication.history, seed)
        with torch.set_grad_enabled(n > 1):
            backbone, correction = publication.issue(model, noise)
            truth = observation.read_window(n)
            publication.reveal(truth)
            if n > 1:
                loss = objective(backbone, correction, truth)
                (loss / 2).backward()
                losses.append(loss.detach())
    grad = torch.nn.utils.clip_grad_norm_(list(trainable(model).values()), 1., error_if_nonfinite=True)
    optimizer.step()
    return [float(loss) for loss in losses], float(grad)
