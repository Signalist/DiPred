"""Observed residuals become available only after a complete window is issued."""
import copy
from typing import NamedTuple

import torch
from torch import nn

FIELD = (2, 20, 32, 64)
MEMBERS = 16


class Memory(NamedTuple):
    residual: torch.Tensor
    correction: torch.Tensor
    available: bool

    @classmethod
    def cold(cls, history):
        return cls(torch.zeros_like(history), torch.zeros_like(history), False)


class Forecast(nn.Module):
    def __init__(self, native, arm):
        super().__init__()
        self.arm = arm
        self.base = copy.deepcopy(native).requires_grad_(False)
        if arm == 'N':
            return
        assert arm in ('J', 'L')
        self.base.net.up2.requires_grad_(True)
        self.base.net.out.requires_grad_(True)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42)
            self.encoder = nn.Sequential(nn.Conv2d(121, 16, 1), nn.SiLU(),
                                         nn.Conv2d(16, 16, 3, padding=1), nn.SiLU())
            self.projection = nn.Conv2d(16, 48, 1)
            self.head = nn.Sequential(nn.Conv2d(96, 16, 1), nn.SiLU(),
                                      nn.Conv2d(16, 16, 3, padding=1), nn.SiLU(),
                                      nn.Conv2d(16, 40, 1))
            for layer in (self.projection, self.head[-1]):
                nn.init.zeros_(layer.weight)
                nn.init.zeros_(layer.bias)
        self.to(next(native.parameters()).device)

    def encode(self, history, memory, masked=False):
        residual = torch.zeros_like(memory.residual) if masked else memory.residual
        correction = torch.zeros_like(memory.correction) if masked else memory.correction
        available = history.new_full((1, 32, 64), float(memory.available))
        slots = torch.cat([item.reshape(-1, 32, 64)
                           for item in (history, residual, correction, available)])
        return self.encoder(slots[None])

    def forward(self, history, noise, memory):
        assert history.shape == FIELD and noise.shape == (MEMBERS, *FIELD)
        history_batch = history.unsqueeze(0).expand_as(noise).contiguous()
        hook = None
        if self.arm != 'N':
            full = self.encode(history, memory)
            backbone_condition = full if self.arm == 'J' else self.encode(history, memory, masked=True)
            feedback = self.projection(backbone_condition)
            # Reuse the native denoiser and inject the same conditioning into all four calls.
            hook = self.base.net.first.register_forward_hook(lambda _module, _inputs, first: first + feedback)
        try:
            noisy_window = noise
            for step in (3, 2, 1, 0):
                alpha = self.base.abar[step]
                labels = torch.full((MEMBERS,), step * 33, device=noise.device, dtype=torch.long)
                estimate = self.base.net(noisy_window, labels, history_batch)
                estimated_noise = (noisy_window - alpha.sqrt() * estimate) / (1 - alpha).sqrt()
                prior_alpha = self.base.abar[step - 1] if step else noise.new_tensor(1.)
                noisy_window = prior_alpha.sqrt() * estimate + (1 - prior_alpha).sqrt() * estimated_noise
        finally:
            if hook is not None:
                hook.remove()
        backbone = noisy_window
        if self.arm == 'N':
            return backbone, torch.zeros_like(backbone)
        fields = torch.cat([backbone.reshape(MEMBERS, 40, 32, 64),
                            history_batch.reshape(MEMBERS, 40, 32, 64),
                            full.expand(MEMBERS, -1, -1, -1)], 1)
        return backbone, self.head(fields).reshape_as(backbone)


class Publication:
    """One trajectory and one policy. Pending forecasts cannot be overwritten."""
    def __init__(self, history):
        self.history = history
        self.memory = Memory.cold(history)
        self.pending = None
        self.next_window = 1

    def issue(self, model, noise):
        if self.pending is not None or self.next_window > 3:
            raise RuntimeError('No window can be issued in the current state')
        backbone, correction = model(self.history, noise, self.memory)
        self.pending = (backbone.detach().mean(0), correction.detach().mean(0))
        return backbone, correction

    def reveal(self, truth):
        if self.pending is None or truth.shape != FIELD:
            raise RuntimeError('Reveal requires a pending forecast and a full window')
        backbone_mean, correction_mean = self.pending
        self.history = truth.detach().clone()
        self.memory = Memory(self.history - backbone_mean, correction_mean, True)
        self.pending = None
        self.next_window += 1


def score_components(paths, truth):
    """CRPS fields and RMS ES from the same paths for U and empirical V."""
    members = len(paths)
    ordered = paths.sort(dim=0).values
    weights = (2 * torch.arange(members, dtype=paths.dtype, device=paths.device) - members + 1)
    weight_shape = (members,) + (1,) * (paths.ndim - 1)
    absolute = (paths - truth).abs().mean(0)
    crps_pair = (ordered * weights.reshape(weight_shape)).sum(0)
    flat = paths.flatten(1)
    left, right = torch.triu_indices(members, members, 1, device=paths.device)
    rms_scale = flat.shape[1] ** .5
    first = torch.linalg.vector_norm(flat - truth.flatten(), dim=1).mean() / rms_scale
    # Each unordered pair contributes once after the factor of 1/2.
    es_pair = torch.linalg.vector_norm(flat[left] - flat[right], dim=1).sum() / rms_scale
    return absolute, crps_pair, first, es_pair


def score_u(paths, truth):
    absolute, crps_pair, first, es_pair = score_components(paths, truth)
    denominator = len(paths) * (len(paths) - 1)
    return (absolute - crps_pair / denominator).mean() + .1 * (first - es_pair / denominator)


def objective(backbone, correction, truth):
    target_correction = (truth - backbone.mean(0)).detach()
    auxiliary = (correction.mean(0) - target_correction).square().mean()
    return score_u(backbone + correction, truth) + .25 * score_u(backbone, truth) + .1 * auxiliary


def field_metrics(paths, truth, scale):
    """Slice scale with channels before calling this function."""
    assert scale.shape == (paths.shape[1], 1, 1, 1)
    error = paths.mean(0) - truth
    variance = paths.var(0, unbiased=False)
    quantiles = torch.quantile(paths, paths.new_tensor([.05, .95]), dim=0)
    return dict(smse=float(error.square().mean()), mse=float((error * scale).square().mean()),
                variance=float((variance * scale.square()).mean()),
                standardized_variance=float(variance.mean()),
                width90=float(((quantiles[1] - quantiles[0]) * scale).mean()),
                residual_mae=float(error.abs().mean()),
                residual_signed_mean=float(-error.mean()),
                residual_physical_mae=float((error * scale).abs().mean()),
                residual_physical_signed_mean=float((-error * scale).mean()))
