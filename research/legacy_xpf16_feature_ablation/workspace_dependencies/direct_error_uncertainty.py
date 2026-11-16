"""Direct predictive-error heads for frozen, full twenty-frame DiPred.

Internal features come only from actual four-step zero-noise inference. Neither
future labels nor forward-noised future labels enter feature extraction.
"""

import torch
from torch import nn
import torch.nn.functional as F


MODES = ("gaussian_base", "gaussian_last", "quantile_base",
         "quantile_last", "quantile_all")


@torch.no_grad()
def forecast_features(model, normalized_history):
    stages = []
    hook = model.net.up2.register_forward_hook(
        lambda module, inputs, output: stages.append(output.detach().clone()))
    try:
        prediction = model.rollout(normalized_history, 500000, sign=0)
    finally:
        hook.remove()
    assert len(stages) == 4
    assert all(stage.shape[1:] == (48, 32, 64) for stage in stages)
    # Call order corresponds to diffusion indices3,2,1,0; PDE future stays20.
    return prediction, torch.cat(stages, dim=1)


def inputs(history, prediction, features, stats, mode):
    b = len(history)
    value = torch.cat([history.reshape(b, 40, 32, 64),
                       prediction.reshape(b, 40, 32, 64), features.float()], 1)
    value = ((value - stats["mean"]) / stats["std"]).clamp(-10, 10)
    if mode.endswith("base"):
        value[:, 80:] = 0
    elif mode.endswith("last"):
        value[:, 80:224] = 0
    return value


class DirectErrorHead(nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode
        output_channels = 40 if mode.startswith("gaussian") else 80
        self.net = nn.Sequential(
            nn.Conv2d(272, 32, 3, padding=1), nn.GroupNorm(4, 32), nn.SiLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.GroupNorm(4, 32), nn.SiLU(),
            nn.Conv2d(32, output_channels, 1))

    def parameters_of_error(self, value):
        output = self.net(value).reshape(len(value), -1, 20, 32, 64)
        if self.mode.startswith("gaussian"):
            return output.clamp(-7, 3).exp()
        midpoint, raw_width = output[:, :2], output[:, 2:]
        half_width = F.softplus(raw_width) + 1e-4
        return midpoint - half_width, midpoint + half_width

    def loss(self, value, residual):
        parameters = self.parameters_of_error(value)
        if self.mode.startswith("gaussian"):
            sigma = parameters
            return (sigma.log() + .5 * (residual / sigma).square()).mean()
        lower, upper = parameters
        e_lower, e_upper = residual - lower, residual - upper
        return (torch.maximum(.05 * e_lower, -.95 * e_lower)
                + torch.maximum(.95 * e_upper, -.05 * e_upper)).mean()

    def bounds(self, value):
        parameters = self.parameters_of_error(value)
        if self.mode.startswith("gaussian"):
            radius = 1.6448536269514722 * parameters
            return -radius, radius
        return parameters
