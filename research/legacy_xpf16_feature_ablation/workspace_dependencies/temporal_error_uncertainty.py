"""A small direct error head sharing filters across PDE forecast time."""

import torch
from torch import nn
import torch.nn.functional as F


MODES = ("quantile_base", "quantile_last", "gaussian_last",
         "logscale_base", "logscale_last")


class TemporalErrorHead(nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode
        # Map last-call diffusion features to four channels per PDE future frame.
        self.projection = nn.Conv2d(48, 80, 1)
        output_channels = 4 if mode.startswith("quantile") else 2
        self.net = nn.Sequential(
            nn.Conv3d(18, 16, 3, padding=1), nn.GroupNorm(4, 16), nn.SiLU(),
            nn.Conv3d(16, 16, 3, padding=1), nn.GroupNorm(4, 16), nn.SiLU(),
            nn.Conv3d(16, output_channels, 1))

    def parameters_of_error(self, base, last):
        feature = self.projection(last).reshape(len(last), 4, 20, 32, 64)
        if self.mode.endswith("base"):
            feature = feature * 0
        output = self.net(torch.cat([base, feature], 1))
        if self.mode.startswith("quantile"):
            midpoint, raw_width = output[:, :2], output[:, 2:]
            width = F.softplus(raw_width) + 1e-4
            return midpoint - width, midpoint + width
        return output.clamp(-7, 3).exp()

    def loss(self, base, last, residual):
        parameters = self.parameters_of_error(base, last)
        if self.mode.startswith("quantile"):
            lower, upper = parameters
            e_lower, e_upper = residual - lower, residual - upper
            return (torch.maximum(.05 * e_lower, -.95 * e_lower)
                    + torch.maximum(.95 * e_upper, -.05 * e_upper)).mean()
        if self.mode.startswith("gaussian"):
            return (parameters.log() + .5 * (residual / parameters).square()).mean()
        return (parameters.log() - (residual.abs() + 1e-4).log()).square().mean()

    def bounds(self, base, last):
        parameters = self.parameters_of_error(base, last)
        if self.mode.startswith("quantile"):
            return parameters
        if self.mode.startswith("gaussian"):
            parameters = 1.6448536269514722 * parameters
        # Logscale is an error-scale estimate, not a claimed Gaussian sigma.
        return -parameters, parameters
