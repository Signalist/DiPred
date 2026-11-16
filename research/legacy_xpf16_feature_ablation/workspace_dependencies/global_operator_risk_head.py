"""A global spatial risk core with a near-matched real parameter budget.

The point predictor is unchanged. Spatial Fourier convolution acts separately
on each future lead, while temporal convolutions mix the complete 20-frame
block. This is not autoregressive forecasting or a time-window subdivision.
"""

import torch
from torch import nn
import torch.nn.functional as F

from fullblock_memory_risk_head import FullBlockMemoryRiskHead


class SpatialSpectralConvolution(nn.Module):
    """Keep two signed height-frequency bands and positive width frequencies.

Complex coefficients use an explicit final real/imaginary axis, so numel()
counts real scalar parameters rather than counting a complex scalar as one.
"""

    def __init__(self, width=12, modes_height=4, modes_width=7):
        super().__init__()
        self.width = width
        self.modes_height = modes_height
        self.modes_width = modes_width
        self.weight = nn.Parameter(
            torch.randn(2, width, width, modes_height, modes_width, 2) / width
        )

    def forward(self, value):
        batch, channels, time, height, width = value.shape
        mh, mw = self.modes_height, self.modes_width
        if channels != self.width or 2 * mh > height or mw > width // 2 + 1:
            raise ValueError("Input grid is too small for the fixed spectral bands.")
        transformed = torch.fft.rfft2(value, dim=(-2, -1), norm="ortho")
        weights = torch.view_as_complex(self.weight)
        output = transformed.new_zeros(batch, self.width, time, height, width // 2 + 1)
        output[:, :, :, :mh, :mw] = torch.einsum(
            "bithw,iohw->bothw", transformed[:, :, :, :mh, :mw], weights[0]
        )
        output[:, :, :, -mh:, :mw] = torch.einsum(
            "bithw,iohw->bothw", transformed[:, :, :, -mh:, :mw], weights[1]
        )
        return torch.fft.irfft2(output, s=(height, width), dim=(-2, -1), norm="ortho")


class GlobalRiskBlock(nn.Module):
    def __init__(self, width=12, padding=4):
        super().__init__()
        self.padding = padding
        self.spectral = SpatialSpectralConvolution(width)
        self.temporal = nn.Conv3d(width, width, (3, 1, 1), padding=(1, 0, 0))
        self.local = nn.Conv3d(width, width, 1)
        self.normalization = nn.GroupNorm(3, width)

    def forward(self, value):
        p = self.padding
        # Replicate extension avoids imposing periodic identification directly
        # on opposite physical boundaries; it is not a boundary-condition proof.
        extended = F.pad(value, (p, p, p, p, 0, 0), mode="replicate")
        global_value = self.spectral(extended)[..., p:-p, p:-p]
        combined = global_value + self.temporal(value) + self.local(value)
        return F.silu(self.normalization(combined))


class GlobalOperatorRiskHead(FullBlockMemoryRiskHead):
    """Keep physical/history/denoising/revealed-error inputs identical to CNN."""

    def __init__(self, memory_scale):
        super().__init__("full", memory_scale)
        self.net = nn.Sequential(
            nn.Conv3d(28, 12, 1),
            nn.SiLU(),
            GlobalRiskBlock(),
            GlobalRiskBlock(),
            nn.Conv3d(12, 4, 1),
        )
        assert sum(parameter.numel() for parameter in self.parameters()) == 48304
