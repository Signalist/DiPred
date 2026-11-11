"""Four-call DiPred with 19 ordered marginal quantiles and online feedback."""
import argparse
from pathlib import Path

import torch
from torch import nn

from .selected16_heads import BASE_INDICES
from .selected16_predict import RequestedThreePredictor


CHECKPOINT = Path(__file__).resolve().parent / 'checkpoints/quantiles19.pt'


class OrderedInteriorHead(nn.Module):
    """Wrap frozen XPF16 endpoints and the learned positive-gap network."""
    def __init__(self, endpoint, checkpoint, device):
        super().__init__()
        self.endpoint = endpoint
        self.gaps = nn.Sequential(
            nn.Conv3d(16, 24, 3, padding=1),
            nn.GroupNorm(4, 24),
            nn.SiLU(),
            nn.Conv3d(24, 24, 3, padding=(1, 2, 2), dilation=(1, 2, 2)),
            nn.GroupNorm(4, 24),
            nn.SiLU(),
            nn.Conv3d(24, 36, 1),
        ).to(device)
        packed = torch.load(checkpoint, map_location=device, weights_only=False)
        self.gaps.load_state_dict(packed['model'], strict=True)
        self.eval().requires_grad_(False)
        self.last_quantiles = None

    def bounds(self, base, last, history, previous_error, available):
        lo, hi = self.endpoint.bounds(base, last, history, previous_error, available)
        batch, _, _, height, width = base.shape
        latent = self.endpoint.feature_projection(last).reshape(batch, 8, 20, height, width)
        y = torch.linspace(-1, 1, height, device=base.device, dtype=base.dtype)
        x = torch.linspace(-1, 1, width, device=base.device, dtype=base.dtype)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        coords = torch.stack((xx, yy))[None, :, None].expand(batch, 2, 20, height, width)
        features = torch.cat((base[:, list(BASE_INDICES)], latent, coords), dim=1)
        logits = self.gaps(features).reshape(batch, 2, 18, 20, height, width)
        fractions = logits.softmax(dim=2).cumsum(dim=2)[:, :, :17]
        inner = lo.unsqueeze(2) + (hi - lo).unsqueeze(2) * fractions
        self.last_quantiles = torch.cat((lo.unsqueeze(2), inner, hi.unsqueeze(2)), dim=2)
        return lo, hi


class Quantiles19Predictor(RequestedThreePredictor):
    """Return marginal 5..95% quantiles; reveal only complete 20-frame blocks."""
    def __init__(self, grid_geometry, device='cuda', quantile_checkpoint=CHECKPOINT):
        super().__init__(grid_geometry, method='selected16', device=device)
        self.head = OrderedInteriorHead(self.head, quantile_checkpoint, self.device)

    @torch.no_grad()
    def predict(self, observed):
        result = super().predict(observed)
        raw = self.head.last_quantiles
        lo, hi = raw[:, :, 0], raw[:, :, -1]
        midpoint = .5 * (lo + hi)
        calibrated_lo, calibrated_hi = self.calibrator.pending[2:4]
        scale = (calibrated_hi - calibrated_lo) / (hi - lo).clamp_min(1e-6)
        calibrated = midpoint.unsqueeze(2) + (raw - midpoint.unsqueeze(2)) * scale.unsqueeze(2)
        point = result['prediction'].unsqueeze(2)
        std = self.norm['std'].unsqueeze(2)
        result['raw_quantiles'] = point + raw * std
        result['quantiles'] = point + calibrated * std
        result['quantiles'][:, :, 0] = result['lower']
        result['quantiles'][:, :, -1] = result['upper']
        result['quantile_levels'] = torch.arange(1, 20, device=self.device) / 20
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--geometry', type=float, nargs=3, required=True, metavar=('DX', 'DY', 'DT'))
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    predictor = Quantiles19Predictor(args.geometry, args.device)
    result = predictor.predict(torch.load(args.input, map_location='cpu', weights_only=False))
    torch.save({k: v.cpu() if torch.is_tensor(v) else v for k, v in result.items()}, args.output)


if __name__ == '__main__':
    main()
