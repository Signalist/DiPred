"""Calibrated 129-level XPF forecasts and calibrated 8/32-path ensembles."""
import argparse
from pathlib import Path

import torch

from .calibration import MemoryQuantileCalibrator
from .features import risk_features
from .quantiles19_predict import Quantiles19Predictor
from .quantiles129_head import Quantiles129Head, quantile_levels


HERE = Path(__file__).resolve().parent
QUANTILE_CHECKPOINT = HERE / 'checkpoints/quantiles129.pt'
ENSEMBLE_CHECKPOINT = HERE / 'checkpoints/calibrated_ensembles.pt'


class Quantiles129Predictor(Quantiles19Predictor):
    """Retain original 19 quantile anchors and issue 129 ordered levels."""
    def __init__(self, grid_geometry, device='cuda', quantile_checkpoint=QUANTILE_CHECKPOINT):
        super().__init__(grid_geometry, device)
        self.head129 = Quantiles129Head().to(self.device).eval().requires_grad_(False)
        packed = torch.load(quantile_checkpoint, map_location=self.device, weights_only=False)
        self.head129.load_state_dict(packed['model'], strict=True)

    @torch.no_grad()
    def predict(self, observed):
        features = []
        hook = self.head.gaps.register_forward_pre_hook(lambda module, args: features.append(args[0]))
        try:
            result = super().predict(observed)
        finally:
            hook.remove()
        if len(features) != 1:
            raise RuntimeError('Expected one 19-level head inference.')
        anchors = self.head.last_quantiles
        raw = self.head129(features[0], anchors)
        midpoint = .5 * (anchors[:, :, 0] + anchors[:, :, -1])
        lo, hi = self.calibrator.pending[2:4]
        scale = (hi - lo) / (anchors[:, :, -1] - anchors[:, :, 0]).clamp_min(1e-6)
        adjusted = midpoint.unsqueeze(2) + (raw - midpoint.unsqueeze(2)) * scale.unsqueeze(2)
        adjusted[:, :, 19] = lo
        adjusted[:, :, 109] = hi
        point, std = result['prediction'].unsqueeze(2), self.norm['std'].unsqueeze(2)
        result['raw_quantiles'] = point + raw * std
        result['quantiles'] = point + adjusted * std
        result['quantiles'][:, :, 19] = result['lower']
        result['quantiles'][:, :, 109] = result['upper']
        result['quantile_levels'] = torch.tensor(quantile_levels(), device=self.device)
        return result


class CalibratedThreePredictor(Quantiles129Predictor):
    """Issue XPF and separately calibrated actual 8/32-path ensembles.

    Each sample is scaled around common zero-noise P by the corresponding
    method's online interval-radius factor. Samples are not forced to have P
    as their mean, and the Gaussian proxy interval is not an empirical
    5/95%-sample interval. Complete 20-frame truth is needed before next issue.
    """
    def __init__(self, grid_geometry, device='cuda', ensemble_checkpoint=ENSEMBLE_CHECKPOINT):
        super().__init__(grid_geometry, device)
        self.ensemble_initial = torch.load(ensemble_checkpoint, map_location=self.device, weights_only=False)
        self.ensemble_controls = None

    def reset(self):
        super().reset()
        self.ensemble_controls = None

    @torch.no_grad()
    def predict(self, observed):
        result = super().predict(observed)
        observed = torch.as_tensor(observed, device=self.device, dtype=torch.float32)
        h = (observed - self.norm['mean']) / self.norm['std']
        point = self.pending
        physical = torch.stack([self.model.rollout(h, seed, sign=1) * self.norm['std'] + self.norm['mean']
                                for seed in range(500000, 500032)])
        samples = (physical - self.norm['mean']) / self.norm['std']
        if self.ensemble_controls is None:
            self.ensemble_controls = {name: MemoryQuantileCalibrator('memory_dynamic', self.ensemble_initial[name], len(point))
                                      for name in ('8-noise', '32-noise')}
        for name, count in (('8-noise', 8), ('32-noise', 32)):
            radius = 1.6448536269514722 * (physical[:count].std(0) / self.norm['std']).clamp_min(1e-6)
            lo, hi = self.ensemble_controls[name].issue(-radius, radius, risk_features(h, point, radius))
            scale = (hi - lo) / (2 * radius).clamp_min(1e-6)
            adjusted = point.unsqueeze(0) + (samples[:count] - point.unsqueeze(0)) * scale.unsqueeze(0)
            result[name + '_calibrated_samples'] = (adjusted * self.norm['std'] + self.norm['mean']).permute(1, 0, 2, 3, 4, 5)
            result[name + '_proxy_lower'] = result['prediction'] + lo * self.norm['std']
            result[name + '_proxy_upper'] = result['prediction'] + hi * self.norm['std']
        return result

    @torch.no_grad()
    def reveal(self, truth):
        if self.pending is None or self.ensemble_controls is None:
            raise RuntimeError('No forecast is awaiting full feedback.')
        truth = torch.as_tensor(truth, device=self.device, dtype=torch.float32)
        if truth.shape != self.pending.shape or not torch.isfinite(truth).all():
            raise ValueError('Feedback must contain the complete 20-frame block.')
        residual = (truth - self.norm['mean']) / self.norm['std'] - self.pending
        for controller in self.ensemble_controls.values():
            controller.reveal(residual)
        super().reveal(truth)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--geometry', type=float, nargs=3, required=True, metavar=('DX', 'DY', 'DT'))
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--include-ensembles', action='store_true')
    args = parser.parse_args()
    klass = CalibratedThreePredictor if args.include_ensembles else Quantiles129Predictor
    predictor = klass(args.geometry, args.device)
    result = predictor.predict(torch.load(args.input, map_location='cpu', weights_only=False))
    torch.save({k: v.cpu() if torch.is_tensor(v) else v for k, v in result.items()}, args.output)


if __name__ == '__main__':
    main()
