"""Portable16-head, original28-head and actual32-noise uncertainty inference.

All methods share the frozen original four-call joint20-frame point forecast.
The32-noise method adds32actual sampled paths, with a common zero-noise center.
"""
import argparse
from pathlib import Path
import torch

from .dynamic_predict import DynamicRiskMemoryPredictor
from .memory_predict import SHARED_CHECKPOINT
from .selected16_heads import Selected16QuantileHead

CHECKPOINT = Path(__file__).resolve().parent / 'checkpoints/selected16_comparison.pt'
METHODS = ('selected16', 'original28', 'noise32')


class Noise32Spread:
    """Estimate uncertainty from the original32 actual sampler paths."""

    def __init__(self, model, normalization):
        self.model = model
        self.norm = normalization
        self.last_physical_std = None

    @torch.no_grad()
    def bounds(self, base, last, history, previous_error, available):
        paths = torch.stack([
            self.model.rollout(history, seed, sign=1) * self.norm['std'] + self.norm['mean']
            for seed in range(500000, 500032)
        ])
        physical_std = paths.std(dim=0)
        self.last_physical_std = physical_std.clone()
        radius = 1.6448536269514722 * (physical_std / self.norm['std']).clamp_min(1e-6)
        return -radius, radius


class RequestedThreePredictor(DynamicRiskMemoryPredictor):
    """Reuse the original complete-window predict/reveal/reset safeguards."""

    def __init__(self, grid_geometry, method='selected16', device='cuda',
                 checkpoint=SHARED_CHECKPOINT, risk_checkpoint=CHECKPOINT):
        if method not in METHODS:
            raise ValueError(f'Choose a method in {METHODS}.')
        super().__init__(grid_geometry, objective='warp', strategy='memory_dynamic',
                         device=device, checkpoint=checkpoint)
        packed = torch.load(risk_checkpoint, map_location=self.device, weights_only=False)
        if packed['format_version'] != 1:
            raise ValueError('Unsupported selected16 checkpoint format.')
        if method == 'selected16':
            head = Selected16QuantileHead(self.head.state_dict())
            head.load_state_dict(packed['heads'][method], strict=True)
            self.head = head.to(self.device).eval().requires_grad_(False)
        elif method == 'original28':
            # Original input architecture, with the matched continuation weights.
            self.head.load_state_dict(packed['heads'][method], strict=True)
            self.head.eval().requires_grad_(False)
        else:
            self.head = Noise32Spread(self.model, self.norm)
        self.initial = packed['calibration'][method]
        self.method = method
        self.reset()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--geometry', type=float, nargs=3, required=True, metavar=('DX', 'DY', 'DT'))
    parser.add_argument('--method', choices=METHODS, default='selected16')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--checkpoint', type=Path, default=SHARED_CHECKPOINT)
    parser.add_argument('--risk-checkpoint', type=Path, default=CHECKPOINT)
    args = parser.parse_args()
    model = RequestedThreePredictor(args.geometry, args.method, args.device, args.checkpoint, args.risk_checkpoint)
    result = model.predict(torch.load(args.input, map_location='cpu', weights_only=False))
    result = {key: value.cpu() if torch.is_tensor(value) else value for key, value in result.items()}
    result.update(method=args.method, native_point_labels=model.last_call_labels,
                  actual_backbone_calls_per_batch=132 if args.method == 'noise32' else 4)
    torch.save(result, args.output)


if __name__ == '__main__':
    main()
