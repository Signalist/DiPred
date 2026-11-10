"""Portable four-call DiPred with frozen learned risk-memory movement."""

import argparse
from pathlib import Path
import torch

from .continued_predict import ContinuedReliabilityPredictor
from .dynamic_heads import DynamicRiskMemoryHead
from .memory_predict import SHARED_CHECKPOINT, STRATEGIES

DYNAMIC_CHECKPOINT = Path(__file__).resolve().parent / 'checkpoints/dynamic_memory.pt'
OBJECTIVES = ('warp', 'modulate')


class DynamicRiskMemoryPredictor(ContinuedReliabilityPredictor):
    """Change intervals only; inherit complete-window predict/reveal guards."""

    def __init__(self, grid_geometry, objective='warp', strategy='memory_dynamic',
                 device='cuda', checkpoint=SHARED_CHECKPOINT,
                 risk_checkpoint=DYNAMIC_CHECKPOINT):
        if objective not in OBJECTIVES:
            raise ValueError(f'Choose an objective in {OBJECTIVES}.')
        super().__init__(grid_geometry, strategy=strategy, device=device, checkpoint=checkpoint)
        packed = torch.load(risk_checkpoint, map_location=self.device, weights_only=False)
        if packed['format_version'] != 1:
            raise ValueError('Unsupported dynamic risk-memory checkpoint format.')
        # The previous published head supplies the frozen 48,340-parameter core.
        head = DynamicRiskMemoryHead(packed['memory_scale'], self.head.state_dict(), objective)
        head.load_state_dict(packed['heads'][objective], strict=True)
        self.head = head.to(self.device).eval().requires_grad_(False)
        self.initial = packed['calibration'][objective]
        self.objective = objective
        self.reset()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--geometry', type=float, nargs=3, required=True, metavar=('DX', 'DY', 'DT'))
    parser.add_argument('--objective', choices=OBJECTIVES, default='warp')
    parser.add_argument('--strategy', choices=STRATEGIES, default='memory_dynamic')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--checkpoint', type=Path, default=SHARED_CHECKPOINT)
    parser.add_argument('--risk-checkpoint', type=Path, default=DYNAMIC_CHECKPOINT)
    args = parser.parse_args()
    model = DynamicRiskMemoryPredictor(args.geometry, args.objective, args.strategy,
                                       args.device, args.checkpoint, args.risk_checkpoint)
    result = model.predict(torch.load(args.input, map_location='cpu', weights_only=False))
    result = {key: value.cpu() if torch.is_tensor(value) else value for key, value in result.items()}
    result.update(objective=args.objective, strategy=args.strategy, native_call_labels=model.last_call_labels)
    torch.save(result, args.output)


if __name__ == '__main__':
    main()
