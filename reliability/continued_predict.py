"""Latest frozen quantile head, with the original four-call point predictor.

The extra training changes uncertainty intervals only. The inherited stream
interface admits error feedback only after the entire 20-frame target arrives.
"""

import argparse
from pathlib import Path

import torch

from .memory_predict import MemoryReliabilityPredictor, SHARED_CHECKPOINT, STRATEGIES


CONTINUED_CHECKPOINT = Path(__file__).resolve().parent / "checkpoints/continued.pt"


class ContinuedReliabilityPredictor(MemoryReliabilityPredictor):
    """Joint 20-to-20 forecasts with directly predicted asymmetric intervals."""

    def __init__(self, grid_geometry, strategy="memory_dynamic", device="cuda",
                 checkpoint=SHARED_CHECKPOINT, risk_checkpoint=CONTINUED_CHECKPOINT):
        super().__init__(
            grid_geometry,
            memory_mode="full",
            strategy=strategy,
            device=device,
            checkpoint=checkpoint,
            memory_checkpoint=risk_checkpoint,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--geometry", type=float, nargs=3, required=True,
                        metavar=("DX", "DY", "DT"))
    parser.add_argument("--strategy", choices=STRATEGIES, default="memory_dynamic")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint", type=Path, default=SHARED_CHECKPOINT)
    parser.add_argument("--risk-checkpoint", type=Path, default=CONTINUED_CHECKPOINT)
    args = parser.parse_args()
    model = ContinuedReliabilityPredictor(
        args.geometry, args.strategy, args.device, args.checkpoint, args.risk_checkpoint,
    )
    history = torch.load(args.input, map_location="cpu", weights_only=False)
    result = model.predict(history)
    result = {key: value.cpu() if torch.is_tensor(value) else value
              for key, value in result.items()}
    result.update(strategy=args.strategy, native_call_labels=model.last_call_labels,
                  risk_head="density_quantile")
    torch.save(result, args.output)


if __name__ == "__main__":
    main()
