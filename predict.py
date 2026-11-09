"""Inference on physical history tensors; never reads future truth."""

import argparse
import json
from pathlib import Path

import torch

from model import PhysicalDiffusion20, configure
from runtime import load_normalization


ROOT = Path(__file__).resolve().parent


@torch.no_grad()
def predict(history, arm="state_diffusion", samples=0, device="cuda", weights_dir=None):
    configure()
    if samples and not arm.endswith("diffusion"):
        raise ValueError("Sampling is available only for diffusion-trained checkpoints.")
    if samples == 1 or samples < 0:
        raise ValueError("Use samples=0 for a point forecast, or >=2 for a sample ensemble.")
    model = PhysicalDiffusion20().to(device).eval()
    directory = Path(weights_dir) if weights_dir is not None else ROOT / "weights"
    checkpoint = torch.load(directory / (arm + ".pt"), map_location=device, weights_only=False)
    if arm.endswith("cnn"):
        model.net.load_state_dict(checkpoint["model"], strict=True)
    else:
        model.load_state_dict(checkpoint["model"], strict=True)
    norm = load_normalization(device)
    history = history.to(device=device, dtype=torch.float32)
    if history.ndim != 5 or history.shape[1:] != (2, 20, 32, 64):
        raise ValueError("History must have shape [B,2,20,32,64] in original data units.")
    h = (history - norm["mean"]) / norm["std"]
    anchor = h[:, :, -1:] if arm.startswith("delta") else 0.

    def path(seed):
        if arm.endswith("cnn"):
            value = model.net(torch.zeros_like(h), torch.full((len(h),), 99, device=device, dtype=torch.long), h)
        else:
            value = model.sample(h, seed)
        return (value + anchor) * norm["std"] + norm["mean"]

    if not samples:
        return dict(mean=path(None).cpu(), arm=arm, mode="zero_noise4" if not arm.endswith("cnn") else "one_call",
                    warning="Existing ensemble-centered interval calibration does not apply to this point center.")
    fields = torch.stack([path(seed) for seed in range(500000, 500000 + samples)])
    mean, sd = fields.mean(0), fields.std(0)
    output = dict(mean=mean.cpu(), std=sd.cpu(), arm=arm, mode="ensemble", samples=samples)
    if samples == 32:
        # This is a fixed, marginal calibration. No simultaneous-field guarantee.
        factors = json.loads((ROOT / "metadata/calibration_locked.json").read_text())[arm]
        scale = torch.tensor(factors, device=device).reshape(1, 2, 1, 1, 1)
        radius = scale * sd.clamp_min(norm["std"] * 1e-4)
        output.update(lower90=(mean - radius).cpu(), upper90=(mean + radius).cpu(),
                      interval_scope="Fixed calibrated marginal intervals; not simultaneous coverage.")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "prediction.pt")
    parser.add_argument("--arm", choices=("state_diffusion", "delta_diffusion", "state_cnn", "delta_cnn", "state_fourstep", "delta_fourstep"), default="state_diffusion")
    parser.add_argument("--samples", type=int, default=0)
    parser.add_argument("--weights-dir", type=Path, default=ROOT / "weights")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    history = torch.load(args.input, map_location="cpu", weights_only=False)
    result = predict(history, args.arm, args.samples, args.device, args.weights_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
