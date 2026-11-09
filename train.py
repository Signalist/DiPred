"""Train one fixed-budget arm using only manifest['files']['train'].

Raw HDF5 files remain external. This script never opens calibration/dev/test
files. An optional archived fresh seed42 initialization/RNG file can preserve the
original historical initialization; otherwise the standalone constructor starts fresh.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import h5py
import numpy as np
import torch

from model import PhysicalDiffusion20, configure
from runtime import load_normalization


ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--initialization", type=Path, help="Optional archived fresh seed42 initialization, for matching the historical run.")
    parser.add_argument("--arm", choices=("state_diffusion", "delta_diffusion", "state_cnn", "delta_cnn", "state_fourstep", "delta_fourstep"), default="state_diffusion")
    args = parser.parse_args()
    configure()
    if not torch.cuda.is_available():
        raise RuntimeError("The original training protocol requires CUDA; CPU inference remains supported.")
    args.output.mkdir(parents=True, exist_ok=False)
    mf = json.loads((ROOT / "metadata/manifest.json").read_text())
    rows = []
    for record in mf["files"]["train"]:
        with h5py.File(args.data / record["file"], "r") as handle:
            values = np.stack([handle[c][:200, ::2, ::2] for c in ("u", "v")]).astype(np.float32)
        assert hashlib.sha256(values.tobytes()).hexdigest() == record["sha256_first200"]
        rows.append(torch.from_numpy(values))
    raw = torch.stack(rows).cuda()
    assert raw.shape == (47, 2, 200, 32, 64)
    norm = load_normalization("cuda")
    assert torch.allclose(raw.mean((0, 2, 3, 4), keepdim=True), norm["mean"], atol=2e-7)
    assert torch.allclose(raw.std((0, 2, 3, 4), keepdim=True), norm["std"], atol=2e-7)
    fields = (raw - norm["mean"]) / norm["std"]
    histories = torch.cat([fields[:, :, s:s + 20] for s in range(0, 161, 5)])
    futures = torch.cat([fields[:, :, s + 20:s + 40] for s in range(0, 161, 5)])
    del raw, fields
    torch.manual_seed(42)
    predictor = PhysicalDiffusion20().cuda()
    if args.initialization is not None:
        initial = torch.load(args.initialization, map_location="cpu", weights_only=False)
        predictor.load_state_dict(initial["model"], strict=True)
        torch.set_rng_state(initial["cpu_rng_state"])
        torch.cuda.set_rng_state(initial["cuda_rng_state"])
    optimizer = torch.optim.AdamW(predictor.parameters(), lr=3e-4, weight_decay=1e-5)
    protocol = dict(arm=args.arm, seed=42, steps=6000, batch=8, training_pairs=len(histories),
                    fit_files=[r["file"] for r in mf["files"]["train"]], fit_frames=[0, 200],
                    initialization=str(args.initialization) if args.initialization else "Fresh seed42 standalone constructor; historical initializer not loaded.",
                    selection="Fixed endpoint; no validation access.")
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    for step in range(1, 6001):
        ix = torch.randint(len(histories), (8,), device="cuda")
        h, y = histories[ix], futures[ix]
        target = y - h[:, :, -1:] if args.arm.startswith("delta") else y
        lr = 1e-6 + (3e-4 - 1e-6) * .5 * (1 + math.cos(math.pi * (step - 1) / 6000))
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        if args.arm.endswith("diffusion"):
            loss = predictor.native_loss(h, target)
        elif args.arm.endswith("cnn"):
            pred = predictor.net(torch.zeros_like(h), torch.full((8,), 99, device="cuda", dtype=torch.long), h)
            loss = (pred - target).square().mean()
        else:
            pred = predictor.sample(h)
            # Match the original terminal-control arithmetic exactly.
            pred = pred + h[:, :, -1:] if args.arm.startswith("delta") else pred
            loss = (pred - y).square().mean()
        assert torch.isfinite(loss)
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.)
        assert torch.isfinite(gradient)
        optimizer.step()
        if step == 1 or step % 250 == 0:
            print(dict(step=step, loss=float(loss)), flush=True)
    weights = predictor.net.state_dict() if args.arm.endswith("cnn") else predictor.state_dict()
    torch.save(dict(model=weights, **protocol), args.output / "model.pt")


if __name__ == "__main__":
    main()
