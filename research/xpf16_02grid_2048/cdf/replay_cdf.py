"""Recompute the complete six-method test CDF from repository artifacts.

Example: python research/xpf16_02grid_2048/cdf/replay_cdf.py --data-dir /data/train_real --weights-dir /tmp/xpf_grid_weights --output /tmp/cdf
Requires all 250 published independent weights, the source HDF5 files, and a
CUDA device. No output is written into the input dataset.
"""
import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'research/xpf16_independent49'))
from reliability.grid02_predict import Grid02Predictor
from reliability.calibration import MemoryQuantileCalibrator
from reliability.features import risk_features
from batched_noise import sample_batched
from cdf_stream import CDFAccumulator


def load_fields(data_dir, records, start, device):
    rows = []
    for record in records:
        with h5py.File(data_dir / record['file'], 'r') as source:
            field = np.stack([source[c][start:start+20, ::2, ::2] for c in ('u', 'v')]).astype(np.float32)
        if field.shape != (2, 20, 32, 64):
            raise ValueError('Unexpected source field shape.')
        rows.append(torch.from_numpy(field))
    return torch.stack(rows).to(device)


def geometry(data_dir, records, device):
    rows = []
    for record in records:
        with h5py.File(data_dir / record['file'], 'r') as source:
            x, y = source['x'][::2, ::2], source['y'][::2, ::2]
            rows.append([float(np.median(np.diff(x, axis=1))), float(np.median(np.diff(y, axis=0))),
                         float(np.median(np.diff(source['t'][:21])))])
    return torch.tensor(rows, device=device, dtype=torch.float32)


def to_device(state, device):
    return {key: value.to(device) for key, value in state.items()}


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--weights-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    if args.output.resolve() == args.data_dir.resolve() or args.data_dir.resolve() in args.output.resolve().parents:
        raise ValueError('Output must be outside the read-only dataset.')
    device = torch.device(args.device)
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    records = json.loads((REPO/'metadata/manifest.json').read_text())['files']['test']
    predictor = Grid02Predictor(geometry(args.data_dir, records, device), args.weights_dir, device=str(device))
    norm = predictor.norm
    common = torch.load(REPO/'reliability/checkpoints/calibrated_ensembles.pt', map_location=device, weights_only=False)
    initial = {n:to_device(common[f'{n}-noise'], device) for n in (8,32)}
    initial[256] = torch.load(REPO/'results/reliability/xpf16_independent49/noise256_calibration.pt', map_location=device, weights_only=False)
    initial[2048] = torch.load(REPO/'results/reliability/xpf16_02grid_2048/calibration.pt', map_location=device, weights_only=False)['proxy']
    controls = {n:MemoryQuantileCalibrator('memory_dynamic', initial[n], len(records)) for n in (8,32,256,2048)}
    accumulator = CDFAccumulator(device)
    for start in range(0, 821, 20):
        observed = load_fields(args.data_dir, records, start, device)
        forecast = predictor.predict(observed)
        h = (observed - norm['mean']) / norm['std']
        # Reuse the exact frozen predictor output that was actually issued.
        point = predictor.pending
        q = ((forecast['calibrated_quantiles'].transpose(0,1) - forecast['prediction'].unsqueeze(0)) / norm['std'])
        paths = torch.empty((2048, *h.shape), dtype=h.dtype, device=device)
        for index in range(0, 2048, 2):
            paths[index:index+2].copy_(sample_batched(predictor.model, h, (500000+index,500001+index)))
        scales = {}
        for count, controller in controls.items():
            radius = 1.6448536269514722 * paths[:count].std(0).clamp_min(1e-6)
            low, high = controller.issue(-radius, radius, risk_features(h, point, radius))
            scales[count] = (high-low) / (2*radius).clamp_min(1e-6)
        # All six forecasts are frozen before the future window is read.
        truth = load_fields(args.data_dir, records, start+20, device)
        residual = (truth-norm['mean']) / norm['std'] - point
        accumulator.add(start, q, paths, point, residual, scales, norm['std'])
        predictor.reveal(truth)
        for controller in controls.values():
            controller.reveal(residual)
        print('CDF_WINDOW', start, flush=True)
    accumulator.save(args.output, records, norm)
    print('FULL_TEST_CDF_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
