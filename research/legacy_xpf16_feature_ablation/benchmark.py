"""Benchmark complete four-step predictor and all 49 heads after training.

Uses known historical inputs only and excludes file I/O and scoring. Run on one
GPU without concurrent training; report both single-trajectory latency and
12-trajectory batch latency rather than dividing a batch into fictitious latency.
"""
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT/'xpf16_fullpercent_study'))
sys.path.insert(0, str(ROOT/'github/DiPred'))
from clean_reliability_common import configure, manifest
from experiment_reliability_transport_e12 import geometry
from reliability.selected16_predict import RequestedThreePredictor
from evaluate49 import get_observed, prepare
from calibration import TwoTailCalibrator
from train49 import configs
from evaluate import OUT, OLD, models_for, all_bounds


@torch.no_grad()
def main():
    configure()
    records = manifest()['files']['test']
    results = []
    for arm in ('X', 'P', 'F', 'XP', 'XF', 'PF', 'XPF'):
        heads, groups = models_for(arm)
        predictor = RequestedThreePredictor(geometry(records), 'selected16', 'cuda')
        observed = get_observed('test', 20, records)
        init_path = OLD/'initial_offsets.pt' if arm == 'XPF' else OUT/'calibration'/f'{arm}.pt'
        initial = torch.load(init_path, map_location='cuda', weights_only=False)
        full_geometry = predictor.geometry
        for n in (1, 12):
            predictor.geometry = full_geometry[:n]
            controls = [TwoTailCalibrator(c['lower'], a, n) for c, a in zip(configs(), initial)]
            times = []
            torch.cuda.reset_peak_memory_stats()
            for iteration in range(6):
                torch.cuda.synchronize()
                start = time.perf_counter()
                _, point, base, last = prepare(predictor, observed[:n])
                pairs = all_bounds(groups, base, last)
                online = [c.issue(*pair) for c, pair in zip(controls, pairs)]
                torch.cuda.synchronize()
                seconds = time.perf_counter() - start
                for c in controls:
                    c.pending = None
                if iteration:
                    times.append(seconds)
                del point, base, last, pairs, online
            results.append(dict(method=arm, batch_size=n, warmups=1, repetitions=5,
                mean_seconds=float(np.mean(times)), median_seconds=float(np.median(times)),
                peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
                scope='4-step DiPred + all 49 heads + online issue; no I/O or score',
                note='single GPU after completion of the new-head training'))
        del predictor, groups, heads
        torch.cuda.empty_cache()
        print('BENCHMARK', arm, flush=True)
    path=OUT/'publication'/'timing_benchmark.csv'
    pd.DataFrame(results).to_csv(path,index=False)
    print('BENCHMARK_COMPLETE', path, flush=True)


if __name__ == '__main__':
    main()
