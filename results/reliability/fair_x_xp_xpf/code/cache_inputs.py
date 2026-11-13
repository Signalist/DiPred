"""Cache only known history and genuine frozen-DiPred predictions/features."""
import json
import os
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent.parent
for path in (ROOT, ROOT / 'xpf16_fullpercent_study', ROOT / 'github/DiPred'):
    sys.path.insert(0, str(path))
from clean_reliability_common import configure, manifest
from experiment_reliability_transport_e12 import geometry, history
from export_aligned_test_distribution import from_hdf
from evaluate49 import prepare
from reliability.selected16_predict import RequestedThreePredictor

OUT = ROOT / 'runs/fair_x_xp_xpf_20261113'


def fields(split, start, records, future=False):
    from experiment_reliability_transport_e12 import target
    if start <= 760:
        return target(split, start) if future else history(split, start)
    return from_hdf(records, start + (20 if future else 0))


@torch.no_grad()
def main():
    configure()
    for split, starts in (('calibration', range(0, 161, 20)),
                          ('development', range(0, 161, 20)),
                          ('test', range(0, 821, 20))):
        records = manifest()['files'][split]
        predictor = RequestedThreePredictor(geometry(records), 'selected16', 'cuda')
        for start in starts:
            path = OUT / 'inputs' / split / f'{start:04d}.pt'
            if path.exists():
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            # No target is read: P and final denoising F use only observed X.
            x, point, base, feature = prepare(predictor, fields(split, start, records))
            pack = dict(x=x.cpu(), p=base[:, :2].cpu(), f=feature.cpu(),
                        start=start, records=records)
            tmp = path.with_suffix('.tmp')
            torch.save(pack, tmp)
            os.replace(tmp, path)
            print('CACHED', split, start, flush=True)
    (OUT / 'inputs_complete.json').write_text(json.dumps({'complete': True}) + '\n')


if __name__ == '__main__':
    main()
