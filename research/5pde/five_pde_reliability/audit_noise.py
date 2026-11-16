"""Audit that the native 8/32 ensembles are prefixes of the 256 paths."""
import argparse
import json
import torch
from common import OUT, Scenario, configure, records, sample_predict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('scenario')
    name = parser.parse_args().scenario
    configure()
    scenario = Scenario(name)
    record = next(r for r in records(name) if r['split'] == 'development')
    x = scenario.batch([record], [0])
    model = scenario.network().eval().requires_grad_(False)
    checkpoint = torch.load(OUT / 'backbones' / name / 'model.pt', map_location='cuda', weights_only=False)
    model.load_state_dict(checkpoint['model'])
    paths = sample_predict(model, x, scenario.mask, 256)
    differences = {}
    for count in (8, 32):
        replay = sample_predict(model, x, scenario.mask, count)
        differences[str(count)] = float((replay - paths[:count]).abs().max())
    passed = bool(torch.isfinite(paths).all()) and all(v <= 1e-6 for v in differences.values())
    result = dict(scenario=name, development_trajectory=record['id'], history_start=0,
                  first_seed=500000, chunk_size=4, prefix_max_absolute_error=differences,
                  passed=passed)
    path = OUT / 'noise_audit' / (name + '.json')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)
    if not passed:
        raise RuntimeError('Noise prefix audit failed')


if __name__ == '__main__':
    main()
