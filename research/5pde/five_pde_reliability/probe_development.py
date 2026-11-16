"""Early fixed-checkpoint sanity probe on development trajectories only.

This does not select checkpoints or configure final evaluation.
"""
import argparse
import json
import torch
from common import OUT, Scenario, configure, predict, records
from evaluate import ARMS, MASKS, load_head


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('scenario')
    args = parser.parse_args()
    configure()
    scenario = Scenario(args.scenario)
    rs = [r for r in records(args.scenario) if r['split'] == 'development']
    backbone = scenario.network().eval().requires_grad_(False)
    checkpoint = torch.load(OUT / 'backbones' / args.scenario / 'model.pt', map_location='cuda', weights_only=False)
    backbone.load_state_dict(checkpoint['model'])
    heads = {arm: load_head(args.scenario, arm, 42) for arm in ARMS}
    statistics = {arm: [] for arm in ARMS}
    for start in (0, 200, 400):
        x = scenario.batch(rs, [start] * len(rs))
        point, feature = predict(backbone, x, scenario.mask, with_features=True)
        q = {}
        for arm, (model, _, mean, std) in heads.items():
            mask = torch.tensor(MASKS[arm], device='cuda').expand(len(rs), 2)
            f = ((feature - mean) / std).clamp(-10, 10)
            q[arm] = model.quantiles(x, point, f, mask, torch.tensor([.05, .5, .95], device='cuda'))
        # Reveal the full future only after every method has issued its forecast.
        y = scenario.batch(rs, [start] * len(rs), future=True)
        for arm in ARMS:
            tau = torch.tensor([.05, .5, .95], device='cuda')[:, None, None, None, None, None]
            error = y[None] - q[arm]
            loss = torch.maximum(tau * error, (tau - 1) * error) * scenario.std
            statistics[arm].append(float(loss[:, :, :scenario.channels].mean()))
    result = dict(scenario=args.scenario, split='development', starts=[0, 200, 400], seed=42,
                  used_for_checkpoint_or_hyperparameter_selection=False, pinball=statistics)
    path = OUT / 'development_probe' / f'{args.scenario}.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
