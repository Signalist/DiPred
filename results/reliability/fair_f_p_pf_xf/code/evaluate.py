"""Causal calibration and full held-out trajectory evaluation for fair heads."""
import json
import math
import os
import sys
import time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
for sub in ('', 'fair_x_xp_xpf', 'xpf16_fullpercent_study', 'xpf16_02grid_2048', 'xpf16_cdf_fulltest', 'github/DiPred'):
    sys.path.insert(0, str(ROOT / sub))
from fair_x_xp_xpf.model import FairQuantileHead
from cache_inputs import fields
from clean_reliability_common import configure, manifest
from experiment_reliability_transport_e12 import normalization
from calibration_grid import GridCalibrator, MedianCalibrator, per_level_quantiles
from distribution_grid import rearrange_quantiles, quantile_distribution_scores
from cdf_stream import _add_at

OUT = ROOT / 'runs/fair_f_p_pf_xf_20261116'
INPUTS = ROOT / 'runs/fair_x_xp_xpf_20261113/inputs'
LEVELS = torch.arange(1, 500, dtype=torch.float64) / 500
ALPHA = LEVELS[:249]
ARMS = ('F', 'P', 'PF', 'XF')
MASKS = {'F': (0., 1.), 'P': (1., 0.), 'PF': (1., 1.), 'XF': (0., 1.)}
SEEDS = (42, 123, 2026)


def load_model(arm, seed):
    state = torch.load(OUT / f'seed_{seed}' / f'{arm}.pt', map_location='cpu', weights_only=False)
    model = FairQuantileHead().cuda().eval().requires_grad_(False)
    model.load_state_dict(state['model'])
    return model, state


def input_batch(split, start, arm):
    value = torch.load(INPUTS / split / f'{start:04d}.pt', map_location='cuda', weights_only=False)
    x = value['x']; p = value['p']; f = value['f']
    # Missing slots are explicitly zeroed before entering the common model.
    if arm != 'XF': x = torch.zeros_like(x)
    if arm in ('F', 'XF'): p = torch.zeros_like(p)
    if arm == 'P': f = torch.zeros_like(f)
    mask = torch.tensor(MASKS[arm], device='cuda').expand(len(x), 2)
    return x, p, f, mask


def pair_quantiles(blocks, levels):
    value = torch.cat(blocks, dim=1).permute(0, 2, 1, 3, 4, 5).flatten(2)
    result = []
    for i in range(0, 249, 8):
        result.append(per_level_quantiles(value[i:i+8].cuda(), levels[i:i+8].cuda()).cpu())
    return torch.cat(result)


@torch.no_grad()
def calibrate(model, arm, seed, norm):
    path = OUT / 'calibration' / f'{arm}_{seed}.pt'
    if path.exists():
        return torch.load(path, map_location='cuda', weights_only=False)
    records = manifest()['files']['calibration']
    left, right, scores, medians = [], [], [], []
    for start in range(0, 161, 20):
        x, p, f, mask = input_batch('calibration', start, arm)
        q = model.quantiles(x, p, f, mask, LEVELS.cuda().float())
        # Forecast is complete before loading the next 20-frame block.
        y = (fields('calibration', start, records, True) - norm['mean']) / norm['std']
        lo, hi = q[:249], q[250:].flip(0)
        left.append((y[None] - lo)[..., ::4, ::4].cpu())
        right.append((y[None] - hi)[..., ::4, ::4].cpu())
        scores.append(torch.maximum(lo - y[None], y[None] - hi)[..., ::4, ::4].cpu())
        medians.append((y - q[249])[..., ::4, ::4].cpu())
        del q, lo, hi
    initial = torch.stack((pair_quantiles(left, ALPHA), pair_quantiles(right, 1 - ALPHA)), 1)
    value = torch.cat(scores, dim=1).permute(0, 2, 1, 3, 4, 5).flatten(2)
    n = value.shape[-1]
    fixed = []
    for i in range(249):
        rank = math.ceil((n + 1) * (1 - 2 * float(ALPHA[i])))
        if rank > n: raise ValueError('Insufficient CQR calibration observations')
        fixed.append(value[i].cuda().sort(dim=-1).values[:, rank - 1].cpu())
    median = torch.cat(medians, 0).permute(1, 0, 2, 3, 4).flatten(1).median(dim=1).values
    result = dict(initial=initial, fixed=torch.stack(fixed), median=median, n=n,
                  note='pooled correlated spatial-temporal observations; no trajectory-level conformal guarantee')
    path.parent.mkdir(exist_ok=True, parents=True)
    torch.save(result, path)
    print('CALIBRATED', arm, seed, n, flush=True)
    return torch.load(path, map_location='cuda', weights_only=False)


def sorted_cdf_sum(q, probs, grid, chunk=8192):
    """CDF sum for already rearranged quantiles; uses endpoint atoms."""
    n=len(grid)
    slope=torch.zeros(n+1,device=q.device,dtype=torch.float64)
    intercept=torch.zeros_like(slope)
    probs=probs.to(device=q.device,dtype=torch.float64)
    for start in range(0,q.shape[1],chunk):
        values=q[:,start:start+chunk].to(torch.float64)
        left,right=values[:-1],values[1:]
        weights=(probs[1:]-probs[:-1])[:,None].expand_as(left)
        equal=right==left
        local=torch.where(equal,torch.zeros_like(left),weights/(right-left).clamp_min(1e-300))
        a,b=torch.bucketize(left.contiguous(),grid),torch.bucketize(right.contiguous(),grid)
        _add_at(slope,a,local);_add_at(slope,b,-local)
        _add_at(intercept,a,-local*left+weights*equal);_add_at(intercept,b,local*right)
        ones=torch.ones(values.shape[1],device=q.device,dtype=torch.float64)
        _add_at(intercept,torch.bucketize(values[0].contiguous(),grid),ones*probs[0])
        _add_at(intercept,torch.bucketize(values[-1].contiguous(),grid),ones*(1-probs[-1]))
    return slope[:-1].cumsum(0)*grid+intercept[:-1].cumsum(0)


def quantile_crps(q, y, levels):
    """Trapezoidal 2*pinball integral with endpoint-clamped quantiles."""
    total = .5 * ((q[0]-y).clamp_min(0) + (y-q[-1]).clamp_min(0))
    for i in range(0, len(levels), 16):
        error = y[None] - q[i:i+16]
        t = levels[i:i+16, None, None, None, None, None]
        total = total + torch.maximum(t*error, (t-1)*error).sum(0)
    return total * .004


def bounds(q, cal, controller, median_controller):
    lo, hi = q[:249], q[250:].flip(0)
    c = cal['fixed'][:, None, :, None, None, None]
    a, b = lo - c, hi + c
    invalid = a > b
    mid = (a + b) / 2
    a, b = torch.where(invalid, mid, a), torch.where(invalid, mid, b)
    fixed = torch.cat((a, q[249:250], b.flip(0)))
    online_invalid = (lo + controller.offsets[:,:,0]) > (hi + controller.offsets[:,:,1])
    online_projected = float(online_invalid.float().mean())
    a, b = controller.issue(lo, hi)
    center = median_controller.issue(q[249])
    online = torch.cat((a, center[None], b.flip(0)))
    return (q, fixed, online), (0., float(invalid.float().mean()), online_projected)


@torch.no_grad()
def run_model(arm, seed, norm):
    path = OUT / 'evaluated' / f'{arm}_{seed}.pt'
    if path.exists():
        print('ALREADY_EVALUATED', arm, seed, flush=True)
        return
    model, state = load_model(arm, seed)
    cal = calibrate(model, arm, seed, norm)
    records = manifest()['files']['test']; count = len(records)
    controller = GridCalibrator(ALPHA.cuda(), cal['initial'], count)
    median = MedianCalibrator(cal['median'], count)
    levels = LEVELS.cuda().float(); grid = torch.linspace(-.35, .35, 701, device='cuda', dtype=torch.float64)
    cdfs = torch.zeros(3, 2, len(grid), device='cuda', dtype=torch.float64)
    qcoverage = torch.zeros(3, 499, 2, device='cuda', dtype=torch.float64)
    qpinball = torch.zeros_like(qcoverage)
    window, frame, snapshots, events = [], [], [], []
    observations = 0
    for start in range(0, 821, 20):
        x, p, f, mask = input_batch('test', start, arm)
        q = model.quantiles(x, p, f, mask, levels)
        predictions, projected = bounds(q, cal, controller, median)
        events.append(dict(event='issued', start=start, time=time.time()))
        # Reveal only after every raw/fixed/online full-window prediction is issued.
        y = (fields('test', start, records, True) - norm['mean']) / norm['std']
        events.append(dict(event='revealed', start=start, time=time.time()))
        if start:
            for mode, original in enumerate(predictions):
                if mode == 0 and (original[1:] < original[:-1]).any():
                    raise ValueError('Raw quantiles violate architecture monotonicity')
                if mode == 0:
                    ordered=original
                    crossing=torch.zeros_like(y)
                    any_crossing=torch.zeros_like(y)
                else:
                    ordered, crossing, any_crossing = rearrange_quantiles(original)
                # All scores below use a monotonic quantile function. Raw is already monotone.
                score = quantile_crps(ordered, y, levels)
                scale_field = norm['std']
                score = score * scale_field
                low, high = ordered[24], ordered[474]
                coverage = ((y >= low) & (y <= high)).float()
                width = (high - low) * scale_field
                interval = width + 20 * ((low-y).clamp_min(0) + (y-high).clamp_min(0)) * scale_field
                per_frame = score.mean((-1, -2)).cpu().numpy()
                values = torch.stack((score.mean((2,3,4)), coverage.mean((2,3,4)),
                                      width.mean((2,3,4)), interval.mean((2,3,4)),
                                      crossing.mean((2,3,4)), any_crossing.mean((2,3,4))), -1).cpu().numpy()
                window.append(dict(mode=mode, start=start, values=values, projected=projected[mode]))
                frame.append(dict(mode=mode, start=start, values=per_frame))
                for i in range(0, 499, 8):
                    z = ordered[i:i+8]
                    e = y[None] - z
                    t = levels[i:i+8, None, None, None, None, None]
                    qcoverage[mode,i:i+8] += (y[None] <= z).double().mean((1,3,4,5))
                    qpinball[mode,i:i+8] += (torch.maximum(t*e, (t-1)*e) * scale_field).double().mean((1,3,4,5))
                for channel in range(2):
                    scale = norm['std'][0, channel, 0, 0, 0]
                    errors = ((ordered[:,:,channel] - y[:,channel][None]) * scale).flatten(1)
                    cdfs[mode,channel] += sorted_cdf_sum(errors, LEVELS, grid)
                if start == 20:
                    # Tiny, genuinely aligned sample to independently recompute scores.
                    snapshots.append(dict(mode=mode, q=ordered[:,0,:,::19,4::12,8::24].cpu(),
                                          y=y[0,:,::19,4::12,8::24].cpu(),
                                          crps=score[0,:,::19,4::12,8::24].cpu()))
                del ordered, crossing, any_crossing, score
            observations += y[:,0].numel()
        controller.reveal(y)
        median.reveal(y)
        del predictions, q, y
        print('EVALUATED', arm, seed, start, flush=True)
    assert controller.updates == 42 and median.updates == 42
    path.parent.mkdir(exist_ok=True, parents=True)
    result = dict(arm=arm, seed=seed, windows=window, frames=frame, snapshots=snapshots, events=events,
                  grid=grid.cpu(), cdf=(cdfs/observations).cpu(), observations=observations,
                  qcoverage=(qcoverage/41).cpu(), qpinball=(qpinball/41).cpu(),
                  records=records, parameters=state['parameters'], train_seconds_group=state['group_wall_seconds'])
    tmp=path.with_suffix('.tmp');torch.save(result,tmp);os.replace(tmp,path)


def main():
    configure(); norm=normalization()
    assert (OUT/'training_complete.json').exists() and (INPUTS.parent/'inputs_complete.json').exists()
    for seed in SEEDS:
        for arm in ARMS:
            run_model(arm, seed, norm)
    (OUT/'evaluation_complete.json').write_text(json.dumps({'complete': True})+'\n')


if __name__ == '__main__': main()
