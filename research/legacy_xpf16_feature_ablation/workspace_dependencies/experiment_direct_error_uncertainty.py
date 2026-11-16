"""Direct Gaussian/quantile uncertainty versus frozen32-path spread.

The common point center is the four-call zero-noise Direct20 forecast. Heads fit
only on the training trajectories'200:400 time holdout. No predictor retraining.
"""

import json
import math
from pathlib import Path
import time

import torch

from clean_reliability_common import ROOT, configure, digest, dump, manifest
from experiment_reliability_transport_e12 import (
    FIT_RECORDS, FIT_STARTS, PREDICTOR, history, target, normalization, issued,
)
from experiment_physical20_augmented_dynamics import new_model
from direct_error_uncertainty import MODES, DirectErrorHead, forecast_features, inputs
from evaluate_reliability_transport_e12 import ranking, summarize
from experiment_reliability_structured_e3 import grouped_quantile


OUT = ROOT / "runs/direct_error_uncertainty_20261109"
VARIANTS = ("raw", "fixed", "online_global", "online_lead")


def event(**values):
    with (OUT / "events.jsonl").open("a") as stream:
        stream.write(json.dumps(dict(time=time.time(), **values)) + "\n")


def cached(split, start):
    return torch.load(OUT / "features" / f"{split}_{start}.pt", map_location="cuda", weights_only=False)


@torch.no_grad()
def cache(model, norm, split, starts):
    folder = OUT / "features"
    folder.mkdir(exist_ok=True)
    for start in starts:
        h = (history(split, start) - norm["mean"]) / norm["std"]
        begin = time.perf_counter()
        p, features = forecast_features(model, h)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - begin
        assert torch.isfinite(features).all() and torch.isfinite(p).all()
        value = dict(prediction=p.cpu(), features=features.half().cpu(), seconds=elapsed)
        # Compare to already issued zero-noise forecasts where available.
        difference = None
        if split in ("development", "test"):
            reference = issued("diffusion", split, start)["zero"]
            physical = p * norm["std"] + norm["mean"]
            difference = float((physical - reference).abs().max())
            assert difference < 1e-6
        torch.save(value, folder / f"{split}_{start}.pt")
        event(event="inference_features_cached", split=split, start=start,
              calls=4, future_truth_read=False, zero_reference_max_error=difference)
    print("CACHE", split, flush=True)


def train(norm):
    histories, predictions, features, residuals = [], [], [], []
    for start in FIT_STARTS:
        value = cached("fit", start)
        histories.append((history("fit", start) - norm["mean"]) / norm["std"])
        predictions.append(value["prediction"])
        features.append(value["features"].float())
        # Labels are loaded after the actual forecast and internal features exist.
        residuals.append((target("fit", start) - norm["mean"]) / norm["std"] - value["prediction"])
    h, p, f, r = [torch.cat(values) for values in (histories, predictions, features, residuals)]
    del histories, predictions, features, residuals
    reference = torch.cat([h.reshape(len(h), 40, 32, 64), p.reshape(len(p), 40, 32, 64), f], 1)
    stats = dict(mean=reference.mean((0, 2, 3), keepdim=True),
                 std=reference.std((0, 2, 3), keepdim=True).clamp_min(.05))
    del reference
    torch.save({key: value.cpu() for key, value in stats.items()}, OUT / "statistics.pt")
    parameters = {}
    for mode in MODES:
        value = inputs(h, p, f, stats, mode)
        torch.manual_seed(42)
        head = DirectErrorHead(mode).cuda()
        parameters[mode] = sum(p.numel() for p in head.parameters())
        optimizer = torch.optim.AdamW(head.parameters(), lr=3e-4, weight_decay=1e-5)
        logs = []
        begin = time.time()
        for step in range(1, 2001):
            ix = torch.randint(len(value), (8,), device="cuda")
            rate = 1e-6 + (3e-4 - 1e-6) * .5 * (1 + math.cos(math.pi * (step - 1) / 2000))
            for group in optimizer.param_groups:
                group["lr"] = rate
            optimizer.zero_grad(set_to_none=True)
            loss = head.loss(value[ix], r[ix])
            assert torch.isfinite(loss)
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(head.parameters(), 1.)
            assert torch.isfinite(gradient)
            optimizer.step()
            if step == 1 or step % 500 == 0:
                logs.append(dict(step=step, loss=float(loss), elapsed=time.time() - begin))
                print("TRAIN", mode, logs[-1], flush=True)
        torch.save(dict(model=head.state_dict(), mode=mode, steps=2000, seed=42), OUT / f"{mode}.pt")
        dump(OUT / f"{mode}_training.json", logs)
        del value, head, optimizer
    dump(OUT / "heads_frozen.json", dict(time=time.time(), parameters=parameters,
         artifacts={path.name: digest(path) for path in OUT.glob("*.pt")}))


class Provider:
    def __init__(self, norm):
        self.norm = norm
        self.stats = torch.load(OUT / "statistics.pt", map_location="cuda", weights_only=False)
        self.heads = {}
        for mode in MODES:
            head = DirectErrorHead(mode).cuda().eval().requires_grad_(False)
            head.load_state_dict(torch.load(OUT / f"{mode}.pt", map_location="cuda", weights_only=False)["model"])
            self.heads[mode] = head

    @torch.no_grad()
    def bounds(self, split, start):
        value = cached(split, start)
        h = (history(split, start) - self.norm["mean"]) / self.norm["std"]
        p = value["prediction"]
        result = {mode: head.bounds(inputs(h, p, value["features"], self.stats, mode))
                  for mode, head in self.heads.items()}
        # Sampling baseline is recentered on the SAME zero-noise point prediction.
        sigma = issued("diffusion", split, start)["std32"] / self.norm["std"]
        radius = 1.6448536269514722 * sigma.clamp_min(1e-6)
        result["spread32"] = (-radius, radius)
        result["constant"] = (-torch.ones_like(p), torch.ones_like(p))
        return p, result


@torch.no_grad()
def calibrate(provider):
    scores = {name: [] for name in (*MODES, "spread32", "constant")}
    for start in range(0, 161, 20):
        p, bounds = provider.bounds("calibration", start)
        residual = (target("calibration", start) - provider.norm["mean"]) / provider.norm["std"] - p
        for name, (lower, upper) in bounds.items():
            midpoint, width = (lower + upper) * .5, ((upper - lower) * .5).clamp_min(1e-6)
            scores[name].append(((residual - midpoint).abs() / width).cpu())
    factors = {}
    for name, values in scores.items():
        ratio = torch.cat(values).cuda()
        global_q = grouped_quantile(ratio, "global", pooled=True).clamp(1e-6, 1e3)
        lead_q = grouped_quantile(ratio, "lead", pooled=True).clamp(1e-6, 1e3)
        factors[name] = dict(global_q=global_q.cpu(), lead_q=(global_q * lead_q).sqrt().cpu())
    torch.save(factors, OUT / "calibration.pt")
    dump(OUT / "calibration_locked.json", dict(time=time.time(), sha256=digest(OUT / "calibration.pt"),
         method="Empirical90% normalized absolute residual about interval midpoint. Independent trajectories; no formal conformal guarantee."))


def metrics(lower, upper, residual):
    inside = (residual >= lower) & (residual <= upper)
    coverage = inside.float().mean((2, 3, 4))
    patch = inside.reshape(len(inside), 2, 20, 4, 8, 8, 8).float().mean((2, 4, 6))
    return dict(coverage=coverage, standardized_width=(upper-lower).mean((2, 3, 4)),
         standardized_interval_score=(upper-lower + 20*(lower-residual).clamp_min(0) + 20*(residual-upper).clamp_min(0)).mean((2, 3, 4)),
         lower_miss=(residual<lower).float().mean((2, 3, 4)), upper_miss=(residual>upper).float().mean((2, 3, 4)),
         worst_frame_coverage=inside.float().mean((3,4)).min(-1).values,
         worst_patch_coverage=patch.flatten(2).min(-1).values,
         window90=(coverage>=.9).float(), window95=(coverage>=.95).float(),
         whole_block_joint_covered=inside.flatten(1).all(-1)[:,None].expand(-1,2).float())


@torch.no_grad()
def evaluate(provider):
    initial = torch.load(OUT / "calibration.pt", map_location="cuda", weights_only=False)
    rows = []
    for split in ("development", "test"):
        records = manifest()["files"][split]
        state = {name: {key: value.expand(len(records), *value.shape[1:]).clone()
                       for key, value in factors.items()} for name, factors in initial.items()}
        for start in range(0, 761, 20):
            p, raw = provider.bounds(split, start)
            intervals = {}
            for name, (lower, upper) in raw.items():
                mid, width = .5*(lower+upper), .5*(upper-lower)
                intervals[name] = dict(raw=(lower, upper))
                for variant in VARIANTS[1:]:
                    q = initial[name]['global_q'] if variant == 'fixed' else state[name]['lead_q' if variant == 'online_lead' else 'global_q']
                    intervals[name][variant] = (mid-q*width, mid+q*width)
            event(event="intervals_issued", split=split, start=start, previous_complete_windows=start//20)
            truth = target(split, start)
            event(event="target_revealed", split=split, start=start)
            residual = (truth-provider.norm['mean'])/provider.norm['std']-p
            mse = (residual*provider.norm['std']).square().mean((2,3,4))
            for name, methods in intervals.items():
                lower, upper = raw[name]
                midpoint, width = .5*(lower+upper), (.5*(upper-lower)).clamp_min(1e-6)
                location = ranking(width, residual.abs())
                for variant, (lo, hi) in methods.items():
                    values = dict(**metrics(lo, hi, residual), **location)
                    for i, record in enumerate(records):
                        for c, channel in enumerate(('u','v')):
                            rows.append(dict(kind='diffusion', split=split, file=record['file'], group=record['group'],
                                 start=start, period='confirmation' if start>=580 else 'main', channel=channel,
                                 map=name, variant=variant, mse=float(mse[i,c]),
                                 **{key:float(value[i,c]) for key,value in values.items()}))
                ratio = (residual-midpoint).abs()/width
                global_q = grouped_quantile(ratio,'global').clamp(1e-6,1e3)
                lead_q = grouped_quantile(ratio,'lead').clamp(1e-6,1e3)
                state[name]['global_q'] = (state[name]['global_q']*global_q).sqrt()
                state[name]['lead_q'] = (state[name]['lead_q']*(global_q*lead_q).sqrt()).sqrt()
            if start in (180,380,580,760):
                print('EVALUATE', split, start, flush=True)
    dump(OUT/'per_case.json',rows)
    dump(OUT/'summary.json',summarize(rows))


def main():
    configure()
    OUT.mkdir(exist_ok=False)
    norm = normalization()
    predictor_path = PREDICTOR/'augmented_diffusion/model.pt'
    predictor_hash = digest(predictor_path)
    dump(OUT/'protocol.json',dict(time=time.time(), source_sha256=digest(Path(__file__)),
         module_sha256=digest(ROOT/'direct_error_uncertainty.py'), predictor_sha256=predictor_hash,
         point_center='Frozen Direct20 zero-noise4-call output, common to ALL methods; no signed point correction.',
         fit_files=[r['file'] for r in FIT_RECORDS], fit_frames=[200,400], predictor_fit_frames=[0,200],
         calibration_files=[r['file'] for r in manifest()['files']['calibration']], calibration_frames=[0,200],
         modes=MODES, steps=2000, batch_size=8, seed=42,
         features='Actual inference up2 48channels at each of4 calls; float16 cache; no future/noised truth in feature extraction.',
         controls='272input channels in every head; missing feature channels masked after train-only standardization. Matched last/all quantile heads.',
         output='Gaussian sigma or ordered unconstrained asymmetric5/95 residual quantiles. Asymmetric interval midpoint may differ from P; point forecast remains P.',
         calibration='New empirical per-channel90% normalized scores about interval midpoint; fixed/global online/lead online. No theorem claimed.',
         update='Only after entire20-frame truth reveal. Geometric step0.5; lead estimates shrink halfway toward global in log scale.',
         selection='No development/test fitting, endpoint selection, crossfold, distillation or multiple training seeds.',
         limitations='All data historically reused; exploration only. Time-heldout risk fit still shares predictor training trajectories.'))
    model = new_model('augmented_diffusion').eval().requires_grad_(False)
    model.load_state_dict(torch.load(predictor_path,map_location='cuda',weights_only=False)['model'])
    cache(model,norm,'fit',FIT_STARTS)
    train(norm)
    frozen=json.loads((OUT/'heads_frozen.json').read_text())
    for split in ('calibration','development','test'):
        cache(model,norm,split,range(0,161 if split=='calibration' else 761,20))
    provider=Provider(norm)
    calibrate(provider)
    evaluate(provider)
    assert digest(predictor_path)==predictor_hash
    assert all(digest(OUT/name)==sha for name,sha in frozen['artifacts'].items())
    dump(OUT/'complete.json',dict(time=time.time(),predictor_unchanged=True,heads_unchanged=True))


if __name__=='__main__':
    main()
