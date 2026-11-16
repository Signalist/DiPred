"""Frozen20->20 point forecasts; directly learn temporally structured error.

Only training trajectories' heldout times fit heads. Calibration and evaluation
follow the previous direct-UQ protocol, including its common point center.
"""

import json
import math
from pathlib import Path
import time

import torch

from clean_reliability_common import ROOT, configure, digest, dump, manifest
from experiment_reliability_transport_e12 import (
    FIT_RECORDS, FIT_STARTS, PREDICTOR, geometry, history, target, normalization,
)
from experiment_direct_error_uncertainty import (
    OUT as REFERENCE, Provider as ReferenceProvider, cached, metrics, VARIANTS,
)
from evaluate_reliability_transport_e12 import ranking, summarize
from experiment_reliability_structured_e3 import grouped_quantile
from reliability_risk_transport import base_features
from temporal_error_uncertainty import MODES, TemporalErrorHead


OUT = ROOT / "runs/temporal_error_uncertainty_20261109"
NAMES = (*MODES, "reference_spread32", "reference_quantile_last")


def event(**values):
    with (OUT / "events.jsonl").open("a") as stream:
        stream.write(json.dumps(dict(time=time.time(), **values)) + "\n")


@torch.no_grad()
def pack(split, start, norm, geo):
    value = cached(split, start)
    h = history(split, start)
    p = value["prediction"]
    physical_p = p * norm["std"] + norm["mean"]
    base = base_features(h, physical_p, norm["mean"], norm["std"], geo)
    return p, base, value["features"][:, -48:].float()


def normalized(base, last, stats):
    return (((base - stats["base_mean"]) / stats["base_std"]).clamp(-10, 10),
            ((last - stats["last_mean"]) / stats["last_std"]).clamp(-10, 10))


def train(norm):
    geo = geometry(FIT_RECORDS)
    bases, lasts, residuals = [], [], []
    for start in FIT_STARTS:
        p, base, last = pack("fit", start, norm, geo)
        bases.append(base)
        lasts.append(last)
        residuals.append((target("fit", start) - norm["mean"]) / norm["std"] - p)
    base, last, residual = [torch.cat(values) for values in (bases, lasts, residuals)]
    del bases, lasts, residuals
    stats = dict(base_mean=base.mean((0, 2, 3, 4), keepdim=True),
                 base_std=base.std((0, 2, 3, 4), keepdim=True).clamp_min(.05),
                 last_mean=last.mean((0, 2, 3), keepdim=True),
                 last_std=last.std((0, 2, 3), keepdim=True).clamp_min(.05))
    torch.save({key: value.cpu() for key, value in stats.items()}, OUT / "statistics.pt")
    base, last = normalized(base, last, stats)
    parameters = {}
    for mode in MODES:
        torch.manual_seed(42)
        head = TemporalErrorHead(mode).cuda()
        parameters[mode] = sum(p.numel() for p in head.parameters())
        optimizer = torch.optim.AdamW(head.parameters(), lr=3e-4, weight_decay=1e-5)
        logs, begin = [], time.time()
        for step in range(1, 2001):
            ix = torch.randint(len(base), (4,), device="cuda")
            rate = 1e-6 + (3e-4 - 1e-6) * .5 * (1 + math.cos(math.pi * (step-1) / 2000))
            for group in optimizer.param_groups:
                group["lr"] = rate
            optimizer.zero_grad(set_to_none=True)
            loss = head.loss(base[ix], last[ix], residual[ix])
            assert torch.isfinite(loss)
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(head.parameters(), 1.)
            assert torch.isfinite(gradient)
            optimizer.step()
            if step == 1 or step % 500 == 0:
                logs.append(dict(step=step, loss=float(loss), elapsed=time.time()-begin))
                print("TRAIN", mode, logs[-1], flush=True)
        torch.save(dict(model=head.state_dict(), mode=mode, steps=2000, seed=42), OUT / f"{mode}.pt")
        dump(OUT / f"{mode}_training.json", logs)
    dump(OUT / "heads_frozen.json", dict(time=time.time(), parameters=parameters,
         artifacts={path.name: digest(path) for path in OUT.glob("*.pt")}))


class Provider:
    def __init__(self, norm):
        self.norm = norm
        self.reference = ReferenceProvider(norm)
        self.stats = torch.load(OUT / "statistics.pt", map_location="cuda", weights_only=False)
        self.heads = {}
        for mode in MODES:
            head = TemporalErrorHead(mode).cuda().eval().requires_grad_(False)
            head.load_state_dict(torch.load(OUT / f"{mode}.pt", map_location="cuda", weights_only=False)["model"])
            self.heads[mode] = head

    @torch.no_grad()
    def bounds(self, split, start):
        p, reference = self.reference.bounds(split, start)
        geo = geometry(manifest()["files"][split])
        _, base, last = pack(split, start, self.norm, geo)
        base, last = normalized(base, last, self.stats)
        result = {mode: head.bounds(base, last) for mode, head in self.heads.items()}
        result.update(reference_spread32=reference["spread32"],
                      reference_quantile_last=reference["quantile_last"])
        return p, result


@torch.no_grad()
def calibrate(provider):
    scores = {name: [] for name in NAMES}
    for start in range(0, 161, 20):
        p, bounds = provider.bounds("calibration", start)
        residual = (target("calibration", start)-provider.norm["mean"])/provider.norm["std"]-p
        for name, (lower, upper) in bounds.items():
            mid, width = .5*(lower+upper), (.5*(upper-lower)).clamp_min(1e-6)
            scores[name].append(((residual-mid).abs()/width).cpu())
    factors = {}
    for name, values in scores.items():
        ratio = torch.cat(values).cuda()
        global_q = grouped_quantile(ratio, "global", pooled=True).clamp(1e-6, 1e3)
        lead_q = grouped_quantile(ratio, "lead", pooled=True).clamp(1e-6, 1e3)
        factors[name] = dict(global_q=global_q.cpu(), lead_q=(global_q*lead_q).sqrt().cpu())
    torch.save(factors, OUT / "calibration.pt")
    dump(OUT / "calibration_locked.json", dict(time=time.time(), sha256=digest(OUT / "calibration.pt")))


@torch.no_grad()
def evaluate(provider):
    initial = torch.load(OUT/"calibration.pt", map_location="cuda", weights_only=False)
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
                intervals[name] = dict(raw=(lower,upper))
                for variant in VARIANTS[1:]:
                    q = initial[name]["global_q"] if variant=="fixed" else state[name]["lead_q" if variant=="online_lead" else "global_q"]
                    intervals[name][variant] = (mid-q*width, mid+q*width)
            event(event="issued", split=split, start=start, previous_complete_windows=start//20)
            residual = (target(split,start)-provider.norm["mean"])/provider.norm["std"]-p
            event(event="revealed", split=split, start=start)
            mse = (residual*provider.norm["std"]).square().mean((2,3,4))
            for name, methods in intervals.items():
                lower, upper = raw[name]
                mid, width = .5*(lower+upper), (.5*(upper-lower)).clamp_min(1e-6)
                location = ranking(width,residual.abs())
                for variant,(lo,hi) in methods.items():
                    values = dict(**metrics(lo,hi,residual), **location)
                    for i,record in enumerate(records):
                        for c,channel in enumerate(("u","v")):
                            rows.append(dict(kind="diffusion", split=split, file=record["file"], group=record["group"],
                                 start=start, period="confirmation" if start>=580 else "main", channel=channel,
                                 map=name, variant=variant, mse=float(mse[i,c]),
                                 **{key:float(value[i,c]) for key,value in values.items()}))
                ratio = (residual-mid).abs()/width
                global_q = grouped_quantile(ratio,"global").clamp(1e-6,1e3)
                lead_q = grouped_quantile(ratio,"lead").clamp(1e-6,1e3)
                state[name]["global_q"] = (state[name]["global_q"]*global_q).sqrt()
                state[name]["lead_q"] = (state[name]["lead_q"]*(global_q*lead_q).sqrt()).sqrt()
            if start in (180,380,580,760):
                print("EVALUATE",split,start,flush=True)
    dump(OUT/"per_case.json",rows)
    dump(OUT/"summary.json",summarize(rows))


def main():
    configure()
    OUT.mkdir(exist_ok=False)
    predictor = PREDICTOR/"augmented_diffusion/model.pt"
    point_hash = digest(predictor)
    reference_hashes = json.loads((REFERENCE/"heads_frozen.json").read_text())["artifacts"]
    dump(OUT/"protocol.json",dict(time=time.time(),source_sha256=digest(Path(__file__)),
         module_sha256=digest(ROOT/"temporal_error_uncertainty.py"),predictor_sha256=point_hash,
         point_center="Exact previous zero-noise Direct20 center; no change or point correction.",
         modes=MODES,steps=2000,batch_size=4,seed=42,
         architecture="14physical risk features plus4projected last-call feature channels; two3D16channel convolutions.",
         objective="Quantile pinball, Gaussian NLL, and log-absolute-error scale control; matched base/last for pinball/logscale.",
         fit_files=[r["file"] for r in FIT_RECORDS],fit_frames=[200,400],predictor_fit_frames=[0,200],
         calibration_files=[r["file"] for r in manifest()["files"]["calibration"]],calibration_frames=[0,200],
         selection="Fixed endpoints and rules before evaluation; no development/test fitting, crossfold, distillation or extra training seeds.",
         calibration="Same independent empirical global90% and online global/lead updates as previous direct-UQ; new heads recalibrated.",
         constraints="Full20generation and feedback only; no sampling required by new heads; reuse inference-only feature cache.",
         limits="Historically reused data, exploratory; correlated data not automatically conformal. Architecture and physical input features both differ from prior2D head."))
    norm = normalization()
    train(norm)
    frozen = json.loads((OUT/"heads_frozen.json").read_text())["artifacts"]
    provider = Provider(norm)
    calibrate(provider)
    evaluate(provider)
    assert digest(predictor)==point_hash
    assert all(digest(OUT/name)==sha for name,sha in frozen.items())
    assert all(digest(REFERENCE/name)==sha for name,sha in reference_hashes.items())
    dump(OUT/"complete.json",dict(time=time.time(),predictor_and_all_heads_unchanged=True))


if __name__=="__main__":
    main()
