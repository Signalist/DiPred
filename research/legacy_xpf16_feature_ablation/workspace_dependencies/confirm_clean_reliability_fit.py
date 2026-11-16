"""Lock a stronger scale control before examining confirmation-time results."""

import json

import torch

from clean_reliability_common import (ROOT, OUT, SOURCE, MAPS, configure, manifest,
    starts, load_field, load_issued, digest, dump)
from evaluate_clean_reliability import MapProvider


CONFIRM = ROOT / "runs/clean_direct20_reliability_confirmation_20261108"
PRIMARY = ("ae", "vae")


@torch.no_grad()
def main():
    configure()
    assert json.loads((OUT / "isolation_audit.json").read_text())["passed"]
    CONFIRM.mkdir(exist_ok=False)
    dump(CONFIRM / "protocol.json", dict(
        motivation="Primary study showed useful average spread information but low-Re counterexamples and overly peaky deterministic profiles. Test those limitations rather than declare universal superiority.",
        models=PRIMARY, task="One direct20 forecast state; model weights and original32-path point means unchanged.",
        confirmation_targets=[600, 800], earlier_diagnostics=[20, 600],
        original_paths=list(range(500000, 500032)), independent_paths=list(range(700000, 700032)),
        spread64="Unbiased pooled variance from two32-path batches; keep original32-path mean as point forecast.",
        power_fit="For each map/channel, training-only beta=clip(cov(log radius, log(abs(error)/train_std+1e-4))/var(log radius),0,2); beta=0 for constant map. Apply radius**beta, then independent calibration. No grid search.",
        scale_fit="Dedicated calibration groups and times0:200; compare raw/power profiles with identical fixed/online scale rules.",
        controls="All11 original maps; four sampling-dependent maps with independent32 paths; std32 and spread-aware learned head with pooled64. Same trained risk heads and training map means.",
        online="Continuous20-frame causal replay from first window through target800; every current prediction precedes its target access. Each map has its own matched scalar update.",
        selection="No model, map, power, sampling set, or temporal-block selection from confirmation results. Report both channels and all held-out Reynolds groups.",
        scope="This is a prospective temporal check relative to the current reliability comparison; these files and late times have historical research use. Not an untouched external test.",
        source_primary_protocol_sha256=digest(OUT / "protocol.json"), source_manifest_sha256=digest(SOURCE / "manifest.json"),
        source_code_sha256={f: digest(ROOT / f) for f in ("confirm_clean_reliability_fit.py", "evaluate_clean_reliability.py", "clean_reliability_common.py")},
    ))
    betas = {}
    regressions = {}
    for kind in PRIMARY:
        provider = MapProvider(kind)
        sums = {name: torch.zeros(5, 2, device="cuda", dtype=torch.float64) for name in MAPS}
        previous = None
        count = 0
        for start in starts("train"):
            issued = load_issued(kind, "train", start)
            history = load_field("history", "train", start)
            profiles = provider.maps(history, issued, previous)
            error = (load_field("target", "train", start) - issued["mean"]).abs()
            y = torch.log(error / provider.norm["std"] + 1e-4).permute(1, 0, 2, 3, 4).flatten(1).double()
            count += y.shape[1]
            for name, profile in profiles.items():
                x = profile.log().permute(1, 0, 2, 3, 4).flatten(1).double()
                sums[name] += torch.stack([x.sum(1), y.sum(1), (x * x).sum(1), (x * y).sum(1), (y * y).sum(1)])
            previous = error
        betas[kind], regressions[kind] = {}, {}
        for name, total in sums.items():
            ex, ey, ex2, exy, ey2 = total / count
            variance = (ex2 - ex.square()).clamp_min(0)
            covariance = exy - ex * ey
            beta = torch.where(variance > 1e-12, covariance / variance.clamp_min(1e-12), 0).clamp(0, 2)
            betas[kind][name] = beta.cpu().tolist()
            regressions[kind][name] = dict(beta=beta.cpu().tolist(), log_radius_variance=variance.cpu().tolist(), samples_per_channel=count)
        del provider
        torch.cuda.empty_cache()
        print("POWER FIT", kind, betas[kind], flush=True)
    dump(CONFIRM / "power_frozen.json", dict(betas=betas, diagnostics=regressions,
                                             fit_split="train", fit_range=[0, 200], selection="none"))


if __name__ == "__main__":
    main()
