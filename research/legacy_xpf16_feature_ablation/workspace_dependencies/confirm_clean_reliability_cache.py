"""Generate independent sampling spreads and an additional temporal check."""

import json
import os
import time

import torch

from clean_reliability_common import (ROOT, OUT, SOURCE, configure, manifest, load_predictor,
    field_path, issued_path, read_range, digest, dump)
from confirm_clean_reliability_fit import CONFIRM, PRIMARY


def confirmation_starts(split):
    return range(0, 161 if split == "calibration" else 761, 20)


def field_file(role, split, start):
    old = field_path(role, split, start)
    return old if old.exists() else CONFIRM / "fields" / f"{role}_{split}_{start}.pt"


def original_file(kind, split, start):
    old = issued_path(kind, split, start)
    return old if old.exists() else CONFIRM / "original" / f"{kind}_{split}_{start}.pt"


def independent_file(kind, split, start):
    return CONFIRM / "independent" / f"{kind}_{split}_{start}.pt"


def event(**values):
    with (CONFIRM / "events.jsonl").open("a") as stream:
        stream.write(json.dumps(dict(time=time.time(), **values)) + "\n")


@torch.no_grad()
def main():
    configure()
    assert os.environ.get("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE") == "0"
    assert (CONFIRM / "power_frozen.json").exists()
    for folder in ("fields", "original", "independent"):
        (CONFIRM / folder).mkdir(exist_ok=False)
    dump(CONFIRM / "sampling_locked.json", dict(power_sha256=digest(CONFIRM / "power_frozen.json"),
        sources={f: digest(ROOT / f) for f in ("confirm_clean_reliability_cache.py", "confirm_clean_reliability_fit.py")}))
    begin = time.time()
    for kind in PRIMARY:
        model, codec, ck, norm = load_predictor(kind)
        for split in ("calibration", "development", "test"):
            for start in confirmation_starts(split):
                hp = field_file("history", split, start)
                if not hp.exists():
                    torch.save(read_range(split, start, start + 20), hp)
                history = torch.load(hp, map_location="cuda", weights_only=False)
                code = (codec.encode((history - norm["mean"]) / norm["std"])[0] - ck["latent_center"]) / ck["latent_scale"]

                def paths(seed):
                    return torch.stack([
                        codec.decode(model.rollout(code, s) * ck["latent_scale"] + ck["latent_center"]) * norm["std"] + norm["mean"]
                        for s in range(seed, seed + 32)
                    ])

                op = original_file(kind, split, start)
                if op.exists():
                    original = torch.load(op, map_location="cuda", weights_only=False)
                else:
                    x = paths(500000)
                    q = torch.quantile(x, torch.tensor([.05, .95], device="cuda"), dim=0)
                    original = dict(mean=x.mean(0), std8=x[:8].std(0), std32=x.std(0), quantile32=(q[1] - q[0]) / 2,
                                    files=[r["file"] for r in manifest()["files"][split]], history_start=start)
                    torch.save({k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in original.items()}, op)
                    del x, q
                x = paths(700000)
                assert torch.isfinite(x).all()
                mu, sd = x.mean(0), x.std(0)
                q = torch.quantile(x, torch.tensor([.05, .95], device="cuda"), dim=0)
                pooled = ((31 * original["std32"].square() + 31 * sd.square() + 16 * (original["mean"] - mu).square()) / 63).sqrt()
                independent = dict(independent_mean=mu.cpu(), std8=x[:8].std(0).cpu(), std32=sd.cpu(),
                                   quantile32=((q[1] - q[0]) / 2).cpu(), std64=pooled.cpu())
                torch.save(independent, independent_file(kind, split, start))
                event(event="issued", kind=kind, split=split, start=start)
                tp = field_file("target", split, start)
                if not tp.exists():
                    torch.save(read_range(split, start + 20, start + 40), tp)
                event(event="target_revealed", kind=kind, split=split, start=start)
                del x, q, original, independent, mu, sd, pooled, code, history
                if start % 200 == 0:
                    print("CONFIRM CACHE", kind, split, start, round(time.time() - begin, 1), flush=True)
                dump(CONFIRM / "status.json", dict(stage="cache", kind=kind, split=split, start=start, pid=os.getpid()))
        del model, codec, ck, norm
        torch.cuda.empty_cache()
    dump(CONFIRM / "cache_complete.json", dict(elapsed=time.time() - begin))


if __name__ == "__main__":
    main()
