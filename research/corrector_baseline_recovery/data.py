"""Extend only the current training split using frozen 32-member means."""
from time import perf_counter
from types import SimpleNamespace

import h5py
import numpy as np
import torch

from reliability.backbone import configure
from research.corrector.run import dump, sha
from research.corrector_predictability.data import EXPANDED, Split, sources
from research.corrector_window_study.data import RAW, ROOT, SEEDS, WORKSPACE, read_json, save_tensor, tensor_sha
from research.residual_dynamics.run import load_backbone

from .legacy import OUT

CACHE = ROOT / ".cache/corrector_baseline_recovery"
NATIVE_PROTOCOL = ROOT / "results/residual_dynamics/protocol.json"


def expansion(base, manifest, out, cache, raw):
    records = read_json(NATIVE_PROTOCOL)["records"]["train"]
    names = [r["file"] for r in records]
    assert names == manifest["splits"]["train"]["files"]
    old = read_json(out / "legacy_windows_819.json")
    old_keys = {(r["file"], r["start"]) for r in old}
    assert len(old_keys) == 819
    allowed = set(names)
    base_keys = {(r["file"], r["start"]) for r in base.cases}
    desired = {(f,s) for f,s in old_keys if f in allowed} | base_keys
    starts = sorted({s for _, s in desired})
    raw_fields = {}
    for record in records:
        file = record["file"]
        with h5py.File(raw / file, "r") as handle:
            assert all(handle[c].shape[0] == record["frames"] for c in ("u","v"))
            field = np.stack([handle[c][starts[0]:starts[-1]+40, ::2, ::2]
                              for c in ("u","v")]).astype("float32")
        assert field.shape[0] == 2 and field.shape[2:] == (32,64) and np.isfinite(field).all()
        raw_fields[file] = torch.from_numpy(field)
    cases, xs, ps, ys, batches = [], [], [], [], []
    skipped = []
    record_by_file = {r["file"]:r for r in records}
    net, norms, identity = load_backbone(SimpleNamespace(workspace=WORKSPACE))
    assert identity == manifest["native_checkpoint"] and manifest["seeds"] == SEEDS
    cache.mkdir(parents=True, exist_ok=True)
    for start in starts:
        position = start - starts[0]
        cut = [f for f in names if (f,start) in desired and record_by_file[f]["frames"] >= start+40]
        skipped.extend(dict(file=f, start=start, frames=record_by_file[f]["frames"])
                       for f in names if (f,start) in desired and f not in cut)
        if not cut:
            continue
        t0 = perf_counter()
        available_history = [f for f in names if record_by_file[f]["frames"] >= start+20]
        x_all = torch.stack([raw_fields[f][:, position:position+20] for f in available_history])
        x = x_all[[available_history.index(f) for f in cut]].cuda()
        y = torch.stack([raw_fields[f][:, position+20:position+40] for f in cut]).cuda()
        entry = dict(start=start, files=cut, history_order=available_history, seeds=SEEDS,
                     history_sha256=tensor_sha(x_all), targets_sha256=tensor_sha(y),
                     backbone=identity, sign=1, normalization_sha256={k:tensor_sha(v) for k,v in norms.items()})
        if start in EXPANDED:
            ix = [i for i,r in enumerate(base.cases) if r["start"] == start]
            assert [base.cases[i]["file"] for i in ix] == cut
            assert torch.equal(x,base.x[ix]) and torch.equal(y,base.y[ix])
            p = base.p[ix]
            entry.update(origin="verified window-study cache", batch_backbone_calls=0,
                         previous_cache_sha256=manifest["splits"]["train"]["native"][str(start)]["sha256"])
        else:
            dest, metadata = cache / f"train_{start}_mean.pt", cache / f"train_{start}_mean.json"
            if dest.exists() and metadata.exists():
                saved = read_json(metadata)
                assert all(saved[k] == v for k,v in entry.items()) and sha(dest) == saved["cache_sha256"]
                p = torch.load(dest, map_location="cuda", weights_only=True)
                entry = saved
            else:
                inp = (x_all.cuda()-norms["mean"])/norms["std"]
                with torch.inference_mode():
                    paths = torch.stack([net.rollout(inp, seed, sign=1)*norms["std"]+norms["mean"]
                                         for seed in SEEDS], dim=1)
                    p = paths.mean(1)[[available_history.index(f) for f in cut]]
                assert torch.isfinite(p).all()
                save_tensor(dest, p)
                entry.update(origin="frozen four-call ensemble", batch_backbone_calls=128,
                             cache_sha256=sha(dest), cache_bytes=dest.stat().st_size,
                             generation_seconds=perf_counter()-t0)
                dump(metadata, entry)
        assert p.shape == y.shape and torch.isfinite(p).all()
        entry["P_sha256"] = tensor_sha(p)
        batches.append(entry)
        cases.extend(dict(file=f, start=start, re=record_by_file[f]["re"],
                          history=[start,start+20], target=[start+20,start+40]) for f in cut)
        for storage, values in ((xs,x),(ps,p),(ys,y)):
            storage.append(values)
        print("EXPANSION", start, len(cut), entry["origin"], flush=True)
    assert len({(c["file"],c["start"]) for c in cases}) == len(cases)
    assert base_keys <= {(c["file"],c["start"]) for c in cases}
    report = dict(legacy_windows=len(old_keys), legacy_disallowed=sorted(old_keys-desired),
                  existing_windows=len(base_keys), expanded_windows=len(cases),
                  missing_windows=skipped, extra_existing_windows=sorted(base_keys-old_keys),
                  model_batch_calls=sum(v["batch_backbone_calls"] for v in batches),
                  added_cache_bytes=sum(v.get("cache_bytes",0) for v in batches),
                  generation_seconds=sum(v.get("generation_seconds",0.) for v in batches),
                  source_raw=str(raw), legacy_manifest_sha256=sha(out/"legacy_windows_819.json"),
                  residual_dynamics_protocol_sha256=sha(NATIVE_PROTOCOL),
                  batches=batches, cases=cases)
    dump(out/"expansion.json",report)
    return SimpleNamespace(cases=cases,x=torch.cat(xs),p=torch.cat(ps),y=torch.cat(ys))


if __name__ == "__main__":
    configure()
    native, _ = sources()
    expansion(Split(native,"train",EXPANDED),native,OUT,CACHE,RAW)
