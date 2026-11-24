"""Truth-free F_last extraction and feature-cache identity."""
import argparse
import os
import time
from pathlib import Path

import torch

from reliability.backbone import configure
from research.corrector.run import dump, sha
from research.corrector_window_study.data import WORKSPACE, read_json, tensor_sha
from research.residual_dynamics.run import load_backbone

from .data import CACHE, OUT, EXPANDED, SMALL, Split, sources

LABELS = (99, 66, 33, 0)
TAP = "net.up2 output before net.out"
BLOCK = 16


def capture(model, x):
    captured, labels = [], []
    handle = model.net.up2.register_forward_hook(lambda _, __, out: captured.append(out))
    label_hook = model.net.register_forward_pre_hook(
        lambda _, args: labels.append(args[1].detach()))
    try:
        model.rollout(x, seed=500000, sign=0)
    finally:
        handle.remove()
        label_hook.remove()
    assert len(captured) == len(labels) == 4
    assert all(torch.equal(v, torch.full_like(v, t)) for v, t in zip(labels, LABELS))
    assert captured[-1].shape == (len(x), 48, 32, 64)
    return captured[-1]


@torch.inference_mode()
def produce(data, name, manifest, cache, out, frozen):
    identity_path = out / f"F_{name}.json"
    path = cache / f"F_{name}.pt"
    if identity_path.exists():
        return load(data, name, cache, out, frozen)
    assert not path.exists(), f"Unverified cache exists: {path}"
    backbone, normalization, identity = load_backbone(argparse.Namespace(workspace=WORKSPACE))
    assert identity == manifest["native_checkpoint"] and all(not p.requires_grad for p in backbone.parameters())
    torch.cuda.synchronize()
    started = time.monotonic()
    history = (data.x - normalization["mean"]) / normalization["std"]
    chunks = [capture(backbone, x) for x in history.split(BLOCK)]
    features = torch.cat(chunks)
    torch.cuda.synchronize()
    elapsed = time.monotonic() - started
    assert torch.isfinite(features).all()
    # Independently intervene on the separate target buffer on development only.
    audit_calls = 0
    if name == "development":
        original = capture(backbone, history[:2])
        p = data.p[:2].clone()
        target = data.y[:2].clone()
        data.y[:2] += 0.2
        try:
            changed = capture(backbone, history[:2])
            assert torch.equal(original, changed) and torch.equal(p, data.p[:2])
        finally:
            data.y[:2] = target
        audit_calls = 8
    cache.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(features.cpu(), tmp)
    os.replace(tmp, path)
    dump(identity_path, dict(split=name, cases=[[c["file"], c["start"]] for c in data.cases],
         history_sha256=tensor_sha(data.x), P_sha256=tensor_sha(data.p), shape=list(features.shape),
         feature_sha256=sha(path), bytes=path.stat().st_size, backbone=identity,
         native_members=manifest["seeds"], feature_path="independent zero initial noise",
         denoising_labels=list(LABELS), feature_tap=TAP,
         sample_network_calls=len(data.x)*4, batch_network_calls=len(chunks)*4,
         audit_network_calls=audit_calls, extraction_seconds=elapsed, frozen_test_authorization=frozen))
    print("FEATURES", name, len(data.x), len(chunks)*4, f"{elapsed:.2f}s", flush=True)
    return features


def load(data, name, cache, out, frozen):
    row = read_json(out / f"F_{name}.json")
    assert row["split"] == name and row["frozen_test_authorization"] == frozen
    assert row["cases"] == [[c["file"], c["start"]] for c in data.cases]
    assert row["history_sha256"] == tensor_sha(data.x) and row["P_sha256"] == tensor_sha(data.p)
    path = cache / f"F_{name}.pt"
    assert sha(path) == row["feature_sha256"]
    result = torch.load(path, map_location=data.x.device, weights_only=True)
    assert list(result.shape) == row["shape"]
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--cache", type=Path, default=CACHE)
    a = p.parse_args()
    configure()
    a.out.mkdir(exist_ok=True, parents=True)
    manifest, _ = sources()
    for name, windows in (("train", EXPANDED), ("development", SMALL)):
        produce(Split(manifest, name, windows), name, manifest, a.cache, a.out, None)


if __name__ == "__main__":
    main()
