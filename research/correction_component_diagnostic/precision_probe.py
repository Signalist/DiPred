"""Compare one fixed member across independently configured CUDA processes."""

import argparse
import os
from pathlib import Path
from time import perf_counter

import h5py
import torch

from research.residual_uncertainty_diffusion.core import OUT, dump, read, state_hash, tensor_hash
from research.residual_uncertainty_diffusion.experiment import freeze_hash
from research.trust_causal_audit_v2.common import DATA, MEMBERS, load_backbone
from research.trust_causal_audit_v2.data import H5Reader, batch_noise
from research.trust_causal_audit_v2.model import native_sample
from .evidence import Evidence
from .sampler import sample


KEY = "development__6300_0.h5__420"


def runtime_flags():
    return dict(matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
                cudnn_tf32=torch.backends.cudnn.allow_tf32,
                cudnn_deterministic=torch.backends.cudnn.deterministic,
                cudnn_benchmark=torch.backends.cudnn.benchmark,
                tf32_override=os.getenv("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"),
                nvidia_tf32_override=os.getenv("NVIDIA_TF32_OVERRIDE"),
                cublas_workspace_config=os.getenv("CUBLAS_WORKSPACE_CONFIG"))


@torch.no_grad()
def probe(mode, destination, matched_batch=False):
    started = perf_counter()
    before_load = runtime_flags()
    evidence = Evidence()
    row = next(r for r in read(OUT / "development_DATA_MANIFEST.json") if r["cache_key"] == KEY)
    item = next(e for e in evidence.load_cache("development_conditions.pt") if e["cache_key"] == KEY)
    cached, scores = evidence.load_development_paths([KEY])
    native, norm = load_backbone()
    before = runtime_flags()
    assert before["matmul_tf32"] == (mode == "legacy")
    assert before["cudnn_deterministic"] and not before["cudnn_tf32"]
    frozen = read(OUT / "FROZEN.json")
    weights = state_hash(native)
    assert weights == frozen["initial_backbone_state"]
    assert freeze_hash(native, exclude_out=True) == frozen["frozen_non_out"]
    history = H5Reader(DATA)(row["trajectory_id"], *row["history_interval"])
    assert tensor_hash(history) == row["history_sha256"]
    normalized = (history - norm["mean"].cpu()) / norm["std"].cpu()
    assert torch.equal(normalized, item["H"])
    key = item["key"]
    assert (key.split, key.trajectory_id, key.start) == (row["split"], row["trajectory_id"], row["start"])
    member_slice = None if matched_batch else slice(0, 1)
    noise, ids = batch_noise(key, evidence.noise_files[key.split], member_slice=member_slice)
    member = MEMBERS[0]
    expected_ids = MEMBERS if matched_batch else (member,)
    assert tuple(rec["member_id"] for rec in ids) == expected_ids
    for tensor, rec in zip(noise, ids):
        assert tensor_hash(tensor) == evidence.noise_hashes[key.split, key.trajectory_id, rec["member_id"]]
    h = normalized.cuda()[None].expand(len(noise), -1, -1, -1, -1)
    raw = native_sample(native, h, noise)
    wrapper_exact = None
    if not matched_batch:
        wrapped = sample(native, dict(history=h, noise=noise))
        wrapper_exact = torch.equal(raw, wrapped)
        assert wrapper_exact
    after = runtime_flags()
    assert before == after and state_hash(native) == weights
    reference = cached[KEY]["native"][0]
    archived = cached[KEY]["native"]
    error = float((raw.cpu()[0] - reference).abs().max())
    ensemble_error = float((raw.cpu() - archived).abs().max()) if matched_batch else None
    dump(destination, dict(
        mode=mode, key=KEY, history_interval=row["history_interval"], member_id=member, member_index=0,
        archived_member_order=list(MEMBERS), noise_identity=ids[0],
        raw_history_sha256=tensor_hash(history), normalized_history_sha256=tensor_hash(normalized),
        normalization_sha256={k: tensor_hash(v) for k, v in norm.items()}, weights_sha256=weights,
        output_dtype=str(raw.dtype), cache_scale="standardized; verified by archived scores in full diagnostic",
        complete_32_member_groups=int(matched_batch), single_member_four_step_calls=0 if matched_batch else 2,
        ensemble_output_sha256=tensor_hash(raw) if matched_batch else None,
        ensemble_maxabs_vs_archive=ensemble_error,
        before_checking_weights=before_load, before_inference=before, after_inference=after,
        torch=torch.__version__, cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
        h5py=h5py.__version__, gpu=torch.cuda.get_device_name(),
        output_sha256=tensor_hash(raw[0]), wrapper_exact=wrapper_exact, archived_member_sha256=tensor_hash(reference),
        archived_ensemble_sha256=scores["native", 0, row["trajectory_id"], "overall"]["output_sha256"],
        maxabs_vs_archived_member=error, within_fixed_2e_minus_6=error <= 2e-6,
        wall_seconds=perf_counter()-started,
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("legacy", "fp32"))
    parser.add_argument("destination", type=Path)
    parser.add_argument("--matched-batch", action="store_true")
    args = parser.parse_args()
    if args.destination.exists():
        parser.error("destination already exists")
    probe(args.mode, args.destination, args.matched_batch)
