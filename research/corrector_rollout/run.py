"""Evaluate frozen Correctors on the original DiPred path cache."""
import argparse
import hashlib
import json
import shlex
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from research.corrector.run import Corrector, dump, load_truth, sha, table
from research.corrector_rollout.summarize import ORACLE, ROLLOUT, write_results
from research.corrector_rollout.verify import verify_rollout, verify_scores

STARTS = (400, 420)
SEEDS = list(range(500000, 500032))
SPLITS = ("development", "test")
METHODS = ("Persistence", "R-only", "State-conditioned")
DISPLAY = ("development", "26700_0.h5", 400, 500000)
PRIOR = REPO / "results/corrector"
DYNAMICS = REPO / "results/residual_dynamics"
OUT = REPO / "results/corrector_rollout"
CHECKPOINTS = {
    "R-only": ("R_only.pt", "3633afc89d52fd5744764ee0b1eb5b1bd19f4cb253ac86158eccbf38e98626e2"),
    "State-conditioned": (
        "State_conditioned.pt",
        "79fe891425f522ae84b5af2ddfc117e0df30e82721f04d3d7e10b45bed0bcf33",
    ),
}


def read_json(path):
    return json.loads(path.read_text())


def require(condition, detail):
    if not condition:
        raise ValueError(detail)


def tensor_sha(value):
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def input_weights(workspace, frozen):
    return {
        **{PRIOR / filename: digest for filename, digest in CHECKPOINTS.values()},
        REPO / "reliability/checkpoints/current.pt": frozen["packaged_checkpoint_sha256"],
        workspace / "runs/physical20_augmented_dynamics_20261109/augmented_diffusion/model.pt":
            frozen["original_checkpoint_sha256"],
    }


def verify_weights(weights):
    for path, expected in weights.items():
        require(sha(path) == expected, f"Checkpoint SHA-256 mismatch: {path}")


def load_models(device):
    models = {}
    training = read_json(PRIOR / "training.json")
    for name, (filename, _) in CHECKPOINTS.items():
        # The legacy checkpoint includes float metadata unsupported by this
        # runtime’s weights-only loader. verify_inputs pins the file digest.
        ck = torch.load(PRIOR / filename, map_location=device, weights_only=False)
        conditioned = name == "State-conditioned"
        require(ck["conditioned"] == conditioned, f"Conditioning mismatch: {name}")
        require(ck["step"] == training["selected_steps"][name] == 1000, f"Wrong step: {name}")
        require(ck["scale"] == training["scalar_train_rms"], f"Normalization mismatch: {name}")
        model = Corrector(conditioned).to(device).eval().requires_grad_(False)
        model.load_state_dict(ck["model"], strict=True)
        require(sum(p.numel() for p in model.parameters()) == 11938, f"Capacity mismatch: {name}")
        models[name] = (model, ck["scale"])
    return models


def verify_inputs(args):
    protocol = read_json(PRIOR / "protocol.json")
    original = read_json(DYNAMICS / "protocol.json")
    manifest = DYNAMICS / "cache_manifest.json"
    require(sha(manifest) == protocol["base_cache_manifest_sha256"], "Original cache manifest changed")
    require(protocol["records"] == original["records"], "Original trajectory records differ")
    require(protocol["starts"] == original["starts"] == list(STARTS), "History starts differ")
    require(protocol["member_seeds"] == original["member_seeds"] == SEEDS, "Member labels differ")
    alignment = read_json(DYNAMICS / "alignment.json")
    frozen = original["frozen"]
    require(alignment["passed"] and alignment["frozen"] == frozen, "Backbone alignment provenance differs")
    require(frozen["predictor_tensors_exact"], "Missing original backbone tensor comparison")
    for snapshot in alignment["snapshots"]:
        require(
            snapshot["truth_exact"] and snapshot["members"] == SEEDS
            and snapshot["all_32_members_all_20_frames_max_abs_error"] <= 2e-6,
            "Original all-member alignment audit failed",
        )

    weights = input_weights(args.workspace, frozen)
    verify_weights(weights)
    entries = {Path(row["path"]).name: row for row in read_json(manifest)}
    require(set(entries) == {f"{split}_{start}.pt" for split in protocol["records"] for start in STARTS},
            "Cache manifest does not cover all splits and starts")
    records = protocol["records"]
    all_files = [rec["file"] for split in records.values() for rec in split]
    require(len(all_files) == len(set(all_files)), "Trajectory leakage across splits")

    data = {}
    audit = []
    for split, split_records in records.items():
        history = load_truth(args.data, split_records, STARTS[0])
        for start in STARTS:
            filename = f"{split}_{start}.pt"
            entry = entries[filename]
            path = args.cache / filename
            require(sha(path) == entry["sha256"], f"Cache SHA-256 mismatch: {path}")
            cache = torch.load(path, map_location="cpu", weights_only=True)
            identity = dict(split=split, start=start, files=[rec["file"] for rec in split_records],
                            seeds=SEEDS, frozen=frozen)
            require(cache["identity"] == entry["identity"] == identity, f"Bad identity: {path}")
            old_residual = cache["residual"]
            truth = load_truth(args.data, split_records, start + 20)
            n = len(split_records)
            require(old_residual.shape == (n, 32, 2, 20, 32, 64), f"Bad residual shape: {path}")
            require(old_residual.dtype == torch.float32 and torch.isfinite(old_residual).all(),
                    f"Invalid cached residual: {path}")
            require(history.shape == truth.shape == (n, 2, 20, 32, 64), f"Bad HDF5 shape: {path}")
            for key, tensor in (("history_sha256", history), ("truth_sha256", truth)):
                require(tensor_sha(tensor) == cache[key] == entry[key], f"{key} mismatch: {path}")
            history = truth
            if split in SPLITS:
                data[split, start] = (truth, old_residual)
            audit.append(dict(path=str(path), split=split, start=start, trajectories=n,
                              sha256=entry["sha256"], history_sha256=entry["history_sha256"],
                              truth_sha256=entry["truth_sha256"], members=SEEDS))
            print(f"VERIFIED CACHE {split} {start}: {n} trajectories, {len(SEEDS)} ordered members",
                  flush=True)
    return records, data, weights, audit


def decode_window(truth, old_residual, device):
    """Recover the fixed backbone prediction before any truth intervention."""
    truth = truth.to(device)
    old_residual = old_residual.to(device)
    return truth, truth.unsqueeze(0) + old_residual, -old_residual


def next_residual(model_scale, current, backbone, state, lead):
    model, scale = model_scale
    target_lead = torch.full((current.shape[0],), lead, device=current.device)
    return -model(state / scale, backbone / scale, -current / scale, target_lead) * scale


def rollout(models, backbone, initial):
    """One window, with axes member, channel, physical frame, y, x."""
    require(backbone.ndim == 5 and backbone.shape[2] == 20, "Expected one 20-frame prediction window")
    require(initial.shape == backbone[:, :, 0].shape, "Initial residual must be one frame per member")
    output = {"Persistence": initial.unsqueeze(2).expand_as(backbone)}
    for name, model_scale in models.items():
        frames = [initial]
        for lead in range(2, 21):
            current = frames[-1]
            if name == "State-conditioned":
                predicted = backbone[:, :, lead - 2]
                state = predicted + current
            else:
                predicted = state = torch.zeros_like(current)
            frames.append(next_residual(model_scale, current, predicted, state, lead))
        output[name] = torch.stack(frames, dim=2)
    return output


def one_step(models, backbone, truth, residual):
    output = {"Persistence": residual[:, :, :-1]}
    truth = truth.unsqueeze(0).expand_as(backbone)
    for name, model_scale in models.items():
        output[name] = torch.stack([
            next_residual(model_scale, residual[:, :, lead - 2], backbone[:, :, lead - 2],
                          truth[:, :, lead - 2], lead)
            for lead in range(2, 21)
        ], dim=2)
    return output


def score(truth, backbone, residual, correction):
    # Subtract in float64 for the physical field to minimize cancellation.
    residual_error = residual.double() - correction.double()
    physical_error = truth.double().unsqueeze(0) - (backbone.double() + correction.double())
    torch.testing.assert_close(physical_error, residual_error, rtol=0, atol=1.2e-7)
    mse = residual_error.square().mean(dim=(0, 3, 4))
    physical_mse = physical_error.square().mean(dim=(0, 3, 4))
    require(torch.isfinite(mse).all() and torch.isfinite(physical_mse).all(), "Nonfinite MSE")
    return mse.cpu().numpy(), physical_mse.cpu().numpy()


def free_window(models, backbone, truth, residual):
    outputs = rollout(models, backbone, residual[:, :, 0])
    corrections = {name: fields[:, :, 1:] for name, fields in outputs.items()}
    corrections["Uncorrected DiPred"] = torch.zeros_like(residual[:, :, 1:])
    return outputs, window_scores(truth, backbone, residual, corrections)


def window_scores(truth, backbone, residual, corrections):
    return {
        name: score(truth[:, 1:], backbone[:, :, 1:], residual[:, :, 1:], correction)
        for name, correction in corrections.items()
    }


def evaluate(args, records, data, models, experiment):
    rows = []
    display = None
    for split in SPLITS:
        for start in STARTS:
            truth_batch, old_residual_batch = data[split, start]
            for rec, raw_truth, old_residual in zip(records[split], truth_batch, old_residual_batch):
                truth, backbone, residual = decode_window(raw_truth, old_residual, args.device)
                if experiment == ORACLE:
                    outputs = one_step(models, backbone, truth, residual)
                    scores = window_scores(truth, backbone, residual, outputs)
                else:
                    outputs, scores = free_window(models, backbone, truth, residual)
                    if (split, rec["file"], start) == DISPLAY[:3]:
                        member = SEEDS.index(DISPLAY[3])
                        display = {"True residual": residual[member, :, 1:].cpu().numpy()}
                        display.update({name: outputs[name][member, :, 1:].cpu().numpy()
                                        for name in models})
                for name, (mse, physical_mse) in scores.items():
                    for channel_index, channel in enumerate(("u", "v")):
                        for lead in range(2, 21):
                            rows.append(dict(
                                experiment=experiment, split=split, trajectory=rec["file"],
                                re_group=rec["group"], re=rec["re"], aoa=rec["aoa"],
                                history_start=start, physical_index=start + 19 + lead,
                                lead=lead, channel=channel, model=name, members=len(SEEDS),
                                residual_mse=mse[channel_index, lead - 2],
                                physical_mse=physical_mse[channel_index, lead - 2],
                            ))
            print(f"SCORED {experiment} {split} {start}: {len(truth_batch)} trajectories", flush=True)
    return rows, display


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("verify", "all"))
    parser.add_argument("--workspace", type=Path, default=REPO.parent / "Forecast/Runtime")
    parser.add_argument("--data", type=Path,
                        default=REPO.parents[1] / "OLD/REALPDE/realpdebench-competition/data/train_real")
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    args.cache = args.cache or args.workspace / "runs/residual_dynamics_20261119"
    OUT.mkdir(parents=True, exist_ok=True)
    # Match the recorded oracle scores on A100, including cuDNN convolution precision.
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = False
    print("COMMAND", shlex.join(["python", *sys.argv]), flush=True)
    print(f"ENV Python {sys.version.split()[0]}, PyTorch {torch.__version__}, device {args.device}",
          flush=True)
    if args.device == "cuda":
        print(f"GPU {torch.cuda.get_device_name()}", flush=True)
    print(f"cuDNN TF32 {torch.backends.cudnn.allow_tf32}, "
          f"cuBLAS TF32 {torch.backends.cuda.matmul.allow_tf32}", flush=True)
    print(f"INPUTS {args.workspace} {args.cache} {args.data}", flush=True)
    records, data, weights, audit = verify_inputs(args)
    models = load_models(args.device)
    oracle, _ = evaluate(args, records, data, models, ORACLE)
    reproduction = verify_scores(oracle, PRIOR)
    print(f"VERIFIED ORACLE {reproduction}", flush=True)
    checks = verify_rollout(data, models, args.device, decode_window, rollout, one_step,
                            next_residual, free_window)
    verify_weights(weights)
    print(f"VERIFIED ROLLOUT {checks}", flush=True)
    verification = dict(passed=True, cache=audit, oracle=reproduction, rollout=checks,
                        checkpoint_sha256_before={str(p): v for p, v in weights.items()})
    if args.stage == "all":
        free, display = evaluate(args, records, data, models, ROLLOUT)
        verify_weights(weights)
        table(OUT / "per_window.csv", oracle + free)
        write_results(OUT, oracle + free, display, DISPLAY)
        print("WROTE scores, paired trajectory statistics, and three figures", flush=True)
    verification["checkpoint_sha256_after"] = {str(p): v for p, v in weights.items()}
    dump(OUT / "verification.json", verification)
    print("PASS", flush=True)


if __name__ == "__main__":
    main()
