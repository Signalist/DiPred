"""Fixed-condition models and metrics for the conditional innovation pilot."""
import csv
import hashlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from research.residual_dynamics_phase3 import run as phase3
from common import WORK, DATA, EXTRA, digest, dump, csvwrite, bootstrap
from mechanisms import simple_predict
from metrics import empirical_crps, energy_score, consecutive_event
from reliability.backbone import configure
from research.residual_dynamics.run import raw

DELIVERY = ROOT / "results/conditional_innovation_pilot"
OUT = Path(os.environ.get("DIPRED_INNOVATION_OUT", str(DELIVERY)))
PRIOR = ROOT / "results/residual_dynamics_phase3_followup"
CONFIG_HASH = "9668ad5b6e0592163ab249bb9a0410e9106fd15dd6fc897b6a133e2881f2e7b7"
assert digest(DELIVERY / "config.json") == CONFIG_HASH
CFG = json.loads((DELIVERY / "config.json").read_text())
RECORDS = CFG["records"]


def read_csv(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def tensor_hash(tensor):
    return hashlib.sha256(tensor.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def operators(device="cuda"):
    specs = json.loads((ROOT / "results/residual_dynamics_phase3/coefficients.json").read_text())
    for key in ("linear", "scale"):
        specs[key]["coef"] = torch.tensor(specs[key]["coef"], dtype=torch.double, device=device)
    return specs["linear"], specs["scale"]


def conditions(history, paths, norm):
    """Only released inputs. paths has shape N,M,C,T,H,W."""
    if history.shape[1:3] != (2, 20) or paths.shape[1:4] != (32, 2, 20):
        raise ValueError("complete20 history and all32 complete20 member paths required")
    last = (history[:, :, -1] - norm["mean"].reshape(1, 2, 1, 1)) / norm["std"].reshape(1, 2, 1, 1)
    mean_history = (history.mean(2) - norm["mean"].reshape(1, 2, 1, 1)) / norm["std"].reshape(1, 2, 1, 1)
    difference = (history[:, :, -1] - history[:, :, -2]) / norm["std"].reshape(1, 2, 1, 1)
    mean_prediction = ((paths.mean(1) - norm["mean"]) / norm["std"]).flatten(1, 2)
    spread = (paths.std(1) / norm["std"]).flatten(1, 2)
    height, width = history.shape[-2:]
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, height, device=history.device),
                            torch.linspace(-1, 1, width, device=history.device), indexing="ij")
    coordinates = torch.stack([xx, yy])[None].expand(len(history), -1, -1, -1)
    return torch.cat([last, mean_history, difference, mean_prediction, spread, coordinates], 1)


def encode_source(bias, operator):
    previous = bias[:, :-1].reshape(-1, 2, *bias.shape[-2:])
    propagated = simple_predict(previous, None, operator).reshape_as(bias[:, 1:])
    return torch.cat([bias[:, :1], bias[:, 1:] - propagated], 1)


def reconstruct(source, operator):
    values = [source[:, 0]]
    for lead in range(1, source.shape[1]):
        values.append(simple_predict(values[-1], None, operator) + source[:, lead])
    return torch.stack(values, 1)


def load_data(split):
    """Training/development only. phase3 loader validates native cache provenance."""
    assert split in ("train", "development")
    data = phase3.load(split)
    fields = phase3.aux(split)
    data.norm = fields["normalization"]
    data.p = data.p.transpose(1, 2)  # N,M,T,C,H,W
    data.x = torch.cat([conditions(fields["history"][:, index],
                    data.p[index * len(RECORDS[split]):(index + 1) * len(RECORDS[split])].transpose(2, 3),
                    data.norm) for index in range(2)])
    data.meta = [dict(trajectory=row["file"], Re=row["group"], start=start)
                 for start in CFG["starts"] for row in RECORDS[split]]
    return data


class Ridge:
    def __init__(self, checkpoint, operator):
        self.checkpoint = checkpoint
        self.operator = operator

    def __call__(self, condition):
        ck = self.checkpoint
        x = (condition.double() - ck["xmean"]) / ck["xstd"]
        values = torch.einsum("nchw,co->nohw", x, ck["coefficient"])
        values = values.reshape(len(x), 20, 2, *x.shape[-2:])
        values = values * ck["std"] + ck["mean"]
        return (reconstruct(values, self.operator) if ck["kind"] == "R1" else values).float()


def fit_ridges(data, operator):
    xmean = data.x.double().mean((0, 2, 3), keepdim=True)
    xstd = data.x.double().std((0, 2, 3), keepdim=True).clamp_min(1e-8)
    design = ((data.x.double() - xmean) / xstd).permute(0, 2, 3, 1).reshape(-1, 88)
    gram = design.T @ design / len(design)
    models = {}
    for kind in ("R1", "D1"):
        label = encode_source(data.b.double(), operator) if kind == "R1" else data.b.double()
        mean = label.mean(0, keepdim=True)
        target = ((label - mean) / data.std).permute(0, 3, 4, 1, 2).reshape(-1, 40)
        moment = design.T @ target / len(design)
        for regularization in CFG["ridge"]:
            coefficient = torch.linalg.solve(gram + regularization * torch.eye(88, device=gram.device), moment)
            checkpoint = dict(kind=kind, regularization=regularization, coefficient=coefficient,
                              xmean=xmean, xstd=xstd, mean=mean, std=data.std.double())
            models[kind, regularization] = Ridge(checkpoint, operator)
    return models


class SmallModel(nn.Module):
    """All 20 conditional leads enter every output before spatial processing."""
    def __init__(self, kind, operator, diagonal, std):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv2d(88, 16, 1), nn.SiLU(),
            nn.Conv2d(16, 16, 3, padding=1, padding_mode="replicate"), nn.SiLU(), nn.Conv2d(16, 40, 1))
        nn.init.zeros_(self.encoder[-1].weight)
        nn.init.zeros_(self.encoder[-1].bias)
        self.kind = kind
        self.operator = diagonal if kind == "T_diag" else operator
        self.register_buffer("std", std.clone())

    def forward(self, condition):
        values = self.encoder(condition).reshape(len(condition), 20, 2, *condition.shape[-2:]) * self.std
        return values if self.kind == "D" else reconstruct(values, self.operator)


@torch.no_grad()
def predict(model, x):
    return torch.cat([model(batch) for batch in x.split(2)])


def admissible(candidate, native):
    for metric in ("mse", "smse"):
        tolerance = CFG["selection"]["mse_absolute_tolerance"]
        tolerance += CFG["selection"]["mse_relative_tolerance"] * abs(native[metric])
        if candidate[metric] > native[metric] + tolerance:
            return False
    return True


def select_score(data, bias):
    """Pairwise CRPS constant is shared by every candidate for this dataset."""
    differences = data.p - data.y[:, None] - bias[:, None]
    crps_without_constant = float((differences / data.std).abs().mean())
    error = data.p.double().mean(1) - data.y.double() - bias.double()
    return dict(crps=crps_without_constant, mse=float(error.square().mean()),
                smse=float((error / data.std).square().mean()))


def old_heads(data):
    """Frozen weights are only used on old windows without stored corrections."""
    result = {}
    for seed in CFG["seeds"]:
        file = ROOT / f"results/residual_dynamics_phase3/weights/crps_{seed}.pt"
        assert digest(file) == CFG["old_heads"][file.name]
        checkpoint = torch.load(file, map_location="cuda", weights_only=False)
        head = phase3.BiasHead().cuda().eval().requires_grad_(False)
        head.load_state_dict(checkpoint["selected"])
        history = data.history
        features = phase3.inputs(history, data.p.transpose(2, 3), None, "current_fields", data.norm)
        values = torch.cat([head(batch) * checkpoint["scale"] for batch in features.split(32)])
        result[seed] = values.reshape(len(data.p), 20, 2, 32, 64)
    return result


def score(data, bias, method, operator, seed=0, existing=None, existing_detail=None):
    """Return complete-block probability scores and implied-source diagnostics."""
    rows, cells = [], []
    true_source = encode_source(data.b.double(), operator)
    implied_source = encode_source(bias.double(), operator)
    for index, metadata in enumerate(data.meta):
        key = (metadata["trajectory"], str(metadata["start"]), str(seed))
        old = existing.get(key) if existing is not None else None
        # Follow phase3 float32 common shift, then double precision scoring.
        paths = (data.p[index] - bias[index, None]).double()
        truth = data.y[index].double()
        error = paths.mean(0) - truth
        mse = error.square().mean((-1, -2))
        smse = (error / data.std).square().mean((-1, -2))
        if old is None:
            crps = empirical_crps(paths / data.std, truth / data.std).mean((-1, -2))
            energy = float(energy_score(paths / data.std, truth / data.std))
        else:
            crps = torch.tensor([[float(existing_detail[(*key, str(lead), channel)]["crps"])
                                   for channel in "uv"] for lead in range(1, 21)], device=paths.device, dtype=torch.double)
            energy = float(old["energy"])
        r_error = implied_source[index, 1:] - true_source[index, 1:]
        r_mse = r_error.square().mean((-1, -2))
        r_smse = (r_error / data.std).square().mean((-1, -2))
        event = int(consecutive_event(truth.transpose(0, 1)[None], CFG["event_threshold"])[0])
        probability = float(consecutive_event(paths.transpose(1, 2), CFG["event_threshold"]).double().mean())
        value = dict(split=data.split, **metadata, method=method, seed=seed,
                     crps=float(crps.double().mean()), mse=float(mse.mean()), smse=float(smse.mean()),
                     energy=energy, r_mse=float(r_mse.mean()), r_smse=float(r_smse.mean()),
                     event=event, probability=probability, brier=(probability-event)**2)
        if old is not None:
            # Preserve published values exactly and only compute new diagnostics.
            for metric in ("crps", "mse", "energy", "brier"):
                value[metric] = float(old[metric])
        rows.append(value)
        for lead in range(20):
            for channel, name in enumerate("uv"):
                cells.append(dict(split=data.split, **metadata, method=method, seed=seed,
                    lead=lead+1, channel=name, crps=float(crps[lead, channel]),
                    mse=float(mse[lead, channel]), smse=float(smse[lead, channel]),
                    r_mse=float(r_mse[lead-1, channel]) if lead else "",
                    r_smse=float(r_smse[lead-1, channel]) if lead else ""))
    return rows, cells


def record_environment():
    configure()
    for name, expected in CFG["dependencies"].items():
        assert digest(ROOT / name) == expected, name
    assert digest(ROOT / "reliability/checkpoints/current.pt") == CFG["frozen"]["packaged_checkpoint_sha256"]
    dump(OUT / "environment.json", dict(torch=torch.__version__, cuda=torch.version.cuda,
         gpu=torch.cuda.get_device_name(), tf32=torch.backends.cuda.matmul.allow_tf32,
         config_sha256=CONFIG_HASH))


def dump_checkpoint(path, checkpoint):
    cpu = {key: value.detach().cpu() if torch.is_tensor(value) else value for key, value in checkpoint.items()}
    torch.save(cpu, path)
    return digest(path)
