"""Paired factorial contrasts and observed-history interventions."""
from itertools import combinations, product

import numpy as np
import torch

from research.corrector_incremental_structure.model import inputs, residual
from research.corrector_residual_optimization.fit import track_weights
from research.corrector_simple_baseline.measure import groups

from .core import load_head

FACTORS = ("increment", "history", "match2")


def arm_for(bits):
    increment, history, match = bits
    return ("I" if increment else "D") + ("B" if history else "0") + ("_match2" if match else "")


def factorial_cells():
    return {arm_for(bits): bits for bits in product((0, 1), repeat=3)}


def effect_terms():
    for order in (1, 2, 3):
        for axes in combinations(range(3), order):
            name = "*".join(FACTORS[i] for i in axes)
            weights = {}
            for arm, bits in factorial_cells().items():
                sign = (-1) ** sum(bits[i] == 0 for i in axes)
                weights[arm] = -sign / 2 ** (3 - order)
            yield "factorial", name, order, weights
    for axes in ((0,), (1,), (2,), (1, 2), (0, 2)):
        rest = tuple(i for i in range(3) if i not in axes)
        for condition in product((0, 1), repeat=len(rest)):
            fixed = dict(zip(rest, condition))
            label = "*".join(FACTORS[i] for i in axes)
            label += "|" + ",".join(f"{FACTORS[i]}={fixed[i]}" for i in rest)
            weights = {}
            for arm, bits in factorial_cells().items():
                if all(bits[i] == a for i, a in fixed.items()):
                    weights[arm] = -(-1) ** sum(bits[i] == 0 for i in axes)
            yield "conditional", label, len(axes), weights


def factorial_scores(rows, draws, axes=()):
    results = []
    dims = ("split", "policy", *axes)
    cells = factorial_cells()
    for keys, category in groups(rows, dims).items():
        values = {name: {} for name in cells}
        s3 = {}
        for row in category:
            if row["arm"] in values:
                values[row["arm"]][row["trajectory"]] = float(row["mse"])
            elif row["arm"] == "S3":
                s3[row["trajectory"]] = float(row["mse"])
        tracks = sorted(s3)
        assert tracks and all(set(v) == set(tracks) for v in values.values())
        rng = np.random.default_rng(42)
        indices = rng.integers(len(tracks), size=(draws, len(tracks)))
        matrix = np.array([[values[k][t] for t in tracks] for k in cells])
        ref_mse = float(np.mean(list(s3.values())))
        for kind, effect, order, weights in effect_terms():
            vector = np.array([weights.get(arm, 0.) for arm in cells])
            assert abs(vector.sum()) < 1e-15
            sample = vector @ matrix
            value = float(sample.mean())
            low, top = np.percentile(sample[indices].mean(axis=1), (2.5, 97.5))
            results.append(dict(zip(dims, keys), kind=kind, effect=effect, order=order,
                                mse_reduction=value, lower=float(low), upper=float(top),
                                pct_of_s3=100 * value / ref_mse, trajectories=len(tracks),
                                positive=int((sample > 0).sum())))
    return results


def same_re_permutation(cases):
    assigned = list(range(len(cases)))
    by_re = {}
    for i, c in enumerate(cases):
        by_re.setdefault(c["re"], []).append(i)
    for ids in by_re.values():
        ids.sort(key=lambda i: (cases[i]["file"], cases[i]["start"]))
        if len(ids) == 1:
            continue
        rotations = [ids[k:] + ids[:k] for k in range(1, len(ids))]
        donor = max(rotations, key=lambda indexes: sum(cases[i]["file"] != cases[j]["file"]
                                                        for i, j in zip(ids, indexes)))
        for i, j in zip(ids, donor):
            assigned[i] = j
    return assigned


@torch.inference_mode()
def interventions(models, view, basis, plan, split, policy):
    cases = view.data.cases
    raw = view.true
    donor = same_re_permutation(cases)
    changed = sum(i != j for i, j in enumerate(donor))
    assert sorted(donor) == list(range(len(cases)))
    assert all(cases[i]["re"] == cases[j]["re"] for i, j in enumerate(donor))
    size = raw.shape[-1] * raw.shape[-2]
    perm = torch.randperm(size, generator=torch.Generator().manual_seed(plan["interventions"]["spatial_seed"]))
    permutations = {"original": raw, "zero": torch.zeros_like(raw), "same_re": raw[donor],
                    "reverse": raw.flip(2), "spatial": raw.flatten(-2)[..., perm].reshape_as(raw)}
    for n in plan["interventions"]["boundary"]:
        keep = torch.zeros_like(raw)
        keep[:, :, -n:] = raw[:, :, -n:]
        permutations[f"retain_last_{n}"] = keep
        permutations[f"mask_last_{n}"] = raw - keep
    for lead in range(20):
        changed_history = raw.clone()
        changed_history[:, :, lead] = 0
        permutations[f"zero_source_{lead+1}"] = changed_history
    weights = track_weights(view.data)[:, None]
    remaining = view.target - view.current.double()
    rows = []
    for name in ("IB", "IB_match2"):
        record = models[name]
        model = load_head(record, record[policy], basis, raw.device)
        model.eval()
        cfg = record["config"]["model"]

        def evaluate(intervened):
            out = [residual(view, ix, cfg, model(inputs(view, ix, cfg, intervened=intervened[ix])))
                   for ix in torch.arange(len(cases), device=raw.device).split(8)]
            return torch.cat(out)

        fixed = evaluate(raw)
        for kind, source in permutations.items():
            pred = fixed if kind == "original" else evaluate(source)
            delta = pred - view.current.double()
            def aggregate(tensor):
                return (tensor.mean((1, 3, 4)) * weights).sum(0)
            mse = aggregate((pred - view.target).square())
            change = aggregate((pred - fixed).square()).sqrt()
            c = aggregate(delta * remaining)
            u = aggregate(delta.square())
            gain = 2 * c - u
            assert torch.allclose(gain, aggregate(remaining.square()) - mse, atol=1e-15, rtol=0)
            for j in range(20):
                rows.append(dict(split=split, policy=policy, arm=name, intervention=kind, lead=j+1,
                                 mse=float(mse[j]), C_delta=float(c[j]), U_delta=float(u[j]),
                                 G_delta=float(gain[j]), response_rms=float(change[j]),
                                 donor_windows=changed if kind == "same_re" else ""))
    return rows
