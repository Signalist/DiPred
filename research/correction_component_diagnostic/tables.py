"""Apply the original scores and summarize equal-weight paired trajectories."""

from collections import defaultdict
from statistics import mean

import torch

from research.residual_uncertainty_diffusion.core import METRICS, csvwrite, score


COMPONENT = {
    "O-N": {"O": 1, "N": -1},
    "G-N": {"G": 1, "N": -1},
    "J-O": {"J": 1, "O": -1},
    "J-G": {"J": 1, "G": -1},
    "J-N": {"J": 1, "N": -1},
    "interaction": {"J": 1, "O": -1, "G": -1, "N": 1},
}
DECOMPOSITION = {
    "P1-P0": {"P1": 1, "P0": -1},
    "P_mean-P0": {"P_mean": 1, "P0": -1},
    "P_shape-P0": {"P_shape": 1, "P0": -1},
    "interaction": {"P1": 1, "P_mean": -1, "P_shape": -1, "P0": 1},
}
AMPLITUDE = {
    "half-O": {"half": 1, "O": -1},
    "half-J": {"half": 1, "J": -1},
    "half-N": {"half": 1, "N": -1},
}
CONDITION = {
    "E_off-J": {"E_off": 1, "J": -1},
    "W_off-J": {"W_off": 1, "J": -1},
}
IDENTITY = ("split", "trajectory_id", "Re", "start")
FD_ABS = 1e-7
FD_REL = 0.02
FD_ZERO = 1e-9


def identity(item):
    key = item["key"]
    return dict(split=key.split, trajectory_id=key.trajectory_id, Re=item["Re"], start=key.start)


def score_rows(meta, arm, state, prediction, target, norm):
    rows = [dict(**meta, arm=arm, state=state, **row) for row in score(prediction, target, norm)]
    for row in rows:
        assert abs(row["crps"] - row["fair_crps"] - row["pair_half"] / (len(prediction) - 1)) < 1e-12
    return rows


def decompose(native, joint):
    native, joint = native.double(), joint.double()
    mu0, mu1 = native.mean(0), joint.mean(0)
    p_mean, p_shape = mu1 + (native - mu0), mu0 + (joint - mu1)
    torch.testing.assert_close(p_shape.mean(0), mu0, atol=1e-12, rtol=0)
    torch.testing.assert_close(p_mean - p_mean.mean(0), native - mu0, atol=1e-12, rtol=0)
    for index in range(1, len(native)):
        torch.testing.assert_close(
            p_mean[index] - p_mean[:index], native[index] - native[:index], atol=1e-12, rtol=0,
        )
    return {"P0": native, "P1": joint, "P_mean": p_mean, "P_shape": p_shape}


def save_scores(destination, family, rows, contrasts):
    csvwrite(destination / f"{family}_SCORES.csv", rows)
    indexed = defaultdict(dict)
    grouping = (*IDENTITY, "arm", "channel")
    for row in rows:
        key = tuple(row[k] for k in grouping)
        assert row["state"] not in indexed[key]
        indexed[key][row["state"]] = row
    paired = []
    for key, states in sorted(indexed.items()):
        for name, terms in contrasts.items():
            differences = {m: sum(c * states[s][m] for s, c in terms.items()) for m in METRICS}
            paired.append(dict(zip(grouping, key), contrast=name, **differences))
    csvwrite(destination / f"{family}_PAIRED.csv", paired)
    metric_summary(destination / f"{family}_SUMMARY.csv", paired, "contrast", signed=True)
    metric_summary(destination / f"{family}_SCORE_SUMMARY.csv", rows, "state", signed=False)


def metric_summary(path, rows, field, signed):
    grouped = defaultdict(list)
    for row in rows:
        for re in ("all", row["Re"]):
            group = (row["split"], row["arm"], row["channel"], re, row[field])
            grouped[group].append(row)
    aggregate = []
    for group, values in sorted(grouped.items()):
        identifiers = {r["trajectory_id"] for r in values}
        assert len(identifiers) == len(values)
        for metric in METRICS:
            numbers = [r[metric] for r in values]
            row = dict(zip(("split", "arm", "channel", "Re", field), group),
                       metric=metric, trajectories=len(values), mean=mean(numbers))
            if signed:
                row.update(lower=sum(x < 0 for x in numbers), higher=sum(x > 0 for x in numbers),
                           equal=sum(x == 0 for x in numbers))
            aggregate.append(row)
    csvwrite(path, aggregate)


def fd_status(autograd, first, second):
    values = (autograd, first, second)
    same_sign = all(x > FD_ZERO for x in values) or all(x < -FD_ZERO for x in values)
    close = all(
        abs(a - b) <= FD_ABS + FD_REL * max(abs(a), abs(b))
        for a, b in ((autograd, first), (autograd, second), (first, second))
    )
    return "supported" if same_sign and close else "uncertain_nonsmooth"
