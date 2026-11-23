"""Paired whole-window and auxiliary scores on fixed forecasts."""
from collections import defaultdict

import numpy as np
import torch

from research.corrector.run import table
from research.corrector_window_study.model import empirical_crps

NAMES = ("crps", "energy", "mean_mse", "standardized_mean_mse", "member_mse",
         "ensemble_variance", "coverage90", "width90", "time_mean_crps")
CONTRASTS = {
    "B-A": {"B": 1, "A": -1},
    "D-B": {"D": 1, "B": -1},
    "E-B": {"E": 1, "B": -1},
    "D-E": {"D": 1, "E": -1},
    "F-D": {"F": 1, "D": -1},
    "C-A": {"C": 1, "A": -1},
    "C-B": {"C": 1, "B": -1},
    "D-C": {"D": 1, "C": -1},
    "interaction": {"D": 1, "B": -1, "C": -1, "A": 1},
    **{f"{arm}-native": {arm: 1, "native": -1} for arm in "ABCDEF"},
}
PRIMARY = ("B-A", "D-B", "E-B", "D-E")
SCOPES = {"all": (400, 420, 440, 460), "original": (400, 420), "later": (440, 460)}


def energy_score(predicted, target, scale):
    # Center by truth before a float64 Gram computation, preserving pair distances.
    errors = ((predicted - target[:, None]) / scale[:, None]).flatten(2).double()
    errors = errors / errors.shape[-1]**0.5
    sq = errors.square().sum(2)
    pairs = (sq[:,:,None] + sq[:,None,:] - 2 * errors.bmm(errors.transpose(1,2)))
    pairs.diagonal(dim1=1, dim2=2).zero_()
    return sq.sqrt().mean(1) - pairs.clamp_min(0).sqrt().mean((1,2)) / 2


def scores(predicted, target, scale):
    assert predicted.shape[1:4] == (32, 2, 20)
    residual = predicted - target[:, None]
    point = empirical_crps(residual / scale[:, None], torch.zeros_like(target))
    mean_error = residual.mean(1)
    lower, upper = torch.quantile(predicted, predicted.new_tensor([0.05, 0.95]), dim=1)
    average = (2, 3, 4)
    global_metrics = {
        "crps": point.mean((1, 2, 3, 4)),
        "energy": energy_score(predicted, target, scale),
        "mean_mse": mean_error.square().mean((1, 2, 3, 4)),
        "standardized_mean_mse": (mean_error / scale).square().mean((1, 2, 3, 4)),
        "member_mse": residual.square().mean((1, 2, 3, 4, 5)),
        "ensemble_variance": predicted.var(dim=1, unbiased=False).mean((1, 2, 3, 4)),
        "coverage90": ((target >= lower) & (target <= upper)).float().mean((1, 2, 3, 4)),
        "width90": (upper - lower).mean((1, 2, 3, 4)),
        "time_mean_crps": empirical_crps((residual / scale[:, None]).mean(3),
                                         torch.zeros_like(target[:, :, 0])).mean((1, 2, 3)),
    }
    assert all(torch.isfinite(value).all() for value in global_metrics.values())
    return {name: value.cpu().tolist() for name, value in global_metrics.items()}, \
        point.mean(average).cpu().tolist(), point.mean((1, 3, 4)).cpu().tolist()


def trajectory_rows(records, cases):
    rows, auxiliary = [], []
    for scope, starts in SCOPES.items():
        method_records = defaultdict(lambda: defaultdict(list))
        auxiliary_records = defaultdict(lambda: defaultdict(list))
        for method, method_scores in records.items():
            seed_count = 1 if method == "native" else 2
            assert len(method_scores) == seed_count * len(cases)
            for case, (value, channels, leads) in zip(cases * seed_count, method_scores):
                if case["start"] in starts:
                    key = (method, case["file"], case["re"])
                    method_records[key][case["start"]].append(value)
                    auxiliary_records[key][case["start"]].append(channels + leads)
        for (method, file, re), windows in method_records.items():
            assert set(windows) == set(starts)
            assert all(len(seeds) == (1 if method == "native" else 2) for seeds in windows.values())
            row = dict(scope=scope, method=method, trajectory=file, re=re, windows=len(starts))
            seed_windows = list(zip(*windows.values()))
            rows.append(dict(row, **{
                metric: np.mean([np.mean([window[metric] for window in seed])
                                 for seed in seed_windows]) for metric in NAMES
            }))
        for (method, _, _), windows in auxiliary_records.items():
            if scope == "all":
                seed_windows = zip(*windows.values())
                auxiliary.append((method, np.mean([np.mean(seed, axis=0) for seed in seed_windows],
                                                 axis=0)))
    return rows, auxiliary


def seed_trajectory_rows(records, cases, method, seed):
    by_traj = defaultdict(list)
    for case, (values, _, _) in zip(cases, records):
        by_traj[case["file"]].append(values)
    return [dict(method=method, seed=seed, trajectory=file,
                 **{metric: np.mean([row[metric] for row in windows]) for metric in NAMES})
            for file, windows in by_traj.items()]


def compare(rows):
    groups = defaultdict(dict)
    for row in rows:
        groups[row["scope"], row["method"]][row["trajectory"]] = row
    summary, comparisons = [], []
    for (scope, method), group in groups.items():
        summary.append(dict(scope=scope, method=method, trajectories=len(group),
                            **{metric: np.mean([r[metric] for r in group.values()]) for metric in NAMES}))
    for scope in SCOPES:
        files = sorted(groups[scope, "native"])
        rng = np.random.default_rng(42)
        sample = rng.integers(len(files), size=(10000, len(files)))
        assert all(set(groups[scope, arm]) == set(files) for arm in ("native", *"ABCDEF"))
        for name, coeffs in CONTRASTS.items():
            for metric in NAMES:
                diff = sum(coef * np.array([groups[scope, arm][file][metric] for file in files])
                           for arm, coef in coeffs.items())
                boot = diff[sample].mean(1)
                lo, hi = np.quantile(boot, [0.025, 0.975])
                lo_adjusted, hi_adjusted = np.quantile(boot, [0.00625, 0.99375])
                comparisons.append(dict(scope=scope, contrast=name, metric=metric, delta=diff.mean(),
                                        ci95_low=lo, ci95_high=hi,
                                        ci98_75_low=lo_adjusted if name in PRIMARY else "",
                                        ci98_75_high=hi_adjusted if name in PRIMARY else "",
                                        lower=int((diff < 0).sum()), trajectories=len(files)))
    return summary, comparisons


def save_results(out, records, seed_rows, cases):
    trajectories, auxiliary = trajectory_rows(records, cases)
    summary, comparisons = compare(trajectories)
    table(out / "per_trajectory.csv", trajectories)
    table(out / "per_seed_trajectory.csv", seed_rows)
    table(out / "scores.csv", summary)
    table(out / "contrasts.csv", comparisons)
    channel_lead = []
    for method in ("native", *"ABCDEF"):
        avg = np.mean([v for arm, v in auxiliary if arm == method], axis=0)
        channel_lead += [dict(method=method, axis="channel", label=c, crps=avg[i])
                         for i, c in enumerate(("u", "v"))]
        channel_lead += [dict(method=method, axis="lead", label=i + 1, crps=avg[i+2])
                         for i in range(20)]
    for re in sorted({r["re"] for r in trajectories}):
        for method in ("native", *"ABCDEF"):
            values = [r["crps"] for r in trajectories if r["scope"] == "all" and r["re"] == re
                      and r["method"] == method]
            channel_lead.append(dict(method=method, axis="re", label=re, crps=np.mean(values)))
    table(out / "auxiliary_crps.csv", channel_lead)
    draw(out, channel_lead)


def draw(out, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (absolute, difference) = plt.subplots(1, 2, figsize=(11.3, 4.3), layout="constrained")
    methods = {"native": "DiPred / F (step 0)", "A": "A · Summary", "B": "B · Full",
               "C": "C · Summary+Memory", "D": "D · Full+Memory", "E": "E · LongHistory"}
    colors = dict(native="#343b47", A="#30807b", B="#3272a8", C="#a55324",
                  D="#8666a4", E="#a48a27")
    by_method = {arm: sorted((r for r in rows if r["axis"] == "lead" and r["method"] == arm),
                             key=lambda r: int(r["label"])) for arm in (*methods, "F")}
    native = np.array([float(r["crps"]) for r in by_method["native"]])
    if not np.array_equal(native, [float(r["crps"]) for r in by_method["F"]]):
        methods["native"] = "Native DiPred"
        methods["F"] = "F · Full+Memory MSE"
        colors["F"] = "#a54c70"
    leads = np.arange(1, 21)
    for arm, label in methods.items():
        values = np.array([float(r["crps"]) for r in by_method[arm]])
        style = "--" if arm == "native" else "-"
        absolute.plot(leads, values, style, color=colors[arm], label=label)
        difference.plot(leads, 1000 * (values - native), style, color=colors[arm])
    absolute.legend(frameon=False, fontsize=8)
    absolute.set_ylabel("Standardized CRPS")
    difference.set_ylabel("CRPS difference vs DiPred (×1,000)")
    for ax in (absolute, difference):
        ax.set(xlabel="Physical lead", xticks=(1, 5, 10, 15, 20))
        ax.spines[["right", "top"]].set_visible(False)
        ax.grid(axis="y", alpha=0.18)
    fig.savefig(out / "crps_by_lead.png", dpi=170)
    plt.close(fig)
