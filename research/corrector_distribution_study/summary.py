"""Aggregate windows before seeds and bootstrap the original trajectories."""

import csv
from collections import defaultdict
from statistics import mean

import numpy as np

from .mechanism import METRICS

ALL_METRICS = (*METRICS, "crps_u", "crps_v")
SCOPES = {"all": (400,420,440,460), "original": (400,420), "later": (440,460)}
TWO_SEED = {arm + suffix for arm in "AC" for suffix in ("", "_mean", "_center", "_native_rank")}
TWO_SEED |= {f"P_{arm}_rank" for arm in "AC"}


def average(rows):
    return {key:mean(float(row[key]) for row in rows) for key in ALL_METRICS}


def aggregate(rows):
    seed_rows, trajectories, summary = [], [], []
    for scope,starts in SCOPES.items():
        windows = defaultdict(list)
        for row in rows:
            if row["start"] in starts:
                k = (row["method"],row["trajectory"],row["re"],row["seed"])
                windows[k].append(row)
        per_traj = defaultdict(list)
        for (method,traj,re,seed),values in sorted(windows.items()):
            assert len(values) == len({r["start"] for r in values})
            record = dict(scope=scope,method=method,trajectory=traj,re=re,seed=seed,
                          windows=len(values),**average(values))
            per_traj[method,traj,re].append(record)
            seed_rows.append(record)
        per_method = defaultdict(list)
        for (method,traj,re), values in sorted(per_traj.items()):
            expected = {42,43} if method in TWO_SEED else {""}
            assert len(values) == len(expected) and {r["seed"] for r in values} == expected
            assert len({r["windows"] for r in values}) == 1
            row = dict(scope=scope,method=method,trajectory=traj,re=re,windows=values[0]["windows"],
                       **average(values))
            per_method[method].append(row)
            trajectories.append(row)
        for method,values in sorted(per_method.items()):
            summary.append(dict(scope=scope,method=method,trajectories=len(values),**average(values)))
    return seed_rows,trajectories,summary


def comparisons(rows,simple,primary):
    indexed = defaultdict(dict)
    for row in rows:
        indexed[row["scope"],row["method"]][row["trajectory"]] = row
    contrasts = {"S*-native":(simple,"native"), "A-S*":("A",simple),"C-S*":("C",simple),
                 "C-A":("C","A")}
    for arm in "AC":
        contrasts.update({
            f"{arm}-native":(arm,"native"),f"{arm}_mean-native":(arm+"_mean","native"),
            f"{arm}_center-native":(arm+"_center","native"),
            f"{arm}-{arm}_native_rank":(arm,arm+"_native_rank"),
            f"P_{arm}_rank-native":(f"P_{arm}_rank","native"),
        })
    # Optional bounded explorations, all excluded from the confirmatory comparison family.
    if ("all","state") in indexed:
        contrasts.update({"recent-S*":("recent",simple), "state-S*":("state",simple),
                          "state-recent":("state","recent"), "state_memory-recent":("state_memory","recent"),
                          "state_memory-S*":("state_memory",simple), "state_memory-state":("state_memory","state")})
    if ("all","mean_on_static") in indexed:
        contrasts["mean_on_static-S*"] = ("mean_on_static",simple)
    report = []
    for scope in SCOPES:
        native = indexed[scope,"native"]
        re_groups = sorted({row["re"] for row in native.values()})
        for group in ["all"] + (re_groups if scope == "all" else []):
            files = sorted(k for k,row in native.items() if group == "all" or row["re"]==group)
            rng = np.random.default_rng(42)
            draws = rng.integers(len(files),size=(10000,len(files)))
            for contrast,(lhs,rhs) in contrasts.items():
                for metric in ("crps","energy","mean_mse","standardized_mean_mse","time_mean_crps",
                               "increment5_crps","coverage90","width90","ensemble_variance","crps_u","crps_v"):
                    delta = np.array([indexed[scope,lhs][f][metric]-indexed[scope,rhs][f][metric] for f in files])
                    simulated = delta[draws].mean(1)
                    main = scope=="all" and group=="all" and primary.get(contrast)==metric
                    lo,hi = np.quantile(simulated, [.00625,.99375] if main else [.025,.975])
                    report.append(dict(scope=scope,re=group,contrast=contrast,metric=metric,delta=float(delta.mean()),
                                       lower=float(lo),upper=float(hi),interval="98.75%" if main else "95%",
                                       primary=main,lower_trajectories=int((delta<0).sum()),
                                       higher_trajectories=int((delta>0).sum()),trajectories=len(files)))
    return report


def reproduce(rows, chosen, prior, split):
    table_name = "development_checkpoints.csv" if split=="development" else "per_seed_trajectory.csv"
    with (prior/table_name).open() as stream:
        old = list(csv.DictReader(stream))
    worst = 0.
    if split == "development":
        for config in chosen:
            method,seed,step = (config[k] for k in ("arm","seed","step"))
            actual = average([r for r in rows if r["method"]==method and r["seed"]==seed])
            expected, = [r for r in old if (r["arm"],int(r["seed"]),int(r["step"]))==(method,seed,step)]
            for key in METRICS[:-1]:
                worst = max(worst,abs(actual[key]-float(expected[key])))
    else:
        seeds,_,_ = aggregate(rows)
        history = {(r["method"],r["seed"],r["trajectory"]):r for r in old}
        for row in seeds:
            if row["scope"]!="all" or row["method"] not in ("native","A","C"):
                continue
            expected = history[row["method"], str(row["seed"]), row["trajectory"]]
            for key in METRICS[:-1]:
                worst = max(worst,abs(row[key]-float(expected[key])))
    assert worst < 2e-7, worst
    return worst
