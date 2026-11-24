"""Matplotlib views of physical scores, per-lead decomposition, and fixed fields."""
import csv
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from research.corrector_window_study.data import read_json


def read(path):
    with path.open() as file:
        return list(csv.DictReader(file))


def save(fig,out,name):
    fig.tight_layout()
    fig.savefig(out/name,dpi=160)
    plt.close(fig)


def numbers(rows,field):
    return np.array([float(r[field]) for r in rows])


def plot(out):
    train = read(out/"training.csv")
    by_policy = read_json(out/"FROZEN.json")["checkpoints"]
    fig,axes = plt.subplots(1,2,figsize=(10,4))
    for ax,name in zip(axes,("train","development")):
        ax.plot(numbers(train,"step"),1e5*numbers(train,f"{name}_mse"),marker="o",ms=3,label="Corrector")
        ax.axhline(float(train[0][f"{name}_native_mse"])*1e5,c="black",ls="--",label="Native")
        if name=="development":
            picked = by_policy["selected"]
            ax.scatter(picked["step"],picked["development_mse"]*1e5,c="#ab522d",label="Selected",zorder=4)
        ax.set(title=name.capitalize(),xlabel="Training updates",ylabel="Physical MSE × 10⁵")
        ax.grid(alpha=.2)
        ax.legend()
    save(fig,out,"01_learning_curves.png")

    test = [r for r in read(out/"leads.csv") if r["split"]=="test"]
    subsets = defaultdict(list)
    for row in test:
        subsets[row["policy"]].append(row)
    for rows in subsets.values():
        rows.sort(key=lambda r:int(r["lead"]))
    grouped = defaultdict(list)
    for key,label in (("selected","Development-selected"),("best_nonzero","Best nonzero"),("final","Final")):
        grouped[by_policy[key]["step"]].append((key,label))
    ticks = list(range(1,21))
    fig,ax = plt.subplots(figsize=(10,4))
    ax.axhline(0,c="black",ls="--",lw=1,label="Native, 0%")
    for step,policies in grouped.items():
        rows = subsets[policies[0][0]]
        caption = " / ".join(x[1] for x in policies)
        ax.plot(ticks,numbers(rows,"correction_pct"),marker="o",ms=4,label=f"{caption}, step {step}")
    ax.set(xlabel="Prediction lead",ylabel="Squared residual energy corrected (%)",xticks=ticks)
    ax.grid(alpha=.2)
    ax.legend()
    save(fig,out,"02_lead_correction_rate.png")

    chosen = subsets["selected"]
    fig,ax = plt.subplots(figsize=(10,4))
    ax.plot(ticks,1e5*numbers(subsets["native"],"mse"),marker="o",ms=3,c="black",ls="--",label="Native")
    ax.plot(ticks,1e5*numbers(chosen,"mse"),marker="o",ms=3,label=f"Selected, step {by_policy['selected']['step']}")
    ax.set(xlabel="Prediction lead",ylabel="Physical MSE × 10⁵",xticks=ticks)
    ax.grid(alpha=.2)
    ax.legend()
    save(fig,out,"03_lead_mse.png")

    fig,ax = plt.subplots(figsize=(10,4))
    ax.axhline(0,c="black",ls="--",lw=.8)
    for key,mult,label in (("C",2,"2C · effective gain"),("E",1,"E · correction cost"),("G",1,"G · net gain")):
        ax.plot(ticks,1e6*mult*numbers(chosen,key),marker="o",ms=3,label=label)
    ax.set(xlabel="Prediction lead",ylabel="Physical squared velocity × 10⁶",xticks=ticks)
    ax.grid(alpha=.2)
    ax.legend()
    save(fig,out,"04_lead_decomposition.png")

    fields = read(out/"fixed_development_field.csv")
    keys = ("truth","prediction","remaining")
    fig,axes = plt.subplots(2,3,figsize=(11,5),layout="constrained")
    for chan,axs in zip(("u","v"),axes):
        own = [r for r in fields if r["channel"]==chan]
        maps = [numbers(own,key).reshape(32,64) for key in keys]
        limit = max(np.abs(a).max() for a in maps)
        for ax,panel,title in zip(axs,maps,("Ground-truth residual","Predicted residual","Remaining residual")):
            im = ax.imshow(panel,origin="lower",cmap="RdBu_r",vmin=-limit,vmax=limit)
            ax.set_title(f"{chan} · {title}")
            ax.set(xticks=[],yticks=[])
        fig.colorbar(im,ax=axs,shrink=.72,label=f"{chan} velocity")
    case = read_json(out/"config.json")["plot"]
    fig.suptitle(f"Development {case['file']} · start {case['start']} · lead {case['lead']}")
    fig.savefig(out/"05_fixed_residual_fields.png",dpi=160)
    plt.close(fig)
