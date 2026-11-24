"""Plot diagnostics, train/development curves, and per-lead correction."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .fit import MODES, SEEDS
from .legacy import OUT

COLORS = dict(zip(MODES,["#475569","#32678f","#007f73","#af6a21","#97458b"]))


def read(path):
    with path.open() as file:
        return list(csv.DictReader(file))


def save(fig,path):
    fig.tight_layout()
    fig.savefig(path,dpi=160)
    plt.close(fig)


def plot(out):
    train = read(out/"training.csv")
    fig, axes = plt.subplots(1,3,figsize=(14,4))
    series = (("train_mse","Training, raw"),("dev_mse","Development, raw"),
              ("dev_fitted_mse","Development, fitted gain"))
    for ax,(field,title) in zip(axes,series):
        for arm in MODES:
            selected = [r for r in train if r["mode"]==arm]
            steps = sorted({int(r["step"]) for r in selected})
            values = np.array([[float(r[field])*1e5 for r in selected if int(r["seed"])==s]
                               for s in SEEDS])
            ax.plot(steps,values.mean(0),label=arm,color=COLORS[arm],marker=".",lw=1.7)
            ax.fill_between(steps,values.min(0),values.max(0),color=COLORS[arm],alpha=.13)
        if field != "train_mse":
            ax.axhline(float(train[0]["dev_native_mse"])*1e5,color="black",ls="--",lw=1,label="Native")
        ax.set(title=title,xlabel="Updates",ylabel="Physical MSE × 10⁵")
        ax.grid(alpha=.2)
    axes[-1].legend(loc="best",fontsize=8)
    save(fig,out/"01_learning_curves.png")

    diag = read(out/"diagnostic_curves.csv")
    fig,axes = plt.subplots(2,2,figsize=(10,7))
    for ax,label,title in zip(axes.flat,("bias","channel_scale","real_1","real_4"),
                              ("Fixed channel bias","Fixed channel scaling","One training window",
                               "Four training windows")):
        for name,color in (("Point","#32678f"),("Block","#007f73")):
            rows = [r for r in diag if r["probe"]==label and r["architecture"]==name]
            values = np.array([float(r["physical_mse"]) for r in rows])
            ax.plot([int(r["step"]) for r in rows],100*values/values[0],marker=".",label=name,color=color)
        ax.set(title=title,xlabel="Updates",ylabel="MSE / initial MSE (%)",yscale="log")
        ax.grid(alpha=.2)
    axes[0,0].legend()
    save(fig,out/"02_diagnostic_fits.png")

    leads = read(out/"leads.csv")
    fig,axes = plt.subplots(1,2,figsize=(12,4),sharex=True)
    for ax,policy,title in zip(axes,("selected","step1000_raw"),
                                ("Development-selected","1000 updates, raw correction")):
        for arm in MODES:
            rows = [r for r in leads if r["mode"]==arm and r["policy"]==policy]
            ax.plot([int(r["lead"]) for r in rows],[float(r["correction_pct"]) for r in rows],
                    marker=".",lw=1.5,label=arm,color=COLORS[arm])
        ax.axhline(0,color="black",ls="--",lw=1)
        ax.set(title=title,xlabel="Forecast lead",ylabel="Squared residual energy corrected (%)",
               xticks=(1,5,10,15,20))
        ax.grid(alpha=.2)
    axes[0].legend(loc="best",fontsize=8)
    save(fig,out/"03_lead_correction_rates.png")


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,default=OUT)
    plot(parser.parse_args().out)
