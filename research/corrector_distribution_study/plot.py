"""Render the frozen paired score comparisons without loading model or path arrays."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def render(directory):
    with (directory/"test_contrasts.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    index = {(r["scope"],r["re"],r["contrast"],r["metric"]):r for r in rows}
    fig, axes = plt.subplots(1,3,figsize=(13.0,4.2),layout="constrained")
    panels = (
        ("Static calibration and frozen Correctors", ("S*-native","A-S*","C-S*"), ("crps",)),
        ("Same marginals, different ranks", ("A-A_native_rank","C-C_native_rank"),
         ("energy","time_mean_crps","increment5_crps")),
        ("Conditioning beyond static scale", ("recent-S*","state_memory-recent","state_memory-state"),
         ("crps",)),
    )
    display = {
        "S*-native":"S2 − native", "A-S*":"A − S2","C-S*":"C − S2",
        "A-A_native_rank":"A ranks − native ranks","C-C_native_rank":"C ranks − native ranks",
        "recent-S*":"Recent changes − S2", "state_memory-recent":"Add residual − recent only",
        "state_memory-state":"Residual − spread feature",
        "crps":"CRPS","energy":"Path ES","time_mean_crps":"Time-mean CRPS","increment5_crps":"Lag-5 CRPS",
    }
    colors = ("#1f6d77","#a0472e","#48537f")
    for ax, (title, contrasts, metrics) in zip(axes,panels):
        labels = []
        for j,metric in enumerate(metrics):
            for i,contrast in enumerate(contrasts):
                row = index["all","all",contrast,metric]
                point, low, high = (1000*float(row[k]) for k in ("delta","lower","upper"))
                y = j*len(contrasts) + i
                ax.plot([low,high],[y,y],linewidth=1.3,color=colors[i])
                ax.scatter([point],[y],s=28,color=colors[i],marker="D" if row["primary"]=="True" else "o")
                labels.append(display[contrast] if len(metrics)==1 else
                              f"{display[metric]} · {contrast[0]}")
        ax.set_yticks(np.arange(len(labels)),labels=labels,fontsize=8)
        ax.invert_yaxis()
        ax.axvline(0,color="#373c44",linewidth=.8)
        ax.set_title(title,loc="left",fontsize=10)
        ax.set_xlabel("Paired score difference (×1,000)")
        ax.grid(axis="x",alpha=.2)
        ax.spines[["top","right"]].set_visible(False)
    fig.suptitle("Corrector distribution study · reused test trajectories (n=12)",
                 x=.02,ha="left",fontsize=13)
    fig.savefig(directory/"mechanisms.png",dpi=155)
    plt.close(fig)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory",type=Path)
    render(parser.parse_args().directory)
