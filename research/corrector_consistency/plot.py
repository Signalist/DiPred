"""Render the preregistered case and the paired test summary."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
import numpy as np

from .run import ARMS_WITH_NATIVE

COLORS = dict(zip(ARMS_WITH_NATIVE, (
    "#4b5563", "#1e62a2", "#a54929", "#377a65", "#7b5c99",
    "#ad7a16", "#a73b68", "#3a727a",
)))


def save(fig, out, name):
    fig.savefig(out / f"{name}.png", dpi=155)
    plt.close(fig)


def render(out):
    with (out / "scores.csv").open() as f:
        rows = list(csv.DictReader(f))
    selected = {r["arm"]: r for r in rows if r["phase"] == "selected"}
    final = {r["arm"]: r for r in rows if r["phase"] == "step1000"}
    arms = list(ARMS_WITH_NATIVE)
    leads = np.arange(1, 21)
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained", sharey=True)
    for a, dataset, title in zip(axes, (selected, final), ("Development-selected · all at step 0", "Step 1000")):
        for arm in arms:
            a.plot(leads, [float(dataset[arm][f"lead_{k:02d}"]) for k in leads],
                   color=COLORS[arm], label=arm, ls="--" if arm == "Native" else "-")
        a.set(xlabel="Physical lead", title=title, xticks=(1, 5, 10, 15, 20))
    axes[0].set_ylabel("Original-scale field MSE")
    axes[0].legend(ncol=2, frameon=False, fontsize=8)
    save(fig, out, "01_lead_mse")

    fig, ax = plt.subplots(figsize=(8.5, 4), layout="constrained")
    indices = np.arange(len(arms))
    ax.bar(indices - .19, [float(selected[a]["improve_Native_pct"]) for a in arms], .36,
           label="Selected", color="#2665a0")
    ax.bar(indices + .19, [float(final[a]["improve_Native_pct"]) for a in arms], .36,
           label="Step 1000", color="#b95e35")
    ax.set(xticks=indices, xticklabels=arms, ylabel="MSE reduction vs Native (%)")
    ax.axhline(0, color="black", lw=.7)
    ax.legend(frameon=False)
    save(fig, out, "02_mse_improvement")

    keys = ("cosine", "magnitude", "d1_energy", "d2_energy")
    data = np.array([[float(final[a][k]) for k in keys] for a in arms])
    data = np.vstack((data, [float(selected["Native"][f"true_{k}"]) for k in keys]))
    fig, ax = plt.subplots(figsize=(7.1, 5.1), layout="constrained")
    ax.imshow(np.divide(data, data[-1]), cmap="PuBu", aspect="auto", vmin=0, vmax=1.2)
    for i, row in enumerate(data):
        for j, val in enumerate(row):
            ax.text(j, i, f"{val:.4g}", ha="center", va="center",
                    color="white" if val/data[-1, j] > .72 else "#161e27")
    ax.set(xticks=range(4), xticklabels=["Neighbor cosine", "Magnitude gap²", "First diff. energy",
                                         "Second diff. energy"],
           yticks=range(len(data)), yticklabels=arms + ["True residual"])
    ax.set_title("Step 1000 · shading relative to true residual")
    save(fig, out, "03_constraint_heatmap")

    cfg = json.loads((out / "config.json").read_text())
    c = cfg["plot_case"]
    with np.load(out / "plot_case.npz") as blob:
        fields = {k: blob[k] for k in blob.files}
    t = c["lead"] - 1
    y, x = c["grid"]
    title = f'{c["trajectory"]} · history start {c["start"]} · step 1000'
    yy, xx = np.mgrid[:32:3, :64:4]
    fig, axes = plt.subplots(3, 3, figsize=(11, 6.8), layout="constrained",
                            sharex=True, sharey=True)
    limit = max(np.linalg.norm(f[:, t], axis=0).max() for f in fields.values())
    for ax, name in zip(axes.flat, ["target", *arms]):
        u, v = fields[name][:, t]
        u, v = u[yy, xx], v[yy, xx]
        magnitude = np.hypot(u, v)
        if magnitude.any():
            divisor = np.where(magnitude > 0, magnitude, 1)
            vector = ax.quiver(xx, yy, u/divisor, v/divisor, magnitude, cmap="viridis",
                               norm=PowerNorm(gamma=.5, vmin=0, vmax=limit),
                               scale=.5, angles="xy", scale_units="xy", width=.004, minlength=.1)
        else:
            ax.text(.5, .5, "Zero correction", transform=ax.transAxes, ha="center", va="center")
        ax.set(title="True residual" if name == "target" else name, aspect="equal",
               xlim=(-1, 64), ylim=(-1, 32))
    fig.colorbar(vector, ax=axes, shrink=.70, label="Vector magnitude")
    fig.suptitle(f"{title} · lead {t+1} · equal-length direction arrows")
    save(fig, out, "04_residual_vectors")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for ch, ax in enumerate(axes):
        for name in ["target", *arms]:
            ax.plot(leads, fields[name][ch, :, y, x], color=COLORS.get(name, "#141414"),
                    label="True residual" if name == "target" else name,
                    ls="--" if name in ("target", "Native") else "-")
        ax.set(title=("u", "v")[ch], xlabel="Physical lead", xticks=(1, 5, 10, 15, 20))
    axes[0].set_ylabel("Residual at fixed grid point")
    axes[1].legend(frameon=False, ncol=2, fontsize=8)
    fig.suptitle(f"{title} · grid ({y}, {x})")
    save(fig, out, "05_point_timeseries")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for derivative, ax in enumerate(axes, 1):
        for name in ["target", *arms]:
            z = np.diff(fields[name], n=derivative, axis=1)
            ax.plot(leads[derivative:], np.sqrt(np.mean(z**2, axis=(0, 2, 3))),
                    color=COLORS.get(name, "#141414"), label="True residual" if name == "target" else name,
                    ls="--" if name in ("Native", "target") else "-")
        ax.set(title=("First difference", "Second difference")[derivative-1],
               xlabel="Ending physical lead")
    axes[0].set_ylabel("Whole-field difference RMS")
    axes[1].legend(frameon=False, ncol=2, fontsize=8)
    fig.suptitle(title)
    save(fig, out, "06_time_differences")

    fig, axes = plt.subplots(8, 4, figsize=(10, 16), layout="constrained")
    abs_limit = max(np.abs(f[:, t]).max() for f in fields.values())
    abs_limit = max(abs_limit, max(np.abs(fields["target"][:, t] - fields[a][:, t]).max() for a in arms))
    for row, arm in zip(axes, arms):
        residual = fields[arm][:, t]
        error = fields["target"][:, t] - residual
        for ax, z in zip(row, (*residual, *error)):
            im = ax.imshow(z, vmin=-abs_limit, vmax=abs_limit, cmap="RdBu_r", origin="lower")
            ax.set(xticks=[], yticks=[])
        row[0].set_ylabel(arm)
    for ax, label in zip(axes[0], ("Correction u", "Correction v", "Error u", "Error v")):
        ax.set_title(label)
    fig.colorbar(im, ax=axes, shrink=.65)
    fig.suptitle(f"{title} · lead {t+1} · Y−Q error")
    save(fig, out, "07_correction_and_error_fields")

    with (out / "trajectories.csv").open() as f:
        tracks = list(csv.DictReader(f))
    fig, ax = plt.subplots(figsize=(7.1, 4.6), layout="constrained")
    for arm in arms[1:]:
        subset = [r for r in tracks if r["arm"] == arm and r["phase"] == "step1000"]
        ax.scatter([float(r["correction_cosine"]) for r in subset],
                   [float(r["improve_Native_pct"]) for r in subset], s=18, alpha=.7,
                   color=COLORS[arm], label=arm)
    ax.axhline(0, color="#444", lw=.8)
    ax.set(xlabel="Correction–truth residual cosine", ylabel="Trajectory MSE reduction vs Native (%)", title="Step 1000")
    ax.legend(frameon=False, ncol=4)
    save(fig, out, "08_direction_vs_skill")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    render(parser.parse_args().out)
