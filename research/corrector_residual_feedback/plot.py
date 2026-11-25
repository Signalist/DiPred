"""The thirteen requested Matplotlib figures, read only from scored CSV files."""
from shutil import copy2
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from research.corrector_residual_optimization.plot import select
from research.corrector_simple_baseline.plot import numbers,save
from research.corrector_simple_baseline.measure import groups
from research.corrector_window_study.data import ROOT,read_json

from .data import FIGURES
from .fit import read
from .model import ARMS,SLOTS

COLORS = dict(zip(ARMS,plt.get_cmap("tab10").colors))
COLORS["Native"] = "black"


def lead_panel(ax,leads,arms,key="native_pct",reference="Native"):
    for arm in arms:
        rows = select(leads,split="test",policy="selected",arm=arm)
        ax.plot(numbers(rows,"lead"),numbers(rows,key),color=COLORS[arm],marker=".",label=arm)
    ax.axhline(0,c="black",ls="--",lw=.8)
    ax.set(xlabel="Prediction lead",ylabel=f"MSE improvement over {reference} (%)",xticks=range(1,21))
    ax.grid(alpha=.2)
    ax.legend(ncol=5,fontsize=8)


def temporal_heatmaps(out,where):
    where.mkdir(parents=True,exist_ok=True)
    corr = read(out/"temporal_correlations.csv")
    fig,axes = plt.subplots(1,3,figsize=(12,4),layout="constrained")
    matrices = [numbers(select(corr,source=k),"pearson").reshape(20,20).T for k in SLOTS]
    limit = max(np.abs(m).max() for m in matrices)
    for k,m,ax in zip(SLOTS,matrices,axes):
        im = ax.imshow(m,origin="lower",extent=(.5,20.5,.5,20.5),vmin=-limit,vmax=limit,cmap="RdBu_r")
        ax.set(title=k.capitalize(),xlabel="Previous window lead",xticks=(1,5,10,15,20),yticks=(1,5,10,15,20))
    axes[0].set_ylabel("Current prediction lead")
    fig.colorbar(im,ax=axes,label="Pearson correlation")
    fig.savefig(where/"06_temporal_correlations.png",dpi=160)
    plt.close(fig)


def plot(out,where):
    where.mkdir(parents=True,exist_ok=True)
    scores = read(out/"scores.csv")
    leads = read(out/"leads.csv")
    selected = read_json(out/"selections.json")
    winner = selected["best_feedback"]
    fig,ax = plt.subplots(figsize=(9,4))
    for j,split in enumerate(("development","test")):
        own = [select(scores,split=split,policy="selected",arm=arm)[0] for arm in ARMS]
        y = numbers(own,"native_pct")
        e = np.vstack((y-numbers(own,"lower_pct"),numbers(own,"upper_pct")-y))
        e = np.maximum(e,0)
        xx = np.arange(8)+(j-.5)*.33
        ax.bar(xx,y,width=.33,label=split.capitalize(),yerr=e,capsize=3)
        a = float(own[0]["native_pct"])
        ax.axhline(a,lw=1,ls=("-." if j else ":"),color=("#df7b32" if j else "#3365a1"))
    ax.axhline(0,color="black",ls="--",lw=.8,label="Native")
    ax.set(xticks=np.arange(8),xticklabels=list(ARMS),xlabel="Feedback arm",ylabel="MSE improvement over Native (%)")
    ax.legend()
    save(fig,where,"01_factorial_results.png")

    for file,arms,key,reference in (
        ("02_lead_native.png",("Native",*ARMS),"native_pct","Native"),
        ("03_lead_over_A.png",tuple(ARMS)[1:],"baseline_pct","A"),
        ("04_single_inputs.png",tuple("ABCD"),"native_pct","Native"),
        ("05_combined_inputs.png",tuple("AEFGH"),"native_pct","Native")):
        fig,ax = plt.subplots(figsize=(11,4))
        lead_panel(ax,leads,arms,key,reference)
        save(fig,where,file)

    temporal_heatmaps(out,where)

    dep = read(out/"input_dependency.csv")
    kinds = (*SLOTS,"joint_true","joint_predicted")
    fig,axes = plt.subplots(8,5,figsize=(14,18),layout="constrained")
    vmax = max(float(r["response_rms"]) for r in dep)
    for row,arm in zip(axes,ARMS):
        for ax,kind in zip(row,kinds):
            own = select(dep,arm=arm,intervention=kind)
            if not own:
                ax.set_axis_off()
                continue
            m = numbers(own,"response_rms").reshape(20,20).T
            im = ax.imshow(m,origin="lower",extent=(.5,20.5,.5,20.5),vmin=0,vmax=vmax)
            ax.set(title=f"{arm} · {kind.replace('_',' ')}",xticks=(1,10,20),yticks=(1,10,20))
    fig.supxlabel("Previous window lead")
    fig.supylabel("Current prediction lead")
    fig.colorbar(im,ax=axes,label="Physical response RMS")
    fig.savefig(where/"07_input_dependency.png",dpi=160)
    plt.close(fig)

    train = read(out/"training.csv")
    fig,axes = plt.subplots(1,2,figsize=(12,4))
    for ax,part in zip(axes,("train","dev")):
        for arm in ARMS:
            own = select(train,run=arm)
            ax.plot(numbers(own,"step"),1e5*numbers(own,part+"_mse"),c=COLORS[arm],label=arm)
        ax.axhline(float(own[0][part+"_native"])*1e5,color="black",ls="--",label="Native")
        ax.set(xlabel="Updates",ylabel="Physical MSE × 10⁵",title="Training" if part=="train" else "Development")
        ax.grid(alpha=.2)
    axes[1].legend(ncol=3,fontsize=8)
    save(fig,where,"08_learning_curves.png")

    fig,axes = plt.subplots(1,2,figsize=(13,4))
    whole = [select(scores,split="test",policy="selected",arm=a)[0] for a in ARMS]
    for j,(key,factor,label) in enumerate((("C",2,"2C · effective gain"),
                                           ("correction_energy",1,"E · cost"),("G",1,"G · net gain"))):
        xx = np.arange(8)+(j-1)*.23
        axes[0].bar(xx,1e6*factor*numbers(whole,key),width=.23,label=label)
        for arm,style in (("A","--"),(winner,"-")):
            own = select(leads,split="test",policy="selected",arm=arm)
            axes[1].plot(numbers(own,"lead"),1e6*factor*numbers(own,key),ls=style,label=f"{arm} · {label}")
    axes[0].set(xticks=np.arange(8),xticklabels=list(ARMS),xlabel="Feedback arm")
    axes[1].set(xticks=range(1,21),xlabel="Prediction lead")
    for ax in axes:
        ax.axhline(0,color="black",lw=.7)
        ax.set_ylabel("Squared velocity × 10⁶")
        ax.legend(fontsize=7)
    save(fig,where,"09_gain_decomposition.png")

    fig,axes = plt.subplots(4,4,figsize=(15,8),layout="constrained")
    fields = read(out/"fixed_fields.csv")
    names = (*SLOTS,"current_true","base","feedback","base_remaining","feedback_remaining")
    titles = ("Previous true","Previous predicted","Previous remaining","Current true",
              "A prediction",f"{winner} prediction","A remaining",f"{winner} remaining")
    for c,label in enumerate(("u","v")):
        own = select(fields,channel=label)
        panels = [numbers(own,k).reshape(32,64) for k in names]
        lo = max(np.abs(m).max() for m in panels)
        for ax,m,title in zip(axes[c*2:c*2+2].flat,panels,titles):
            im = ax.imshow(m,origin="lower",cmap="RdBu_r",vmin=-lo,vmax=lo)
            ax.set(title=f"{label} · {title}",xticks=[],yticks=[])
        fig.colorbar(im,ax=axes[c*2:c*2+2],label=f"{label} velocity")
    fig.suptitle("Development 26700_0.h5 · start 400 · lead 10")
    fig.savefig(where/"10_fixed_physical_fields.png",dpi=160)
    plt.close(fig)

    regimes,regime_leads = (read(out/f"{file}.csv") for file in ("regimes","regime_leads"))
    re_ids = sorted({r["re"] for r in regimes if r["split"]=="test"},key=float)
    fig,axes = plt.subplots(2,len(re_ids),figsize=(5*len(re_ids),7),squeeze=False)
    for re,bar,curve in zip(re_ids,axes[0],axes[1]):
        own = [select(regimes,re=re,split="test",policy="selected",arm=a)[0] for a in ARMS]
        bar.bar(list(ARMS),numbers(own,"native_pct"),color=[COLORS[a] for a in ARMS])
        bar.set(title=f"Re {float(re):g}",ylabel="MSE improvement (%)")
        for arm in ARMS:
            vals = select(regime_leads,re=re,split="test",policy="selected",arm=arm)
            curve.plot(numbers(vals,"lead"),numbers(vals,"native_pct"),c=COLORS[arm],label=arm)
        curve.set(xticks=(1,5,10,15,20),xlabel="Prediction lead",ylabel="MSE improvement (%)")
        for ax in (bar,curve):
            ax.axhline(0,c="black",ls="--",lw=.7)
    axes[1][-1].legend(ncol=4,fontsize=8)
    save(fig,where,"11_regimes.png")

    effects = read(out/"factorial_effects.csv")
    fig,ax = plt.subplots(figsize=(11,4))
    for j,split in enumerate(("development","test")):
        own = select(effects,split=split,policy="selected")
        y = numbers(own,"G_effect")*1e6
        bounds = np.vstack((y-1e6*numbers(own,"lower"),1e6*numbers(own,"upper")-y)).clip(0)
        ax.bar(np.arange(len(own))+(j-.5)*.34,y,width=.34,yerr=bounds,capsize=3,label=split.capitalize())
    ax.axhline(0,c="black",lw=.8)
    ax.set(xticks=range(len(own)),xticklabels=[r["effect"].replace("_"," × ") for r in own],
           ylabel="Factorial gain effect × 10⁶")
    ax.legend()
    save(fig,where,"12_factorial_effects.png")

    rolls = read(out/"feedback_rollout.csv")
    fig,axes = plt.subplots(1,2,figsize=(12,4))
    for split,ax in zip(("development","test"),axes):
        for arm in selected["rollout"]:
            own = select(rolls,split=split,arm=arm)
            cohorts = groups(own,("chain_window",))
            for metric,style in (("fixed_reference_mse","--"),("self_mse","-")):
                pos,y = [],[]
                for (step,),block in sorted(cohorts.items(),key=lambda kv:int(kv[0][0])):
                    tracks = groups(block,("trajectory",))
                    pos.append(int(step))
                    y.append(1e5*mean(mean(float(r[metric]) for r in v) for v in tracks.values()))
                ax.plot(pos,y,c=COLORS[arm],ls=style,marker=".",label=f"{arm} {'self' if metric=='self_mse' else 'fixed'}")
        ax.set(title=split.capitalize(),xlabel="Consecutive issued window",ylabel="Physical MSE × 10⁵")
    axes[1].legend(fontsize=8,ncol=2)
    save(fig,where,"13_feedback_rollout.png")
    if where.resolve()==FIGURES.resolve():
        mirror = ROOT.parent/".User/Reports/corrector_residual_feedback"
        mirror.mkdir(parents=True,exist_ok=True)
        for path in sorted(where.glob("*.png")):
            copy2(path,mirror/path.name)
