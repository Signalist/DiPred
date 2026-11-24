"""Matplotlib figures for each executed stage and for the gated gradient diagnostic."""
from collections import defaultdict
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from research.corrector_window_study.data import read_json
from research.corrector_simple_baseline.plot import numbers, save

from .fit import read
from .model import CHANNELS, LOSSES


def select(rows,**fields):
    return [r for r in rows if all(r[k]==str(v) for k,v in fields.items())]


def lead_chart(ax,leads,policies,field="correction_pct"):
    for label,name in policies.items():
        rows = select(leads,run=name,split="test")
        ax.plot(numbers(rows,"lead"),numbers(rows,field),marker=".",label=label)
    if field=="correction_pct":
        ax.axhline(0,c="black",ls="--",lw=.75)
    ax.set(xticks=range(1,21),xlabel="Prediction lead",
           ylabel="Corrected ground residual (%)" if field=="correction_pct" else "Physical MSE")
    ax.grid(alpha=.2)
    ax.legend(fontsize=8)


def plot_inputs(out):
    grid = read(out/"input_dependency.csv")
    training = read(out/"training.csv")
    fig,axes = plt.subplots(1,5,figsize=(16,3.6),layout="constrained")
    matrices = [numbers(select(grid,run=name),"response_rms").reshape(20,20).T for name in CHANNELS]
    vmax = max(float(a.max()) for a in matrices)
    for ax,name,a in zip(axes,CHANNELS,matrices):
        im = ax.imshow(a,origin="lower",extent=(.5,20.5,.5,20.5),vmin=0,vmax=vmax)
        ax.set(title=name,xlabel="History frame",xticks=(1,5,10,15,20),yticks=(1,5,10,15,20))
    axes[0].set_ylabel("Prediction lead")
    fig.colorbar(im,ax=axes,label="Physical response RMS")
    fig.savefig(out/"01_input_dependency.png",dpi=160)
    plt.close(fig)

    fig,axes = plt.subplots(1,2,figsize=(11,4))
    for ax,which in zip(axes,("train","dev")):
        for name in CHANNELS:
            own = select(training,run=name)
            ax.plot(numbers(own,"step"),1e5*numbers(own,which+"_mse"),marker=".",label=name)
        ax.axhline(float(training[0][which+"_native"])*1e5,c="black",ls="--",label="Native")
        ax.set(title={"train":"Training","dev":"Development"}[which],xlabel="Updates",ylabel="Physical MSE × 10⁵")
        ax.grid(alpha=.2)
    axes[1].legend()
    save(fig,out,"02_input_learning_curves.png")



def plot_hyperparameters(out,selected,hyper):
    by_name = {r["run"]:r for r in hyper}
    stages = [("Learning rate",[selected["input_best"],"lr_1e-4","lr_1e-3"],"lr"),
              ("Width",[selected["lr_best"],"width_16","width_32"],"width"),
              ("Budget",[selected["width_best"],"budget_6000"],"updates")]
    if "wd_1e-3" in by_name:
        plan = selected["runs"]["wd_1e-3"]["config"]
        original = next(k for k,v in selected["runs"].items()
                        if not v["config"]["aux"] and v["config"]==dict(plan,weight_decay=1e-5))
        stages.append(("Weight decay",[original,"wd_1e-3"],"weight_decay"))
    fig,axes = plt.subplots(1,len(stages),figsize=(4*len(stages),4),squeeze=False)
    for ax,(title,names,field) in zip(axes[0],stages):
        own = sorted((by_name[n] for n in names),key=lambda r:float(r[field]))
        xx = range(len(own))
        ax.plot(xx,numbers(own,"train_gain_pct"),marker="o",label="Training")
        ax.plot(xx,numbers(own,"dev_gain_pct"),marker="o",label="Development")
        ax.axhline(0,c="black",ls="--",lw=.8)
        ax.set(title=title,ylabel="MSE improvement (%)",xticks=list(xx),
               xticklabels=[f"{float(r[field]):g}" for r in own])
        ax.grid(alpha=.2)
    axes[0][-1].legend()
    save(fig,out,"03_hyperparameters.png")


def plot(out):
    plot_inputs(out)
    selected = read_json(out/"selections.json")
    base = selected["base"]
    leads = read(out/"leads.csv")
    scores = read(out/"scores.csv")
    plot_hyperparameters(out,selected,read(out/"hyperparameter_search.csv"))

    fig,ax = plt.subplots(figsize=(11,4))
    ranking = dict(Native="Native",**{k:k for k in CHANNELS})
    ranking[f"Base · {base}"] = base
    if base in CHANNELS:
        del ranking[base]
    lead_chart(ax,leads,ranking)
    save(fig,out,"04_lead_rates.png")
    fig,ax = plt.subplots(figsize=(10,4))
    lead_chart(ax,leads,{"Native":"Native",f"Base · {base}":base},"mse")
    save(fig,out,"05_lead_mse.png")

    sub = select(leads,split="test",run=base)
    fig,ax = plt.subplots(figsize=(10,4))
    for factor,key,label in ((2,"C","2C · effective gain"),(1,"E","E · correction cost"),(1,"G","G · net gain")):
        ax.plot(numbers(sub,"lead"),1e6*factor*numbers(sub,key),label=label,marker=".")
    ax.axhline(0,c="black",ls="--",lw=.8)
    ax.set(xticks=range(1,21),xlabel="Prediction lead",ylabel="Squared velocity × 10⁶")
    ax.legend()
    ax.grid(alpha=.2)
    save(fig,out,"06_lead_decomposition.png")

    whole = select(read(out/"regimes.csv"),split="test",run=base)
    regimes = read(out/"regime_leads.csv")
    fig,axes = plt.subplots(1,2,figsize=(13,4),gridspec_kw={"width_ratios":[1,2]})
    labels = [str(int(float(r["re"]))) for r in whole]
    axes[0].bar(labels,numbers(whole,"correction_pct"))
    axes[0].set(xlabel="Re",ylabel="Full-window improvement (%)")
    for r,label in zip(whole,labels):
        rows = select(regimes,split="test",run=base,re=r["re"])
        axes[1].plot(numbers(rows,"lead"),numbers(rows,"correction_pct"),label=label,marker=".")
    axes[1].set(xticks=range(1,21),xlabel="Prediction lead",ylabel="Correction rate (%)")
    for ax in axes:
        ax.axhline(0,c="black",ls="--",lw=.8)
        ax.grid(alpha=.2)
    axes[1].legend(title="Re")
    save(fig,out,"07_regime_rates.png")

    fields = read(out/"fixed_residual.csv")
    fig,axes = plt.subplots(2,3,figsize=(11,5),layout="constrained")
    for channel,axs in zip(("u","v"),axes):
        own = select(fields,channel=channel)
        panels = [numbers(own,k).reshape(32,64) for k in ("truth","prediction","remaining")]
        bounds = max(np.abs(v).max() for v in panels)
        for ax,a,title in zip(axs,panels,("Ground truth","Predicted","Remaining")):
            im = ax.imshow(a,origin="lower",cmap="RdBu_r",vmin=-bounds,vmax=bounds)
            ax.set(title=f"{channel} · {title}",xticks=[],yticks=[])
        fig.colorbar(im,ax=axs,shrink=.7,label=f"{channel} velocity")
    fig.suptitle("Development 26700_0.h5 · start 400 · lead 10")
    fig.savefig(out/"08_fixed_residual_fields.png",dpi=160)
    plt.close(fig)

    gradients = read(out/"gradient_diagnostics.csv")
    fig,ax = plt.subplots(figsize=(10,4))
    for j,(field,label) in enumerate((("gradient_l2","Raw"),("fixed_0_1_gradient","λ=0.1"),
                                      ("calibrated_rho_0_1_gradient","ρ=0.1"),("calibrated_rho_0_3_gradient","ρ=0.3"))):
        xx = np.arange(len(LOSSES))+(j-1.5)*.19
        ax.bar(xx,[mean(numbers(select(gradients,loss=k),field)) for k in LOSSES],width=.19,label=label)
    mse_norm = mean(numbers(select(gradients,loss="mse"),"gradient_l2"))
    ax.axhline(mse_norm,c="black",ls="--",lw=1,label="MSE")
    ax.set(xticks=range(len(LOSSES)),xticklabels=LOSSES,ylabel="Mean parameter gradient L2",yscale="log")
    ax.legend(ncol=3)
    save(fig,out,"09_loss_gradient_scales.png")

    aux = selected["auxiliary"]
    if aux:
        auxiliary_figures(out,leads,aux,base)
    policies = {"Native":"Native","S0":"S0","Input best":selected["input_best"],"MSE tuned":base}
    if aux:
        policies["Loss tuned"] = aux["best"]
    fig,ax = plt.subplots(figsize=(9,4))
    for j,name in enumerate(("development","test")):
        y = [float(select(scores,split=name,run=k)[0]["correction_pct"]) for k in policies.values()]
        ax.bar(np.arange(len(y))+(j-.5)*.34,y,width=.34,label=name.capitalize())
    ax.axhline(0,c="black",ls="--",lw=.8)
    ax.set(ylabel="MSE improvement over Native (%)",xticks=list(range(len(policies))),xticklabels=list(policies))
    ax.legend()
    save(fig,out,"12_research_stages.png")


def auxiliary_figures(out,leads,aux,base):
    sweep = read(out/"loss_search.csv")
    ref = float(select(read(out/"scores.csv"),split="development",run=base)[0]["correction_pct"])
    fig,axes = plt.subplots(1,2,figsize=(12,4))
    for ax,stage,label in zip(axes,("raw","balanced"),("λ","ρ")):
        for letter,name in zip("BCDEF",LOSSES):
            own = [r for r in sweep if r["candidate"].startswith(name+"_") and r["stage"]==stage]
            if stage=="raw":
                own += select(sweep,candidate=f"fixed_{letter}")
            own.sort(key=lambda r:float(r["coordinate"]))
            ax.plot(numbers(own,"coordinate"),numbers(own,"dev_gain_pct"),label=name,marker=".")
        ax.axhline(0,c="black",ls=":",label="Native")
        ax.axhline(ref,c="#555555",ls="--",label="Pure MSE")
        ax.set(xscale="log",xlabel=label,ylabel="Development correction rate (%)")
    axes[1].legend(fontsize=8)
    save(fig,out,"10_loss_sensitivity.png")

    fixed = {name:aux["fixed"][letter] for letter,name in zip("BCDEF",LOSSES)}
    for tag,entries in (("fixed",dict(fixed,Joint=aux["fixed"]["G"])),
                        ("tuned",dict(aux["tuned"],Joint=aux["best_joint"]))):
        arms = {"Native":"Native","Pure MSE":base,**entries}
        fig,axes = plt.subplots(1,2,figsize=(14,4.5))
        lead_chart(axes[0],leads,arms)
        target = numbers(select(leads,split="test",run=base),"mse")
        for label,name in arms.items():
            values = numbers(select(leads,split="test",run=name),"mse")
            axes[1].plot(range(1,21),100*(1-values/target),label=label,marker=".")
        axes[1].axhline(0,c="black",ls="--",lw=.8)
        axes[1].set(xticks=range(1,21),xlabel="Prediction lead",ylabel="MSE improvement over pure MSE (%)")
        axes[1].grid(alpha=.2)
        save(fig,out,f"11_loss_ablation_{tag}.png")
