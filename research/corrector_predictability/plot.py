"""Render development curves, frozen comparisons and a predeclared case."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
import numpy as np


def render(out):
    def read(name):
        with (out/f"{name}.csv").open() as f:
            return list(csv.DictReader(f))
    def save(fig,name):
        fig.savefig(out/f"{name}.png",dpi=150)
        plt.close(fig)
    plt.rcParams.update({"axes.spines.top":False,"axes.spines.right":False})
    scores={(r["mode"],r["policy"]):r for r in read("scores")}
    old=list("ABG")
    colors={"A":"#286399","B":"#a95025","G":"#3c7c70"}
    grid=read("stage0_development_alpha")
    fig, ax = plt.subplots(figsize=(7.4,4.0),layout="constrained")
    for arm in old:
        group=[r for r in grid if r["arm"]==arm]
        base=float(group[0]["native_mse"])
        c,e=(float(group[0][key]) for key in ("inner","energy"))
        a=np.linspace(0,1,101)
        ax.plot(a,(base+a*a*e-2*a*c)*1e5,color=colors[arm],label=arm)
        ax.scatter([float(r["alpha"]) for r in group if r["mode"]=="fixed"],
                   [float(r["mse"])*1e5 for r in group if r["mode"]=="fixed"],
                   s=16,color=colors[arm])
        ax.scatter(float(group[0]["alpha"]),float(group[0]["mse"])*1e5,s=55,marker="*",
                   color=colors[arm])
    ax.set(xlabel="Global α",ylabel="Development MSE (×10⁵)")
    ax.legend(frameon=False,title="Frozen method")
    save(fig,"01_alpha_mse")

    oracle=read("stage0_ORACLE")
    fig, ax=plt.subplots(figsize=(8,4.3),layout="constrained")
    names=("Dev-fit","Global","Lead","Channel","Trajectory","Window")
    index=np.arange(len(names))
    for j,arm in enumerate(old):
        values=[float(scores["old-"+arm,"selected"]["improvement_pct"])]
        for name in names[1:]:
            key="all" if name=="Global" else "ALL"
            row,=[r for r in oracle if (r["arm"],r["scope"],r["group"],r["bound"])
                  == (arm,name.lower(),key,"0..1")]
            values.append(float(row["improvement_pct"]))
        ax.bar(index+(j-1)*.24,values,.24,label=arm,color=colors[arm])
    ax.set(xticks=index,xticklabels=names,ylabel="Test reduction vs Native (%)",
           title="Frozen α and test-oracle granularity")
    ax.legend(frameon=False)
    save(fig,"02_legal_and_oracle")

    training=read("training")
    fig, axs=plt.subplots(1,2,figsize=(11,4),layout="constrained")
    steps=(0,50,100,250,500,750,1000)
    for mode,color in (("Small","#2b6595"),("Expanded","#b25631")):
        for ax,k in zip(axs,("training_mse","dev_raw_mse")):
            points=[[float(r[k]) for r in training if r["mode"]==mode and int(r["step"])==s]
                    for s in steps]
            avg=np.mean(points,axis=1)*1e5
            ax.plot(steps,avg,color=color,label=mode,marker="o",ms=3)
            ax.fill_between(steps,np.min(points,axis=1)*1e5,np.max(points,axis=1)*1e5,
                            color=color,alpha=.12)
    for ax,title in zip(axs,("Fixed train windows","Development windows")):
        ax.set(title=title,xlabel="Optimizer step",ylabel="Unscaled MSE (×10⁵)")
        ax.legend(frameon=False)
    save(fig,"03_data_learning_curves")

    feature=("XP-zero","XP-active-control","XPF")
    fig, axs=plt.subplots(1,2,figsize=(10,4.0),layout="constrained")
    x=np.arange(3)
    for offs,pol,color in ((-.17,"selected","#2c6ba3"),(.17,"final_raw","#a84f2c")):
        for i,(ax,k) in enumerate(zip(axs,("mse","correction_cosine"))):
            values=[float(scores[method,pol][k])*(1e5 if not i else 1) for method in feature]
            ax.bar(x+offs,values,.34,color=color,label=pol)
    for ax,k in zip(axs,("Test MSE (×10⁵)","Correction–truth cosine")):
        ax.set(xticks=x,xticklabels=feature,ylabel=k)
    axs[0].axhline(float(scores["Native","selected"]["mse"])*1e5,color="#333",ls="--",lw=1)
    axs[0].legend(frameon=False)
    save(fig,"04_feature_controls")

    modes=[m for m,p in scores if p=="final_raw"]
    fig, axs=plt.subplots(2,1,figsize=(10.5,7),layout="constrained",sharex=True)
    for ax,policy in zip(axs,("selected","final_raw")):
        x=np.arange(len(modes))
        for offset,key,label,color,mult in ((-.22,"C","2 × inner product","#2f679d",2),
                                           (0,"E","Correction energy","#ad5f24",1),
                                           (.22,"G","Net benefit","#357a67",1)):
            ax.bar(x+offset,[float(scores[m,policy][key])*mult*1e6 for m in modes],
                   .21,color=color,label=label)
        ax.axhline(0,color="#444",lw=.6)
        ax.set(title=policy,ylabel="Original scale (×10⁶)")
    axs[0].legend(frameon=False,ncol=3)
    axs[1].set(xticks=x,xticklabels=modes)
    axs[1].tick_params(axis="x",labelrotation=20)
    save(fig,"05_residual_decomposition")

    fig, axs=plt.subplots(1,2,figsize=(11,4),layout="constrained",sharey=True)
    palette=plt.cm.tab10(np.linspace(0,1,10))
    for ax,policy in zip(axs,("selected","final_raw")):
        for j,m in enumerate(["Native",*modes]):
            row=scores[m,"selected" if m=="Native" else policy]
            ax.plot(np.arange(1,21),[float(row[f"lead_{k:02d}"]) for k in range(1,21)],
                    color=palette[j%10],ls="--" if m=="Native" else "-",label=m)
        ax.set(title=policy,xlabel="Physical lead",xticks=(1,5,10,15,20))
    axs[0].set_ylabel("Original-scale MSE")
    axs[0].legend(frameon=False,ncol=2,fontsize=7)
    save(fig,"06_lead_mse")

    subset=[r for r in read("windows") if r["policy"]=="final_raw"]
    fig, ax=plt.subplots(figsize=(8,4.8),layout="constrained")
    for j,m in enumerate(modes):
        own=[r for r in subset if r["mode"]==m]
        ax.scatter([float(r["raw_cosine"]) for r in own],
                   [float(r["improvement_pct"]) for r in own],s=14,alpha=.6,
                   color=palette[j%10],label=m)
    ax.axhline(0,color="#333",lw=.7)
    ax.set(xlabel="Window correction–truth cosine",ylabel="Window MSE reduction vs Native (%)",
           title="Raw step-1000 corrections")
    ax.legend(frameon=False,ncol=2,fontsize=8)
    save(fig,"07_cosine_and_skill")

    config=json.loads((out/"config.json").read_text())
    with np.load(out/"plot_case.npz") as f:
        fields={k:f[k] for k in f.files}
    case=config["plot_case"]
    cols=3
    fig, axes=plt.subplots((len(fields)+cols-1)//cols,cols,figsize=(11,2.5*((len(fields)+2)//3)),
                           layout="constrained",sharex=True,sharey=True)
    xx,yy=np.meshgrid(np.arange(0,64,4),np.arange(0,32,3))
    t=case["lead"]-1
    peak=max(np.linalg.norm(f[:,t],axis=0).max() for f in fields.values())
    for ax,(name,z) in zip(axes.flat,fields.items()):
        u,v=z[:,t,yy,xx]
        norm=np.hypot(u,v)
        div=np.where(norm>0,norm,1)
        q=ax.quiver(xx,yy,u/div,v/div,norm,cmap="viridis",norm=PowerNorm(.5,vmin=0,vmax=peak),
                    scale=.5,angles="xy",scale_units="xy",width=.004,minlength=.1)
        ax.set(title="True residual" if name=="truth" else name,aspect="equal")
    for ax in list(axes.flat)[len(fields):]:
        ax.set_visible(False)
    fig.colorbar(q,ax=axes,shrink=.7,label="Original magnitude")
    fig.suptitle(f'{case["trajectory"]} · start {case["start"]} · lead {case["lead"]} · seed 42 endpoints')
    save(fig,"08_fixed_direction_fields")

    if any(k.startswith("Direction-") for k in fields):
        sample=fields["truth"]/config["common_raw_residual_rms"]
        eligible=np.linalg.norm(sample,axis=0)>.05
        fig,ax=plt.subplots(figsize=(7.6,4),layout="constrained")
        for name,field in fields.items():
            if name!="truth" and not name.startswith("Direction-"):
                continue
            norm=np.linalg.norm(field,axis=0)
            z=field/np.maximum(norm,.01*config["common_raw_residual_rms"])
            edge=np.concatenate([((z[:,:,:,:-1]*z[:,:,:,1:]).sum(0))[eligible[:,:,:-1]&eligible[:,:,1:]],
                                 ((z[:,:,:-1]*z[:,:,1:]).sum(0))[eligible[:,:-1]&eligible[:,1:]]])
            ax.hist(edge,bins=np.linspace(-1,1,41),density=True,histtype="step",label=name)
        ax.set(xlabel="Neighbor cosine on valid edges",ylabel="Density",
               title="Fixed development window · horizontal and vertical edges")
        ax.legend(frameon=False)
        save(fig,"09_direction_relations")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("out",type=Path)
    render(p.parse_args().out)
