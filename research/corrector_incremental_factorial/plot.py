"""Matplotlib outputs from audited physical scores and diagnostics."""
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from research.corrector_incremental_structure.validation import read_rows
from research.corrector_simple_baseline.measure import groups

from .core import BASE, FACTORIAL

DEV, HIST = "development", "historical_test"
REGIONS = ("holdout_interpolation", "holdout_upper", "holdout_lower")
RES = (17812, 25482, 8863)


def bars(ax, labels, values, ylabel):
    colors = ["#287d8e" if v >= 0 else "#bf6e4a" for v in values]
    ax.bar(range(len(values)), values, color=colors)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=24, ha="right")
    ax.set_ylabel(ylabel)
    ax.axhline(0, color=".65", lw=.6)


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot(out, figures, freeze):
    figures.mkdir(parents=True, exist_ok=True)
    scores = {(r["split"], r["policy"], r["arm"]): r for r in read_rows(out/"scores.csv")}
    main = freeze["selection"]["best_new"]

    def fixed(split, arm, column):
        return float(scores[split,"final",arm][column])

    fig, axs = plt.subplots(1,2,figsize=(12,4))
    for ax, label in zip(axs,(DEV,HIST)):
        bars(ax, FACTORIAL, [fixed(label,n,"native_pct") for n in FACTORIAL], "Residual correction versus Native (%)")
        ax.set_title(label.replace("_"," ").title())
    save(fig, figures/"01_factorial_overview.png")

    leads = groups(read_rows(out/"leads.csv"), ("split","policy","arm"))
    for k, metric, caption in ((2,"native_pct","Correction versus Native"),
                               (3,"s3_pct","Additional correction versus S3")):
        fig, axs = plt.subplots(1,2,figsize=(13,4))
        for ax, split in zip(axs,(DEV,HIST)):
            for name in FACTORIAL:
                rows = sorted(leads[split,"final",name],key=lambda r:int(r["lead"]))
                ax.plot(range(1,21),[float(r[metric]) for r in rows],label=name,marker=".",ms=3)
            ax.axhline(0,color=".65",lw=.6)
            ax.set_xticks(range(1,21))
            ax.set_xlabel("Lead")
            ax.set_ylabel(caption+" (%)")
            ax.set_title(split.replace("_"," ").title())
        axs[1].legend(fontsize=7,ncol=2)
        save(fig,figures/f"{k:02d}_{metric}_by_lead.png")

    fig, axs = plt.subplots(1,2,figsize=(9,4))
    for ax, label in zip(axs,(DEV,HIST)):
        bars(ax,BASE,[fixed(label,n+"_match2","own_pct") for n in BASE],"MSE reduction versus own MSE arm (%)")
        ax.set_title(label.replace("_"," ").title())
    save(fig,figures/"04_matched_own_loss.png")

    for idx, axis in ((5,"history"),(6,"parameterization")):
        fig, axs = plt.subplots(1,2,figsize=(10,4),sharey=False)
        for ax, split in zip(axs,(DEV,HIST)):
            if axis=="history":
                controls=[(s, s+"0", s+"B") for s in ("D","I")]
                xlabels=("No history","True residual history")
            else:
                controls=[(f"History {s}","D"+s,"I"+s) for s in ("0","B")]
                xlabels=("Direct","Incremental")
            for label,a,b in controls:
                for loss, style in (("","--"),("_match2","-")):
                    ax.plot((0,1),[fixed(split,n+loss,"mse") for n in (a,b)],style,marker="o",
                            label=f"{label}, {'TM2' if loss else 'MSE'}")
            ax.set_xticks((0,1))
            ax.set_xticklabels(xlabels)
            ax.set_title(split.replace("_"," ").title())
            ax.set_ylabel("Physical MSE")
        axs[1].legend(fontsize=7)
        save(fig,figures/f"{idx:02d}_{axis}_interactions.png")

    effects=read_rows(out/"factorial_effects.csv")+read_rows(out/"factorial_interactions.csv")
    fig, axs=plt.subplots(1,2,figsize=(12,4),sharey=True)
    for ax,split in zip(axs,(DEV,HIST)):
        rows=[r for r in effects if r["split"]==split and r["policy"]=="final" and r["kind"]=="factorial"]
        values=np.array([float(r["mse_reduction"]) for r in rows])
        bounds=np.array([[float(r["lower"]),float(r["upper"])] for r in rows])
        pos=np.arange(len(rows))
        ax.hlines(pos,bounds[:,0],bounds[:,1],color="#276789")
        ax.scatter(values,pos,color="#276789")
        ax.axvline(0,color=".55",lw=.7)
        ax.set_yticks(pos)
        ax.set_yticklabels([r["effect"] for r in rows])
        ax.set_xlabel("Physical MSE reduction contrast")
        ax.set_title(split.replace("_"," ").title())
    save(fig,figures/"07_factorial_intervals.png")

    fig,ax=plt.subplots(figsize=(10,3.3))
    matrix=np.array([[fixed(s,a,"s3_pct") for a in FACTORIAL] for s in REGIONS])
    bound=max(abs(matrix).max(),.01)
    im=ax.imshow(matrix,cmap="RdBu",vmin=-bound,vmax=bound,aspect="auto")
    for i,j in np.ndindex(matrix.shape):
        ax.text(j,i,f"{matrix[i,j]:+.2f}",ha="center",va="center",fontsize=8,
                color="white" if abs(matrix[i,j])>.7*bound else "black")
    ax.set_xticks(range(len(FACTORIAL)))
    ax.set_xticklabels(FACTORIAL,rotation=25,ha="right")
    ax.set_yticks(range(3))
    ax.set_yticklabels([f"Re {r}" for r in RES])
    fig.colorbar(im,ax=ax,label="Additional reduction versus isolated S3 (%)")
    save(fig,figures/"08_heldout_factorial.png")

    fig,axs=plt.subplots(1,2,figsize=(12,4))
    for ax,split in zip(axs,(DEV,HIST)):
        j=np.arange(len(FACTORIAL))
        for i,(col,k,label) in enumerate((("C_delta",2,"2C delta"),("U_delta",-1,"−U delta"),("G_delta",1,"G delta"))):
            ax.bar(j+(i-1)*.23,[k*fixed(split,n,col) for n in FACTORIAL],width=.22,label=label)
        ax.set_xticks(j)
        ax.set_xticklabels(FACTORIAL,rotation=25,ha="right")
        ax.axhline(0,color=".65",lw=.6)
        ax.set_ylabel("Physical squared units")
        ax.set_title(split.replace("_"," ").title())
    axs[1].legend(fontsize=7)
    save(fig,figures/"09_incremental_decomposition.png")

    rows=read_rows(out/"history_interventions.csv")
    fig,axs=plt.subplots(2,2,figsize=(11,7))
    for name,axs_row in zip(("IB","IB_match2"),axs):
        selected=[r for r in rows if r["split"]==DEV and r["arm"]==name]
        by=groups(selected,("intervention",))
        original=mean(float(r["mse"]) for r in by["original",])
        experiments=[k[0] for k in by if k[0]!="original" and not k[0].startswith("zero_source")]
        bars(axs_row[0],experiments,[100*(1-mean(float(r["mse"]) for r in by[k,])/original)
                                     for k in experiments],"MSE improvement versus intact input (%)")
        axs_row[0].set_title(name)
        response=np.zeros((20,20))
        for i in range(20):
            for row in by[f"zero_source_{i+1}",]:
                response[i,int(row["lead"])-1]=float(row["response_rms"])
        im=axs_row[1].imshow(response,origin="lower",aspect="auto",cmap="viridis",extent=(.5,20.5,.5,20.5))
        axs_row[1].set_xlabel("Output lead")
        axs_row[1].set_ylabel("History frame ablated")
        fig.colorbar(im,ax=axs_row[1],label="Output change RMS")
    save(fig,figures/"10_history_intervention.png")

    grad=read_rows(out/"gradient_diagnostics.csv")
    grouped=groups([r for r in grad if r["term"]!="mse" and not r["arm"].endswith("_probe")],("arm",))
    fig,axs=plt.subplots(1,3,figsize=(14,4))
    for (name,),row in grouped.items():
        points=groups(row,("step",))
        steps=sorted(int(k[0]) for k in points)
        for ax,col in zip(axs,("weighted_ratio","cosine_to_mse","clip_factor")):
            ax.plot(steps,[mean(float(r[col]) for r in points[str(step),]) for step in steps],label=name,marker=".")
            ax.set_xlabel("Update")
            ax.set_ylabel(col.replace("_"," "))
    axs[2].legend(fontsize=6,ncol=2)
    save(fig,figures/"11_gradients_over_training.png")

    names=[n for n in freeze["models"] if n not in FACTORIAL]
    fig,axs=plt.subplots(1,2,figsize=(12,4))
    for ax,split in zip(axs,(DEV,HIST)):
        bars(ax,names,[fixed(split,n,"s3_pct") for n in names],"Additional MSE reduction (%)")
        ax.set_title(split.replace("_"," ").title())
    save(fig,figures/"12_calibrated_and_extension_results.png")

    fig,axs=plt.subplots(3,1,figsize=(10,8))
    for ax,filename,key in zip(axs,("regimes","trajectories","channels"),("re","trajectory","channel")):
        rows=[r for r in read_rows(out/f"{filename}.csv") if r["split"]==HIST and r["policy"]=="selected" and r["arm"]==main]
        bars(ax,[r[key].removesuffix(".h5") if hasattr(r[key],"removesuffix") else r[key].replace(".h5","") for r in rows],
             [float(r["s3_pct"]) for r in rows],"Reduction versus S3 (%)")
    save(fig,figures/"13_regime_trajectory_channel.png")

    patches=groups(read_rows(out/"fixed_fields.csv"),("channel","field"))
    names=("truth_residual","s3_correction",main+"_increment",main+"_remaining")
    fig,axs=plt.subplots(2,4,figsize=(13,4),layout="constrained")
    for c,row in zip(("u","v"),axs):
        bound=max(abs(float(r["value"])) for name in names for r in patches[c,name])
        for ax,name in zip(row,names):
            matrix=np.zeros((32,64))
            for z in patches[c,name]:
                matrix[int(z["y"]),int(z["x"])]=float(z["value"])
            im=ax.imshow(matrix,cmap="RdBu_r",vmin=-bound,vmax=bound)
            ax.set_title((c+" "+name).replace("_"," "),fontsize=8)
            ax.set_axis_off()
        fig.colorbar(im,ax=list(row),shrink=.75)
    fig.savefig(figures/"14_fixed_development_fields.png",dpi=170,bbox_inches="tight")
    plt.close(fig)

    cost={r["arm"]:r for r in read_rows(out/"inference_cost.csv")}
    fig,axs=plt.subplots(1,2,figsize=(12,4))
    for ax,xlabel,xkey in zip(axs,("End-to-end median latency (ms)","Trainable parameters"),
                             ("ms_end_to_end","trainable_parameters")):
        for name in ("Native","S3",*FACTORIAL):
            x=float(cost[name][xkey])
            y=float(scores[HIST,"selected",name]["native_pct"])
            ax.scatter(x,y,label=name)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Historical correction versus Native (%)")
    axs[1].legend(fontsize=7,ncol=2)
    save(fig,figures/"15_accuracy_and_end_to_end_cost.png")
