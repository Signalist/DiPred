"""Render the frozen study's physical results and mechanism diagnostics."""
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from research.corrector_simple_baseline.measure import groups
from research.corrector_window_study.data import read_json

from .validation import read_rows


def save(fig,path):
    fig.savefig(path,dpi=155,bbox_inches="tight")
    plt.close(fig)


def decorate(ax,title,ylabel=None):
    ax.set_title(title,loc="left",fontsize=11)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(axis="y",alpha=.21)
    ax.spines[["top","right"]].set_visible(False)


def legend(ax):
    ax.legend(frameon=False,fontsize=8)


def heat(ax,rows,metric,title,center=False):
    matrix=np.zeros((20,20))
    for r in rows:
        matrix[int(r["lead"])-1,int(r["source"])-1]=float(r[metric])
    scale=np.abs(matrix).max()+1e-15
    options=dict(cmap="RdBu_r",vmin=-scale,vmax=scale) if center else dict(cmap="viridis")
    image=ax.imshow(matrix,origin="lower",extent=(.5,20.5,.5,20.5),**options)
    ax.set_title(title,fontsize=10)
    ax.set_xticks((1,10,20))
    ax.set_yticks((1,10,20))
    return image


def plot(out,figures):
    figures.mkdir(parents=True,exist_ok=True)
    select=read_json(out/"FROZEN.json")["selection"]
    best,chosen=select["best_new"],select["best_MSE_input"]
    policy_rows=read_rows(out/"scores.csv")
    scores={(r["split"],r["policy"],r["arm"]):r for r in policy_rows}
    leads=groups(read_rows(out/"leads.csv"),("split","policy","arm"))
    contrasts=read_rows(out/"contrasts.csv")

    picked=[r for r in contrasts if r["comparison"] in ("P1","P2","P3","P4")
            and r["split"] in ("development","historical_test") and r["policy"]=="selected"]
    fig,axes=plt.subplots(1,2,figsize=(13,6),layout="constrained")
    for ax,split in zip(axes,("development","historical_test")):
        rows=[r for r in picked if r["split"]==split]
        estimates=np.array([float(r["effect_pct"]) for r in rows])
        ends=np.array([[float(r[k]) for k in ("lower_pct","upper_pct")] for r in rows]).T
        positions=np.arange(len(rows))
        ax.hlines(positions,*ends,color="#597489")
        ax.scatter(estimates,positions,color="#244c66",s=22,zorder=3)
        ax.set_yticks(positions,[f"{r['comparison']}  {r['arm']} / {r['control']}" for r in rows],fontsize=8)
        ax.axvline(0,color=".5",lw=.8)
        ax.invert_yaxis()
        ax.set_xlabel("Paired MSE improvement (%)")
        decorate(ax,split)
    save(fig,figures/"01_primary_comparisons.png")

    lead_arms=list(dict.fromkeys(("S3","I0","IB","IE",best,chosen)))
    for file,key,title in (("02_native_by_lead.png","native_pct","Relative to Native"),
                           ("03_s3_by_lead.png","s3_pct","Relative to frozen S3")):
        fig,axes=plt.subplots(1,2,figsize=(11,4.3),layout="constrained",sharey=True)
        for ax,split in zip(axes,("development","historical_test")):
            for arm in lead_arms:
                rows=leads[split,"selected",arm]
                ax.plot([int(r["lead"]) for r in rows],[float(r[key]) for r in rows],marker=".",label=arm)
            ax.axhline(0,color=".45",lw=.8)
            ax.set_xticks((1,5,10,15,20))
            ax.set_xlabel("Forecast lead")
            decorate(ax,split,title+" (%)")
        legend(axes[0])
        save(fig,figures/file)

    gradients=groups(read_rows(out/"matched_gradients.csv"),("model","step"))
    fig,axes=plt.subplots(2,3,figsize=(12,7),layout="constrained")
    for row,rho in zip(axes,("0.1","0.3")):
        for ax,metric,title in zip(row,("aux_to_mse","cosine","dev_mse"),
                                   ("Auxiliary / MSE gradient norm","Gradient cosine","Development physical MSE")):
            for loss in ("match2","diag","perm"):
                steps=(1000,2000,4000,6000)
                ax.plot(steps,[mean(float(r[metric]) for r in gradients[f"Ecal_{loss}_{rho}",str(s)])
                               for s in steps],marker="o",label=loss)
            ax.set_xlabel("Update")
            decorate(ax,f"{title}   ρ={rho}")
        legend(row[0])
    save(fig,figures/"04_matched_gradient_controls.png")

    training=groups(read_rows(out/"training.csv"),("arm",))
    fig,axes=plt.subplots(1,3,figsize=(13,4.2),layout="constrained",sharey=True)
    for ax,suffix in zip(axes,("0","B","E")):
        for family,color in (("I","#28615d"),("D","#8c4f26")):
            arm=family+suffix
            rows=training[arm,]
            steps=[int(r["step"]) for r in rows]
            for metric,style in (("train_mse","--"),("dev_mse","-")):
                ax.plot(steps,[float(r[metric]) for r in rows],style,color=color,label=f"{arm} {metric.split('_')[0]}")
        ax.set_xlabel("Update")
        decorate(ax,f"I{suffix} and D{suffix}","Physical MSE")
        legend(ax)
    save(fig,figures/"05_direct_incremental_learning.png")

    terms=("C_delta","U_delta","G_delta")
    fig,axes=plt.subplots(1,2,figsize=(11,4.6),layout="constrained",sharey=True)
    names=list(dict.fromkeys(("I0","IB","IE","D"+chosen[1:],best)))
    for ax,split in zip(axes,("development","historical_test")):
        p=np.arange(len(names))
        for j,key in enumerate(terms):
            ax.bar(p+(j-1)*.25,[float(scores[split,"selected",a][key]) for a in names],width=.23,label=key)
        ax.axhline(0,color=".5",lw=.8)
        ax.set_xticks(p,names,rotation=27)
        decorate(ax,split,"Physical residual energy")
    legend(axes[0])
    save(fig,figures/"06_incremental_gain.png")

    t=groups(read_rows(out/"temporal_mode_scores.csv"),("split","policy","arm"))
    d=groups(read_rows(out/"incremental_mode_scores.csv"),("split","policy","arm"))
    fig,axes=plt.subplots(2,2,figsize=(11,7.5),layout="constrained",sharex=True)
    for row,anchor in zip(axes,(chosen,"D"+chosen[1:])):
        for ax,book in zip(row,(t,d)):
            for name in (anchor,f"{anchor}_K4",f"{anchor}_K8"):
                data=book["development","final",name]
                data=groups(data,("mode",))
                modes=sorted(data,key=lambda x:int(x[0]))
                a,b=("predicted","squared_error") if book is t else ("delta_energy","final_error_energy")
                for key,style in ((a,"-"),(b,"--")):
                    energies=[mean(float(r[key]) for r in data[k]) for k in modes]
                    ax.semilogy(range(1,21),np.maximum(energies,1e-20),style,label=f"{name} {key}")
            decorate(ax,f"{anchor}  "+("total" if book is t else "increment"),"Physical energy")
            legend(ax)
    for ax in axes[-1]:
        ax.set_xlabel("Second-difference mode")
    save(fig,figures/"07_temporal_modes.png")

    maps=groups(read_rows(out/"time_maps.csv"),("policy","model"))
    impact=groups(read_rows(out/"history_interventions.csv"),("policy","model"))
    fig,axes=plt.subplots(2,6,figsize=(16,6.3),layout="constrained")
    for i,name in enumerate(("IB","IB_last","IB_reverse","IB_lag","IB_dual","IB_free")):
        im=heat(axes[0,i],maps["selected",name],"value",name,center=True)
        fig.colorbar(im,ax=axes[0,i],shrink=.57)
        im=heat(axes[1,i],impact["selected",name],"response_rms","Source intervention")
        fig.colorbar(im,ax=axes[1,i],shrink=.57)
        axes[1,i].set_xlabel("Historical source frame")
    axes[0,0].set_ylabel("Current lead")
    axes[1,0].set_ylabel("Current lead")
    save(fig,figures/"08_time_maps_and_interventions.png")

    tracks=groups(read_rows(out/"trajectories.csv"),("split","policy","arm"))
    regimes=groups(read_rows(out/"regimes.csv"),("split","policy","arm"))
    channels=groups(read_rows(out/"channels.csv"),("split","policy","arm"))
    fig,axes=plt.subplots(3,1,figsize=(12,9),layout="constrained")
    track=sorted(tracks["historical_test","selected",best],key=lambda x:float(x["re"]))
    axes[0].bar(np.arange(len(track)),[float(r["s3_pct"]) for r in track],color="#55768d")
    axes[0].set_xticks(np.arange(len(track)),[r["trajectory"][:-3] for r in track],rotation=35)
    decorate(axes[0],"Historical trajectories  "+best,"S3 improvement (%)")
    a=sorted(regimes["historical_test","selected",best],key=lambda x:float(x["re"]))
    axes[1].bar(np.arange(len(a)),[float(r["s3_pct"]) for r in a],color="#47726f")
    axes[1].set_xticks(np.arange(len(a)),[r["re"] for r in a])
    decorate(axes[1],"Historical Reynolds numbers","S3 improvement (%)")
    group=("I0","IB","IE",best)
    group=list(dict.fromkeys(group))
    for i,c in enumerate(("u","v")):
        value=[float(next(r["s3_pct"] for r in channels["historical_test","selected",name] if r["channel"]==c))
               for name in group]
        axes[2].bar(np.arange(len(group))+.3*i,value,width=.28,label=c)
    axes[2].set_xticks(np.arange(len(group))+.15,group)
    decorate(axes[2],"Historical physical channels","S3 improvement (%)")
    for ax in axes:
        ax.axhline(0,lw=.8,color=".45")
    legend(axes[2])
    save(fig,figures/"09_regimes_tracks_channels.png")

    feedback=groups(read_rows(out/"feedback_rollout.csv"),("arm","run_mode"))
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout="constrained",sharey=True)
    for ax,tag in zip(axes,("MSE","aux")):
        for train in ("ref","self"):
            for mode in ("ref","self"):
                rows=feedback[f"Q_{tag}_{train}",mode]
                by_start=groups(rows,("start",))
                starts=sorted(by_start,key=lambda k:int(k[0]))
                metric=[mean(float(r["mse"]) for r in by_start[s]) for s in starts]
                ax.plot(range(1,len(starts)+1),metric,marker="o",label=f"train {train} / run {mode}")
        decorate(ax,tag,"Physical MSE")
        ax.set_xlabel("Consecutive release")
        legend(ax)
    save(fig,figures/"10_feedback_train_run.png")

    costs={r["arm"]:r for r in read_rows(out/"inference_cost.csv")}
    arms=list(dict.fromkeys(("Native","S3","I0","IB","IE","D"+chosen[1:],best)))
    fig,axes=plt.subplots(1,2,figsize=(11.5,4.8),layout="constrained")
    p=np.arange(len(arms))
    bottom=np.zeros(len(arms))
    for label,key in (("Frozen DiPred","backbone_parameters"),("Frozen S3","frozen_s3_parameters"),
                      ("Trainable head","trainable_parameters")):
        number=np.array([float(costs[a][key]) for a in arms])
        axes[0].bar(p,number,bottom=bottom,label=label)
        bottom+=number
    axes[1].bar(p,[float(costs[a]["ms_end_to_end"]) for a in arms],color="#4b7a78")
    for ax,title,label in zip(axes,("Model size","Batch-one current release"),("Parameters","Median milliseconds")):
        ax.set_xticks(p,arms,rotation=32)
        decorate(ax,title,label)
    legend(axes[0])
    save(fig,figures/"11_inference_cost.png")

    fields=groups(read_rows(out/"fixed_fields.csv"),("arm","channel"))
    fig,axes=plt.subplots(2,5,figsize=(14,5.8),layout="constrained")
    specs=(("truth_residual","value","True residual"),("S3","value","S3 correction"),
           (best,"value",best+" correction"),("S3","remaining","S3 remaining"),(best,"remaining",best+" remaining"))
    for row,channel in zip(axes,("u","v")):
        grids=[]
        for name,key,title in specs:
            pixels=fields[name,channel]
            shape=(1+max(int(r["y"]) for r in pixels),1+max(int(r["x"]) for r in pixels))
            grid=np.empty(shape)
            for r in pixels:
                grid[int(r["y"]),int(r["x"])]=float(r[key])
            grids.append(grid)
        cap=max(np.abs(v).max() for v in grids)
        for ax,grid,(_,_,title) in zip(row,grids,specs):
            im=ax.imshow(grid,origin="lower",vmin=-cap,vmax=cap,cmap="RdBu_r")
            ax.set_title(title,fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
        row[0].set_ylabel(channel)
        fig.colorbar(im,ax=row,shrink=.7)
    save(fig,figures/"12_fixed_development_fields.png")

    fig,axes=plt.subplots(1,3,figsize=(13,4.3),layout="constrained",sharey=True)
    for ax,sub in zip(axes,("interpolation","upper","lower")):
        split="holdout_"+sub
        arms=select["outer"]
        values=[float(scores[split,"final",a]["s3_pct"]) for a in arms]
        ax.bar(np.arange(len(arms)),values,color="#51768c")
        ax.axhline(0,color=".45",lw=.8)
        ax.set_xticks(np.arange(len(arms)),arms,rotation=38)
        decorate(ax,sub,"S3 improvement (%)")
    save(fig,figures/"13_holdout_stress_tests.png")
