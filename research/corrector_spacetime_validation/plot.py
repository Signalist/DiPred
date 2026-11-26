"""Matplotlib figures backed by the published physical tables."""
import csv
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from research.corrector_simple_baseline.measure import groups
from research.corrector_window_study.data import read_json

PALETTE = ("#283d55","#16786e","#c46b3e","#824f77","#506a2e","#3a73a8","#a23f47")
FIXED = ("M0","EN","TS1","TS2","TM1","TM2")


def read(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def save(fig,folder,name):
    fig.savefig(folder/f"{name}.png",dpi=145,bbox_inches="tight",facecolor="white")
    plt.close(fig)


def bars(ax,labels,values,units):
    x = np.arange(len(labels))
    cols = list(values.items())
    width = .8/len(cols)
    for i,(label,series) in enumerate(cols):
        ax.bar(x+(i-(len(cols)-1)/2)*width,series,width,label=label,color=PALETTE[i])
    ax.axhline(0,color=".6",lw=.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels,rotation=25,ha="right")
    ax.set_ylabel(units)
    if len(cols)>1:
        ax.legend(frameon=False,fontsize=8)


def plot(out,folder):
    folder.mkdir(parents=True,exist_ok=True)
    selection = read_json(out/"selections.json")
    choices = selection["choices"]
    models = read_json(out/"FROZEN.json")["models"]
    scores = {(r["split"],r["policy"],r["arm"]):r for r in read(out/"scores.csv")}
    leads = {(r["split"],r["policy"],r["arm"],int(r["lead"])):r for r in read(out/"leads.csv")}
    contrasts = read(out/"contrasts.csv")

    def metric(split,arm,k,policy="final"):
        return float(scores[split,policy,arm][k])

    def effect(split,arm,base="S3",policy="final"):
        return 100*(1-metric(split,arm,"mse",policy)/metric(split,base,"mse",policy))

    def effect_ci(ax,split,arms,controls,policy,labels):
        vals = []
        for a,b in zip(arms,controls):
            if a==b:
                vals.append((0.,0.,0.))
                continue
            row = next(r for r in contrasts if (r["split"],r["policy"],r["arm"],r["control"])==(split,policy,a,b))
            vals.append(tuple(float(row[k]) for k in ("effect_pct","lower_pct","upper_pct")))
        x = np.arange(len(arms))
        for i,(v,lo,hi) in enumerate(vals):
            ax.scatter(i,v,color=PALETTE[0])
            ax.plot([i,i],[lo,hi],color=PALETTE[0])
        ax.axhline(0,color=".6",lw=.7)
        ax.set_xticks(x)
        ax.set_xticklabels(labels,rotation=27,ha="right")
        ax.set_ylabel("Relative physical MSE improvement (%)")

    fig,axs = plt.subplots(2,2,figsize=(12,7),sharey=True,constrained_layout=True)
    for row,b in zip(axs,("B","E")):
        for ax,split in zip(row,("development","historical_test")):
            arms = [choices[b][k] for k in FIXED]
            effect_ci(ax,split,arms,[choices[b]["M0"]]*len(arms),"selected",FIXED)
            ax.set_title(f"{b} · {split.replace('_',' ')}")
    save(fig,folder,"01_mechanism_matrix")

    fig,axs = plt.subplots(2,2,figsize=(13,7),constrained_layout=True)
    x = list(range(1,21))
    for row,split in zip(axs,("development","historical_test")):
        lines = [("S3","S3"),("B M0","B_M0"),("B TM2",choices["B"]["TM2"]),
                 ("E M0","E_M0"),("E EN",choices["E"]["EN"]),("E TM1",choices["E"]["TM1"]),("E TM2",choices["E"]["TM2"])]
        for j,key in enumerate(("native_pct","own_pct")):
            ax = row[j]
            for i,(caption,arm) in enumerate(lines):
                if arm=="S3" and j:
                    continue
                ax.plot(x,[float(leads[split,"selected",arm,t][key]) for t in x],".-",label=caption,color=PALETTE[i])
            ax.set_xticks(x)
            ax.axhline(0,color=".6",lw=.7)
            ax.set_title(f"{split.replace('_',' ')} · {'Native' if not j else 'own M0'}")
            ax.set_xlabel("Prediction lead")
            ax.set_ylabel("Improvement (%)")
            ax.legend(frameon=False,ncol=2,fontsize=8)
    save(fig,folder,"02_all_twenty_leads")

    fig,axs = plt.subplots(2,2,figsize=(12,7),constrained_layout=True)
    energy = read(out/"energy_matching.csv")
    for row,b in zip(axs,("B","E")):
        e = [r for r in energy if r["branch"]==b and r["policy"]=="final"]
        names = [r["model"] for r in e]
        names.append(e[0]["target"])
        vals = [float(r["train_U"]) for r in e]+[float(e[0]["target_U"])]
        bars(row[0],[n[len(b)+1:] for n in names],{"Train U":[v*1e6 for v in vals]},"Correction energy (×10⁶)")
        row[0].set_title(b)
        bars(row[1],[n[len(b)+1:] for n in names],{
            "2C":[2e6*metric("development",a,"C") for a in names],
            "U":[1e6*metric("development",a,"correction_energy") for a in names],
            "G":[1e6*metric("development",a,"G") for a in names]},"Development energy (×10⁶)")
    save(fig,folder,"03_train_energy_matching")

    fig,axs = plt.subplots(1,2,figsize=(11,4.2),constrained_layout=True,sharey=True)
    for ax,split in zip(axs,("development","historical_test")):
        ids = ("historical_E_ext_match2","E_DIAG","E_PERM")
        bars(ax,("Real TM2","Diagonal","Permuted"),{
            "Selected":[effect(split,a,"E_M0","selected") for a in ids],
            "Step 6000":[effect(split,a,"E_M0") for a in ids]},"Improvement vs E M0 (%)")
        ax.set_title(split.replace("_"," "))
    save(fig,folder,"04_diagonal_and_permuted")

    fig,axs = plt.subplots(1,2,figsize=(11,4.2),constrained_layout=True,sharey=True)
    for ax,split in zip(axs,("development","historical_test")):
        keys = ("EN","TM1","TM2")
        bars(ax,keys,{b:[(metric(split,choices[b]["M0"],"mse")-metric(split,f"{b}_{k}_0.3","mse"))*1e6 for k in keys]
                      for b in ("S3","B","E")},"Physical MSE reduction (×10⁶)")
        ax.set_title(split.replace("_"," ")+" · rho 0.3")
    save(fig,folder,"05_cross_branch_absolute_gain")

    fig,axs = plt.subplots(1,2,figsize=(12,4.2),constrained_layout=True,sharey=True)
    for ax,split in zip(axs,("development","historical_test")):
        bars(ax,("Original joint","J4","J5"),{b:[effect(split,f"{b}_{x}",f"{b}_M0")
                                               for x in ("JOINT_original","J4","J5")] for b in ("B","E")},
             "Improvement vs own M0 (%)")
        ax.set_title(split.replace("_"," "))
    save(fig,folder,"06_joint_attribution")

    fig,axs = plt.subplots(2,3,figsize=(12,7),constrained_layout=True)
    grad,clip = read(out/"gradient_diagnostics.csv"),read(out/"gradient_clipping.csv")
    for row,b in zip(axs,("B","E")):
        for j,k in enumerate(("EN","TS2","TM1","TM2")):
            arm = choices[b][k]
            found = models[arm]["name"]
            subset = [r for r in grad if r["model"]==found and r["term"]!="mse" and r["other"]=="mse"]
            by_step = groups(subset,("step",))
            xs = sorted(int(s[0]) for s in by_step)
            for ax,name in zip(row,("weighted_norm","cosine","clip_factor")):
                vals = ([ [float(r[name]) for r in by_step[str(s),]] for s in xs ] if name!="clip_factor" else
                        [[float(r[name]) for r in clip if r["model"]==found and int(r["step"])==s] for s in xs])
                data = np.array(vals)
                ax.plot(xs,data.mean(1),"o-",label=k,color=PALETTE[j])
                ax.fill_between(xs,data.min(1),data.max(1),color=PALETTE[j],alpha=.12)
                ax.set_xticks(xs)
                ax.set_title(f"{b} · {name.replace('_',' ')}")
                ax.legend(frameon=False,fontsize=8)
    save(fig,folder,"07_gradient_training_dynamics")

    fig,axs = plt.subplots(2,2,figsize=(12,7),constrained_layout=True)
    modes = read(out/"temporal_mode_scores.csv")
    keys = {(r["split"],r["policy"],r["arm"],r["channel"],int(r["mode"])):r for r in modes}
    for row,split in zip(axs,("development","historical_test")):
        selected = [("E M0","E_M0"),("E EN",choices["E"]["EN_train_U_final"]),("E TM2",choices["E"]["TM2"])]
        for ax,k in zip(row,("predicted","squared_error")):
            for i,(name,model) in enumerate(selected):
                vals = [mean(float(keys[split,"final",model,c,i][k])*1e6 for c in ("u","v")) for i in range(20)]
                ax.plot(range(20),vals,".-",label=name,color=PALETTE[i])
            ax.axvline(5.5,c=".6",lw=.8)
            ax.axvline(12.5,c=".6",lw=.8)
            ax.set_title(f"{split.replace('_',' ')} · {k.replace('_',' ')}")
            ax.set_xlabel("Finite-difference mode")
            ax.set_ylabel("Mean energy (×10⁶)")
            ax.set_yscale("log")
            ax.legend(frameon=False)
    save(fig,folder,"08_finite_difference_modes")

    fig,axs = plt.subplots(2,3,figsize=(15,7),constrained_layout=True)
    for row,split in zip(axs,("development","historical_test")):
        for ax,filename,key in zip(row,("regimes","trajectories","channels"),("re","trajectory","channel")):
            table = read(out/f"{filename}.csv")
            lookup = {(r["arm"],r[key]):float(r["own_pct"]) for r in table if r["split"]==split and
                      r["policy"]=="selected" and r["own_pct"]}
            labels = sorted({sub for arm,sub in lookup})
            bars(ax,labels,{b:[lookup[choices[b]["TM2"],f] for f in labels] for b in ("B","E")},"Improvement vs own M0 (%)")
            ax.set_title(f"{split.replace('_',' ')} · {key}")
            ax.tick_params(axis="x",labelsize=7)
    save(fig,folder,"09_regime_trajectory_channel")

    fig,axs = plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
    sens = read(out/"selection_sensitivity.csv")
    for ax,b in zip(axs,("B","E")):
        picks = [r for r in sens if r["branch"]==b and r["scope"]=="expanded" and r["policy"]=="selected"]
        series = {}
        for rule in ("A","B","C"):
            rows = [r for r in picks if r["rule"]==rule]
            series[rule+" · "+selection["rules"][b]["selected"][rule]] = [mean(float(r["held_vs_plain"]) for r in rows
                                                                           if r["kind"]==k) for k in ("trajectory","regime")]
        bars(ax,("Leave one trajectory","Leave one Re"),series,"Held-out improvement vs own M0 (%)")
    save(fig,folder,"10_selection_sensitivity")

    fig,axs = plt.subplots(2,2,figsize=(12,7),constrained_layout=True)
    hold = ("holdout_interpolation","holdout_upper","holdout_lower")
    split_plan = read_json(out/"outer_split_manifest.json")
    for ax,split in zip(axs.flat,(*hold,"period_new")):
        if split.startswith("holdout"):
            arms = ["B_ACC","B_MAG","B_TM2","E_EN","E_TM1","E_TM2"]
        else:
            arms = [choices[b][k] for b,k in (("B","TM2"),("E","EN"),("E","TM1"),("E","TM2"))]
        bars(ax,arms,{"Native":[metric(split,a,"native_pct") for a in arms],
                      "Own M0":[metric(split,a,"own_pct") for a in arms]},"Improvement (%)")
        if split in hold:
            re = split_plan[split[len("holdout_"):]]["re"]
            ax.set_title(f"{split.replace('_',' ')} · Re {re:g}")
        else:
            ax.set_title("New periods on the same trajectories")
    save(fig,folder,"11_regime_holdout_and_new_period")

    fig,axs = plt.subplots(2,2,figsize=(11,7),constrained_layout=True,sharex=True)
    feedback = read(out/"feedback_rollout.csv")
    for ax,kind in zip(axs.flat,("M0","EN","TM1","TM2")):
        rows = [r for r in feedback if r["mode"]==kind]
        groups_now = groups(rows,("chain_index",))
        times = sorted(int(t[0]) for t in groups_now)
        for i,k in enumerate(("fixed_mse","self_mse")):
            values = [100*(1-mean(float(v[k]) for v in groups_now[str(t),])/
                           mean(float(v["native_mse"]) for v in groups_now[str(t),])) for t in times]
            ax.plot(times,values,"o-",color=PALETTE[i],label=k.removesuffix("_mse") if hasattr(k,"removesuffix") else k[:-4])
        ax.set_title("E · "+kind)
        ax.axhline(0,color=".6",lw=.7)
        ax.set_xticks(times)
        ax.set_ylabel("Improvement vs Native (%)")
        ax.set_xlabel("Consecutive release")
        ax.legend(frameon=False)
    save(fig,folder,"12_consecutive_feedback")

    field = read(out/"fixed_fields.csv")
    ids = ("true","S3",choices["B"]["TM2"],choices["E"]["EN_train_U_final"],choices["E"]["TM2"])
    fig,axs = plt.subplots(2,len(ids),figsize=(12,4),constrained_layout=True)
    limit = max(abs(float(r[k])) for r in field for k in ids)
    for row,c in zip(axs,("u","v")):
        points = [r for r in field if r["channel"]==c]
        for ax,k in zip(row,ids):
            im = ax.imshow(np.array([float(v[k]) for v in points]).reshape(32,64),origin="lower",
                           cmap="RdBu_r",vmin=-limit,vmax=limit)
            ax.set_title(f"{c} · {k}",fontsize=8)
            ax.set_axis_off()
    fig.colorbar(im,ax=axs.ravel().tolist(),label="Physical residual")
    fig.suptitle("Fixed development case · 26700_0.h5 · start 400 · lead 10")
    save(fig,folder,"13_fixed_physical_case")
