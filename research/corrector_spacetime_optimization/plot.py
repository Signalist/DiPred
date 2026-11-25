"""Render the fifteen prespecified figures from frozen score and search tables."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np

from research.corrector_window_study.data import read_json
from .data import BRANCHES
from .loss import NAMES

LABELS = dict(zip(NAMES.values(),NAMES))
COLORS = ("#425970","#d07436","#148474","#a74e66","#7063a2","#7c7a42","#4974a2","#984e37")
POLICY_LABELS = {"selected":"Dev-selected","final":"Step 6000"}


def read_rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def index(rows,keys):
    return {tuple(r[k] for k in keys):r for r in rows}


def bars(ax,labels,series,ylabel):
    x = np.arange(len(labels))
    width = .78/len(series)
    for i,(name,values) in enumerate(series):
        ax.bar(x+(i-(len(series)-1)/2)*width,values,width,label=name,color=COLORS[i])
    ax.set_xticks(x)
    ax.set_xticklabels(labels,rotation=24,ha="right")
    ax.set_ylabel(ylabel)
    ax.axhline(0,color=".45",linewidth=.7)
    if len(series)>1:
        ax.legend(frameon=False)


def lead_curves(ax,leads,arms,split,field):
    x = np.arange(1,21)
    for i,(name,arm) in enumerate(arms):
        y = [float(leads[split,"selected",arm,str(t)][field]) for t in x]
        ax.plot(x,y,label=name,color=COLORS[i%len(COLORS)],marker=".",linewidth=1.5)
    ax.set_xticks(x)
    ax.set_xlabel("Prediction lead")
    ax.axhline(0,color=".45",linewidth=.7)
    ax.grid(axis="y",alpha=.25)
    ax.legend(frameon=False,ncol=2,fontsize=8)


def plot(out,figures):
    out,figures = Path(out),Path(figures)
    figures.mkdir(parents=True,exist_ok=True)
    selected = read_json(out/"FROZEN.json")["selections"]
    scores = index(read_rows(out/"scores.csv"),("split","policy","arm"))
    leads = index(read_rows(out/"leads.csv"),("split","policy","arm","lead"))
    loss_search = read_rows(out/"loss_search.csv")
    diagnostics = read_rows(out/"gradient_diagnostics.csv")
    conflicts = read_rows(out/"gradient_conflicts.csv")

    def finish(fig,name):
        fig.savefig(figures/f"{name}.png",dpi=155,bbox_inches="tight",facecolor="white")
        plt.close(fig)

    def score(split,arm,field="native_pct",policy="selected"):
        return float(scores[split,policy,arm][field])

    final = [("Native","Native"),("S3","S3"),*((f"{b} plain",f"{b}_MSE") for b in BRANCHES),
             *((f"{b} optimized",selected["best"][b]) for b in BRANCHES)]
    if selected["best_extension"]:
        final.append(("Best extension",selected["best_extension"]))
    fig,ax = plt.subplots(figsize=(8,4),constrained_layout=True)
    bars(ax,[label for label,_ in final],[(label,[score(s,a) for _,a in final]) for label,s in (
        ("Development","development"),("Historical test","test"))],"Correction vs Native (%)")
    finish(fig,"01_overall")

    order = ("MSE",*NAMES,"JOINT")
    limits = [score(s,selected["fixed"][b][k],policy=policy)
              for b in BRANCHES for k in order for s in ("development","test") for policy in POLICY_LABELS]
    margin = max(.2,.06*(max(limits)-min(limits)))
    for i,b in enumerate(BRANCHES,2):
        fig,axs = plt.subplots(1,2,figsize=(11,4),sharey=True,constrained_layout=True)
        for ax,split in zip(axs,("development","test")):
            arms = [selected["fixed"][b][k] for k in order]
            bars(ax,order,[(label,[score(split,a,policy=p) for a in arms])
                          for p,label in POLICY_LABELS.items()],"Correction vs Native (%)")
            ax.set_ylim(min(0,min(limits))-margin,max(limits)+margin)
            ax.set_title(split.capitalize())
        finish(fig,f"{i:02d}_{b}_fixed")

    fig,axs = plt.subplots(2,5,figsize=(16,6),constrained_layout=True)
    for row,b in zip(axs,BRANCHES):
        for ax,loss in zip(row,LABELS):
            raw = [r for r in loss_search if r["branch"]==b and r["loss"]==loss
                   and r["stage"] in ("raw","raw_boundary")]
            pts = sorted((float(r["coordinate"]),float(r["gain_native_pct"])) for r in raw)
            ax.plot(*zip(*pts),"o-",label="Raw λ",color=COLORS[0])
            balanced = [r for r in loss_search if r["branch"]==b and r["loss"]==loss and r["stage"]=="balanced"]
            for r in balanced:
                ax.scatter(json.loads(r["weights"])[loss],float(r["gain_native_pct"]),marker="D",color=COLORS[1])
            ax.axhline(pts[0][1],color=".55",linewidth=.8)
            ax.set_xscale("symlog",linthresh=.001)
            ax.set_title(f"{b} · {LABELS[loss]}")
            ax.set_xlabel("λ (diamonds: gradient-calibrated)")
        row[0].set_ylabel("Development correction (%)")
    finish(fig,"04_weight_sensitivity")

    fig,axs = plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
    for row,b in zip(axs,BRANCHES):
        def avg(loss,key):
            v = [float(r[key]) for r in diagnostics if r["run"]==f"{b}_MSE" and r["step"]=="1000"
                 and r["loss"]==loss]
            return np.mean(v)
        names = ("mse",*LABELS)
        bars(row[0],[n.upper() if n=="mse" else LABELS[n] for n in names],
             [("Raw",[avg(n,"gradient_l2") for n in names])],"Gradient L2")
        row[0].set_title(f"{b} · Common update 1000")
        base = avg("mse","gradient_l2")
        bars(row[1],list(LABELS.values()),[(name,[100*avg(loss,k)/base for loss in LABELS]) for name,k in (
            ("λ = 0.1","fixed_0_1_gradient"),("ρ = 0.1","rho_0_1_norm"),("ρ = 0.3","rho_0_3_norm"))],
            "Auxiliary gradient / MSE gradient (%)")
        row[1].set_yscale("log")
    finish(fig,"05_gradient_scale")

    fig,axs = plt.subplots(2,3,figsize=(12,7),constrained_layout=True)
    names = ("mse",*LABELS)
    tick = ["MSE",*LABELS.values()]
    for row,b in zip(axs,BRANCHES):
        for ax,step in zip(row,(1000,3000,6000)):
            cells = {(loss,other):[] for loss in names for other in names}
            for r in conflicts:
                if r["run"]==f"{b}_MSE" and int(r["step"])==step:
                    cells[r["loss"],r["other"]].append(float(r["cosine"]))
            matrix = np.array([[np.mean(cells[n,m]) for m in names] for n in names])
            image = ax.imshow(matrix,cmap="coolwarm",vmin=-1,vmax=1)
            ax.set_xticks(range(len(tick)))
            ax.set_yticks(range(len(tick)))
            ax.set_xticklabels(tick,rotation=35,ha="right")
            ax.set_yticklabels(tick)
            ax.set_title(f"{b} · Update {step}")
    fig.colorbar(image,ax=axs.ravel().tolist(),label="Gradient cosine")
    finish(fig,"06_gradient_conflicts")

    fig_own,axes_own = plt.subplots(2,2,figsize=(13,8),constrained_layout=True,sharey=True)
    for i,b in enumerate(BRANCHES,7):
        fig,axs = plt.subplots(1,2,figsize=(13,4),constrained_layout=True,sharey=True)
        own = [(f"{b} plain",f"{b}_MSE"),
               *((tag,selected["single"][b][loss]["best"]) for loss,tag in LABELS.items())]
        for j,(ax,split) in enumerate(zip(axs,("development","test"))):
            lead_curves(ax,leads,[("Native","Native"),("S3","S3"),*own],split,"native_pct")
            ax.set_title(split.capitalize())
            ax.set_ylabel("Correction vs Native (%)")
            other = axes_own[i-7,j]
            lead_curves(other,leads,own,split,"own_pct")
            other.set_title(f"{b} · {split}")
            other.set_ylabel("Improvement vs own MSE (%)")
        finish(fig,f"{i:02d}_{b}_loss_leads")
    finish(fig_own,"09_own_gain_leads")

    fig,axs = plt.subplots(1,2,figsize=(12,4),constrained_layout=True,sharey=True)
    for ax,b in zip(axs,BRANCHES):
        arms = [selected["joint"][b][k] for k in ("original","tuned","balanced","top2")]
        single = min((r["best"] for r in selected["single"][b].values()),
                     key=lambda n:float(scores["development","selected",n]["mse"]))
        bars(ax,["Original","Tuned","Balanced","Top two","Best single"],[
            (s.capitalize(),[score(s,a,"own_pct") for a in (*arms,single)]) for s in ("development","test")],
            "Improvement vs own MSE (%)")
        ax.set_title(b)
    finish(fig,"10_joint")

    fig,axs = plt.subplots(1,2,figsize=(13,4.2),constrained_layout=True,sharey=True)
    for ax,split in zip(axs,("development","test")):
        lead_curves(ax,leads,final,split,"native_pct")
        ax.set_title(split.capitalize())
        ax.set_ylabel("Correction vs Native (%)")
    finish(fig,"11_final_leads")

    fig,axs = plt.subplots(4,2,figsize=(13,11),constrained_layout=True)
    terms = [("2C","C",2),("U","correction_energy",1),("G","G",1)]
    for col,split in enumerate(("development","test")):
        bars(axs[0,col],[label for label,_ in final],
             [(term,[1e6*m*score(split,a,k) for _,a in final]) for term,k,m in terms],
             "Physical MSE units (×10⁶)")
        axs[0,col].set_title(split.capitalize())
        x = np.arange(1,21)
        # Native has zero 2C, U and G by construction.
        shown = [(label,a) for label,a in final if a!="Native"]
        for j,(term,k,m) in enumerate(terms,1):
            ax = axs[j,col]
            for i,(label,a) in enumerate(shown):
                ax.plot(x,[1e6*m*float(leads[split,"selected",a,str(t)][k]) for t in x],
                        ".-",label=label,color=COLORS[i])
            ax.axhline(0,color=".45",linewidth=.7)
            ax.set_xticks(x)
            ax.set_ylabel(f"{term} (×10⁶)")
            if j==1:
                ax.legend(frameon=False,fontsize=8,ncol=2)
            if j==3:
                ax.set_xlabel("Prediction lead")
    finish(fig,"12_energy_decomposition")

    field = read_rows(out/"fixed_fields.csv")
    names = ["truth",*dict.fromkeys(a for _,a in final if a!="Native")]
    columns = ["truth",*(k for a in names[1:] for k in (a,f"{a}_remaining"))]
    case = read_json(out/"config.json")["case"]
    fig,axs = plt.subplots(2,len(columns),figsize=(2*len(columns),4),constrained_layout=True)
    limit = max(abs(float(r[k])) for r in field for k in columns)
    for row,channel in zip(axs,("u","v")):
        cells = [r for r in field if r["channel"]==channel]
        shape = (1+max(int(r["y"]) for r in cells),1+max(int(r["x"]) for r in cells))
        for ax,k in zip(row,columns):
            image = ax.imshow(np.array([float(r[k]) for r in cells]).reshape(shape),
                              cmap="RdBu_r",vmin=-limit,vmax=limit,origin="lower")
            ax.set_title(k.replace("_remaining","\nremaining"),fontsize=8)
            ax.set_axis_off()
        row[0].text(-.1,.5,channel,transform=row[0].transAxes)
    fig.suptitle(f'{case["file"]} · start {case["start"]} · lead {case["lead"]}')
    fig.colorbar(image,ax=axs.ravel().tolist(),label="Physical residual")
    finish(fig,"13_residual_fields")

    fig,axs = plt.subplots(2,3,figsize=(15,8),constrained_layout=True)
    for row,split in zip(axs,("development","test")):
        win = read_rows(out/"trajectories.csv")
        regime = read_rows(out/"regimes.csv")
        contrast = index(read_rows(out/"contrasts.csv"),("split","policy","arm","control"))
        for ax,records,field in ((row[0],regime,"re"),(row[1],win,"trajectory")):
            identifiers = sorted({r[field] for r in records if r["split"]==split})
            vals = index(records,("split","policy","arm",field))
            bars(ax,identifiers,[(b,[float(vals[split,"selected",selected["best"][b],k]["own_pct"])
                                      for k in identifiers]) for b in BRANCHES],"Improvement vs own MSE (%)")
            ax.tick_params(axis="x",labelsize=7)
            ax.set_title(f"{split.capitalize()} · {field}")
        for j,b in enumerate(BRANCHES):
            r = contrast[split,"selected",selected["best"][b],f"{b}_MSE"]
            row[2].plot([float(r["lower_pct"]),float(r["upper_pct"])],[j,j],color=COLORS[j])
            row[2].scatter(float(r["gain_pct"]),j,color=COLORS[j])
        row[2].set_yticks([0,1])
        row[2].set_yticklabels(BRANCHES)
        row[2].set_title("Paired trajectory bootstrap")
        row[2].set_xlabel("Improvement vs own MSE (%), 95% interval")
        row[2].axvline(0,color=".45",linewidth=.7)
    finish(fig,"14_regimes_and_trajectories")

    extension = [r for r in read_json(out/"exploration_results.json") if r["executed"]]
    fig,ax = plt.subplots(figsize=(max(8,1.3*len(extension)),4),constrained_layout=True)
    if extension:
        bars(ax,[f'{r["branch"]} {r["loss"]}' for r in extension],[
            (s.capitalize(),[100*(1-score(s,r["run"],"mse")/score(s,r["control"],"mse"))
                             for r in extension]) for s in ("development","test")],
            "Improvement vs matched control (%)")
    else:
        ax.text(.5,.5,"No extension met gradient calibration criteria",ha="center",va="center")
        ax.set_axis_off()
    finish(fig,"15_extensions")
