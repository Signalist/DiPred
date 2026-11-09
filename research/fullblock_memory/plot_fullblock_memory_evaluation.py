"""Display all information controls and full96-cell/trajectory comparisons."""

import json
from collections import defaultdict
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import numpy as np

from clean_reliability_common import ROOT, dump
from experiment_fullblock_memory_evaluation import OUT
from report_structured_quantile_uncertainty import METRICS


def main():
    rows=json.loads((OUT/"test_macro.json").read_text())
    table={(r["head"],r["policy"]):r for r in rows}
    names=("frozen_rich","none","last","mean","full")
    columns=(("standardized_interval_score","Interval score (lower)"),
             ("mean_absolute_group_coverage_error","Mean absolute group coverage error (lower)"),
             ("top10_energy","Top10 error energy captured (higher)"))
    fig,axes=plt.subplots(2,3,figsize=(15,7),constrained_layout=True)
    for i,mode in enumerate(("memory_dynamic","scale_fastcal")):
        for j,(metric,title) in enumerate(columns):
            values=[table[(name,mode)][metric] for name in names]
            ax=axes[i,j]
            ax.bar(range(5),values,color=["#a5a7aa","#7293b4","#c99758","#967cad","#429585"])
            span=max(values)-min(values)
            pad=max(span*.3,max(values)*.0001)
            ax.set_ylim(max(0,min(values)-pad),max(values)+pad)
            ax.set_xticks(range(5),["Frozen core","No memory","Last frame","Time mean","Full block"],rotation=20,ha="right")
            ax.set_title(mode+" | "+title,fontsize=10)
    fig.suptitle("Reused test windows2-39,Re/channel equal weight; zoomed axes\n"
                 "Frozen core differs in fit budget/capacity. Four memory-information controls are matched.")
    fig.savefig(OUT/"memory_information_comparison.png",dpi=160)
    plt.close(fig)

    summary=json.loads((OUT/"summary.json").read_text())
    old=json.loads((ROOT/"runs/centered_score_control_followup_20261109/summary.json").read_text())
    table={tuple(r[k] for k in ("split","group","channel","map","variant")):r for r in summary+old
           if r["period"]=="warm" and r["group"]!="macro"}
    groups=sorted({key[:3] for key in table},key=lambda key:(key[0],int(key[1]),key[2]))
    references=(("none","memory_dynamic"),("mean","memory_dynamic"),("core_history_old","ordinary_mse"))
    fig,axes=plt.subplots(3,1,figsize=(13,13),constrained_layout=True)
    cmap=ListedColormap(["#c95d63","#eeeeee","#3d9b81"])
    norm=BoundaryNorm([-1.5,-.5,.5,1.5],3)
    cells=[]
    for ax,(reference,mode) in zip(axes,references):
        matrix=np.zeros((8,12))
        for j,group in enumerate(groups):
            new=table[group+("full","memory_dynamic")]
            baseline=table[group+(reference,mode)]
            for i,(metric,direction) in enumerate(METRICS.items()):
                a=abs(new["coverage"]-.9) if metric=="coverage_error" else new[metric]
                b=abs(baseline["coverage"]-.9) if metric=="coverage_error" else baseline[metric]
                gain=direction*(a-b)
                state=1 if gain>1e-6 else -1 if gain< -1e-6 else 0
                matrix[i,j]=state
                cells.append(dict(candidate="full",policy="memory_dynamic",reference=reference,reference_policy=mode,
                    split=group[0],group=group[1],channel=group[2],metric=metric,value=a,baseline=b,directional_gain=gain,state=state))
        ax.imshow(matrix,cmap=cmap,norm=norm,aspect="auto")
        ax.set_xticks(range(12),[f"{'D' if split=='development' else 'T'}{group}-{channel}" for split,group,channel in groups],rotation=35,ha="right")
        ax.set_yticks(range(8),["Interval score","Width","Top10 energy","Frame top10","AURC","Worst frame","Worst patch","Coverage error"])
        count=[int((matrix==value).sum()) for value in (1,0,-1)]
        ax.set_title(f"full/memory_dynamic vs {reference}/{mode}\n{count[0]} improve / {count[1]} tie / {count[2]} regress")
    fig.suptitle("Complete96cells per reference; tolerance1e-6\n"
                 "Green improves,red regresses,gray ties. Historically reused exploratory evaluation.")
    fig.savefig(OUT/"full_memory_96cells.png",dpi=160)
    plt.close(fig)
    assert len(cells)==288
    dump(OUT/"selected_primary_cell_values.json",cells)
    cases=[r for r in json.loads((OUT/"per_case.json").read_text()) if r["start"]>=20]
    prior=[r for r in json.loads((ROOT/"runs/centered_score_control_followup_20261109/per_case.json").read_text())
           if r["map"]=="core_history_old" and r["variant"]=="ordinary_mse" and r["start"]>=20]
    buckets=defaultdict(list)
    for row in cases+prior:
        buckets[tuple(row[k] for k in ("split","file","group","channel","map","variant"))].append(row)
    comparisons=[]
    for name,mode in references+(("last","memory_dynamic"),):
        for key,values in buckets.items():
            if key[-2:]!=("full","memory_dynamic"):continue
            baseline=buckets[key[:4]+(name,mode)]
            assert len(values)==len(baseline)==38
            comparisons.append(dict(candidate="full",policy="memory_dynamic",reference=name,reference_policy=mode,
                split=key[0],file=key[1],group=key[2],channel=key[3],
                score=mean(r["standardized_interval_score"] for r in values),
                reference_score=mean(r["standardized_interval_score"] for r in baseline),
                score_gain=mean(r["standardized_interval_score"] for r in baseline)-mean(r["standardized_interval_score"] for r in values)))
    dump(OUT/"paired_trajectory_scores.json",comparisons)
    print("FULLBLOCK_MEMORY_FIGURES_COMPLETE",flush=True)


if __name__=="__main__":
    main()
