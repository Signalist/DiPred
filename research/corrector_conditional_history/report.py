"""Chinese decision report and twelve figures from the audited score tables."""
from collections import defaultdict
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from research.corrector_incremental_structure.validation import read_rows
from research.corrector_window_study.data import read_json


def read(out,name):
    return read_rows(out/f"{name}.csv")


def pct(value):
    return f"{float(value):+.3f}%"


def tag(rows):
    if not rows:
        return "数据不足"
    if all(float(r["lower_pct"])>0 for r in rows):
        return "支持"
    if any(float(r["effect_pct"])>0 for r in rows):
        return "部分支持"
    return "未支持"


def describe_pairs(rows, label):
    return "。".join(
        f"{label(r)} {pct(r['effect_pct'])} [{pct(r['lower_pct'])}, {pct(r['upper_pct'])}]"
        for r in rows
    ) + "。" if rows else "该队列缺少成对评分。"


def render(out,freeze,plan):
    scores={(r["split"],r["policy"],r["arm"]):r for r in read(out,"scores")}
    hist="historical_test"
    all_pairs=read(out,"contrasts")
    primary=[r for r in all_pairs if (r["split"],r["policy"])==(hist,"final")]
    overlap=[r for r in read(out,"history_pairing_interactions") if (r["split"],r["policy"])==(hist,"final")]
    family_differences=[r for r in read(out,"history_pairing_family_contrasts") if
                        (r["split"],r["policy"])==(hist,"final")]
    chosen=freeze["selection"]["final"]
    early=freeze["selection"]["selected"]
    proof=read_json(out/"verification.json")
    counts=proof["counts"]
    selection=[r for r in primary if r["comparison"]=="P7" and r["arm"]==chosen][0]
    result=scores[hist,"final",chosen]
    outer=[(key,plan["outer"].get(key,freeze["additional"]["re"])) for key in proof["cohorts"]["valid_outer"]]
    hold=[r for r in all_pairs if r["split"].startswith("holdout_") and r["policy"]=="final"
          and r["comparison"]=="P7" and r["arm"]==chosen]
    runtime=(chosen if float(selection["lower_pct"])>0 and len(hold)>=3 and
             all(float(r["effect_pct"])>0 for r in hold) and float(scores["period_extended","final",chosen]["s3_pct"])>0
             else "S3")
    text=["# DiPred 跨窗口残差的条件预测价值与双状态纠错","",
          f"开发固定步候选 **{chosen}**，开发早停候选 **{early}**（第 {freeze['models'][early]['selected']['step']} 步）。"
          f"上轮正式开发胜者仍为 **{plan['previous_final']}**。默认运行建议 **{runtime}**。",
          f"本轮固定步候选在历史 24 窗相对 Native {pct(result['native_pct'])}，相对 S3 {pct(result['s3_pct'])}。"
          f"后者的 95% 轨迹区间为 [{pct(selection['lower_pct'])}, {pct(selection['upper_pct'])}]。",
          f"主队列新增 {counts['main_new']} 组、复用 {counts['main_reused']} 个配置。外层新增 {counts['outer_new']} 组、"
          f"复用 {counts['outer_reused']} 个配置，另登记 {counts['prefix_phases']} 个仅到 1000 步的独立归档前缀及 {counts['teachers']} 个新教师。","",
          "## 历史 24 窗固定 6000 步主要比较","",
          "| 比较 | 模型 | 对照 | MSE 降幅 | 95% 轨迹区间 |",
          "| --- | --- | --- | ---: | ---: |"]
    for r in primary:
        if r["comparison"].startswith("P"):
            text.append(f"| {r['comparison']} | {r['arm']} | {r['control']} | {pct(r['effect_pct'])} | "
                        f"[{pct(r['lower_pct'])}, {pct(r['upper_pct'])}] |")
    text.extend(["","## 训练身份 × 评价身份","",
                 "| 输入与目标 | 正训正评 | 错训正评 | 正训错评 | 错训错评 | MSE 交互及 95% 区间 |",
                 "| --- | ---: | ---: | ---: | ---: | --- |"])
    for r in overlap:
        cells=" | ".join(f"{float(r[k]):.6g}" for k in ("correct_correct","wrong_correct","correct_wrong","wrong_wrong"))
        text.append(f"| {r['family']} {r['loss']} | {cells} | {float(r['interaction']):+.3g} "
                    f"[{float(r['lower']):+.3g}, {float(r['upper']):+.3g}] |")
    text.extend(["","## 工况和时间迁移","",
                 "| 队列 | ID_TM2 对 S3 | 本轮固定步候选对 S3 |",
                 "| --- | ---: | ---: |"])
    for split,label in [("period","连续 96"),("period_new","较晚 48"),("period_extended","额外 168"),
                        *((f"holdout_{k}",f"Re={re}") for k,re in outer)]:
        def entry(arm):
            row=scores.get((split,"final",arm))
            return pct(row["s3_pct"]) if row else "—"
        text.append(f"| {label} | {entry('ID_TM2')} | {entry(chosen)} |")
    by=defaultdict(list)
    for r in primary:
        by[r["comparison"]].append(r)
    decomposition=read(out,"history_gain_decomposition")
    gain=[r for r in decomposition if
          (r["split"],r["policy"],r["scope"],r["dimension"])==(hist,"final","overall","window")]
    gain_text="。".join(f"{r['arm']} 的 C_H={float(r['C_H']):+.3g}、U_H={float(r['U_H']):.3g}、G_H={float(r['G_H']):+.3g}" for r in gain)+"。"
    history_tracks=defaultdict(list)
    for r in decomposition:
        if (r["split"],r["policy"],r["scope"],r["dimension"])==(hist,"final","trajectory","window"):
            history_tracks[r["arm"]].append(float(r["G_H"]))
    negative="。".join(f"{arm} 有 {sum(v<0 for v in values)}/{len(values)} 条轨迹的历史净收益为负"
                      for arm,values in sorted(history_tracks.items()))+"。"
    ridge=[r for r in read(out,"conditional_history_scores")]
    ridge_mse={key:mean(float(r["mse"]) for r in ridge if r["model"]==key)
               for key in ("current","current_R","current_E")}
    evidence=[r for r in read(out,"observable_strata") if r["split"]==hist and "_pair_" in r["comparison"]]
    best=max(evidence,key=lambda r:float(r["gain"]))
    worst=min(evidence,key=lambda r:float(r["gain"]))
    groups=f"最高组为 {best['comparison']} 的 {best['feature']} 第 {best['tercile']} 组，G_H={float(best['gain']):+.3g}，覆盖 {best['trajectories']} 条轨迹。"
    groups+=f"最低组为 {worst['comparison']} 的 {worst['feature']} 第 {worst['tercile']} 组，G_H={float(worst['gain']):+.3g}，覆盖 {worst['trajectories']} 条轨迹。"
    previous_s3=[r for r in evidence if r["feature"]=="history_S3_gain" and r["tercile"]=="3"]
    groups+="上一窗 S3 收益最高的分组中，各模型历史净收益为 " + "、".join(
        f"{r['comparison']} {float(r['gain']):+.3g}" for r in previous_s3)+"。"
    early_pairs=[r for r in all_pairs if r["comparison"]=="P8" and r["policy"]=="final" and r["split"].startswith("holdout_")]
    across_re=[r for r in all_pairs if r["comparison"]=="ID_vs_IB" and r["policy"]=="final"
               and r["split"].startswith("holdout_") and r["arm"].endswith(("_MSE","_EN"))]
    re_labels={f"holdout_{key}":f"Re={value}" for key,value in outer}
    new_pairs=[r for r in all_pairs if (r["split"],r["policy"])==("holdout_additional","final")
               and r["comparison"] in ("P1","P3","P4","P5","P6")]
    interaction_positive=sum(float(r["interaction"])>0 for r in overlap)
    interaction_bounded=sum(float(r["lower"])>0 for r in overlap)
    family_labels={"correct_training_gain":"正确训练收益差", "training_evaluation_interaction":"训练与评价交互差"}
    family_text="。".join(f"{r['loss']} 的 ID 减 IB {family_labels[r['metric']]}为 {float(r['id_minus_ib']):+.3g} "
                         f"[{float(r['lower']):+.3g}, {float(r['upper']):+.3g}]" for r in family_differences)+"。"
    q=[
      ("正确样本对应关系",tag(by["P1"]),"在正确历史评价下，正确训练相对错配训练的历史窗 MSE 降幅为 "
       +describe_pairs(by["P1"],lambda r:r["arm"])+"错配训练与错配评价的条件联合分布仍可能不同。"),
      ("错配训练的收益",tag(by["shuffled_vs_I0"]),"错配模型对同目标 I0 的历史窗 MSE 降幅为 "
       +describe_pairs(by["shuffled_vs_I0"],lambda r:r["arm"])+"共同可置换样本的敏感性结果另见附表。"),
      ("样本特定与分布信息","部分支持" if any(float(r["effect_pct"])>0 for r in by["P1"]+by["shuffled_vs_I0"]) else "未支持",
       f"历史固定步中，{interaction_positive}/{len(overlap)} 个交互为正，{interaction_bounded} 个区间下界大于零。"
       +family_text+
       "正确训练能恢复错配训练丢失的收益，尤其在 ID 上，支持样本特定信息的解释。"
       "同 Re 与近时刻错配保留了部分分布信息，这一设计仍允许条件分布变化。"),
      ("ID 与 IB 的工况差异","部分支持" if across_re else "数据不足",
       "ID 相对同目标 IB 的 MSE 降幅及区间为 "+
       describe_pairs(across_re,lambda r:f"{re_labels[r['split']]} {r['arm']}")),
      ("额外条件预测价值","部分支持" if min(ridge_mse["current_R"],ridge_mse["current_E"])<ridge_mse["current"] else "未支持",
       f"折内轨迹交叉拟合 MSE 为仅当前 {ridge_mse['current']:.6g}、加真实残差 {ridge_mse['current_R']:.6g}、加剩余残差 {ridge_mse['current_E']:.6g}。"),
      ("双状态是否提高完整状态效果",tag(by["P5"]),"同模型两状态可用于测量历史的增量效果。"
       "是否改善完整状态取决于 P5。历史窗 full 相对普通模型的 MSE 降幅为 "+describe_pairs(by["P5"],lambda r:r["arm"])),
      ("full 与 missing",tag(by["P4"]),"使用同一个头和当前输入时，提供历史残差的 MSE 降幅为 "+
       describe_pairs(by["P4"],lambda r:r["arm"])),
      ("missing 与独立 I0",tag(by["P6"]),"历史窗缺失态相对同目标 I0 的 MSE 降幅为 "+
       describe_pairs(by["P6"],lambda r:r["arm"].split("__")[0])),
      ("历史条件修正分解","部分支持",gain_text+"G_H 是同模型两个发布结果之间的代数净收益。"),
      ("观测状态和收益","部分支持",groups+"分位组是探索性分析。"),
      ("EN 与不可靠历史修正","部分支持" if any(
          sum(v<0 for v in values)<sum(v<0 for v in history_tracks.get(arm.replace("_EN","_MSE"),[]))
          for arm,values in history_tracks.items() if arm.endswith("_EN")) else "未支持",
       negative+"EN 降低了两个家族的历史修正能量 U_H。IB 的净收益从负转正，ID 的净收益仍为正且略降。"),
      ("ID_TM1_early 工况迁移",tag(early_pairs),"相对常规启用 TM1 与纯 MSE 的降幅及区间为 "+
       describe_pairs(early_pairs,lambda r:f"{re_labels[r['split']]} 对 {r['control']}")),
      ("新工况压力测试",tag(new_pairs) if freeze["additional"]["executed"] else "尚未完成",
       f"按训练元数据登记 Re={freeze['additional']['re']}。"+
       ("主要成对降幅及区间为 "+describe_pairs(new_pairs,lambda r:f"{r['comparison']} {r['arm']}")+
        "四个折外教师和完整 S3 的归一化均来自各自训练观测，Corrector 的统计量来自完整子训练集。冻结主干曾接触该工况。" if freeze["additional"]["executed"]
        else freeze["additional"]["reason"])),
      ("保留跨窗口反馈","部分支持","建议在研究中保留正确匹配的剩余残差反馈。ID 双状态的历史净收益在历史窗与新工况均有正向证据，"
       "完整状态相对普通 ID 的改善和缺失态的表现限制了默认部署。"),
      ("历史可靠性建模","数据不足","值得在独立数据上检验高历史残差幅度与净收益方向的关系。分组样本数较少，阈值的迁移与门控收益仍待验证。"),
      ("研究和运行候选","支持",f"研究候选为 {chosen}，默认运行建议为 {runtime}。"),
      ("论文主张与独立确认","数据不足",
       f"可报告在本协议下的正确配对收益、训练与评价交互和同模型两状态收益。登记库存的独立轨迹为 {len(proof['independent']['verified'])} 条。"
       "历史各队列复用同 12 条轨迹，单训练种子、多项比较和冻结主干接触既有工况限制泛化。")]
    for i,(title,level,body) in enumerate(q,1):
        text.extend(["",f"## {i}. {title}","",f"**{level}。** {body}"])
    cost=read(out,"inference_cost")
    text.extend(["","## 发布路径的实测成本", "",
                 f"在预登记开发案例上用 {cost[0]['gpu']} 测量。误差为该案例的物理 MSE。", "",
                 "| 模型 | 发布路径 | 延迟中位数（毫秒） | 物理 MSE |",
                 "| --- | --- | ---: | ---: |"])
    for r in cost:
        text.append(f"| {r['arm']} | {r['mode']} | {float(r['ms_median']):.2f} | {float(r['example_mse']):.6g} |")
    text.extend(["","## 产物","",
     "[完整评分](scores.csv) · [逐帧评分](leads.csv) · [主要配对](contrasts.csv) · [历史交互](history_pairing_interactions.csv) · "
     "[ID 与 IB 交互差](history_pairing_family_contrasts.csv) · "
     "[条件收益](history_gain_decomposition.csv) · [逐窗条件收益](history_gain_windows.csv.gz) · "
     "[可置换窗口敏感性](eligible_history_pairing.csv) · [发布成本](inference_cost.csv) · "
     "[训练成本](training_cost.csv) · [图表](../../.User/corrector_conditional_history) · [复现](PROTOCOL.md)"])
    (out/"DECISION_REPORT.md").write_text("\n".join(text)+"\n",encoding="utf-8")


def plot(out,figures,freeze,plan):
    figures.mkdir(parents=True,exist_ok=True)
    hist="historical_test"
    scores=read(out,"scores")
    lead=read(out,"leads")
    pair=read(out,"contrasts")
    lookup={(r["split"],r["policy"],r["arm"]):r for r in scores}
    def save(fig,name):
        fig.tight_layout()
        fig.savefig(figures/f"{name}.png",dpi=170,bbox_inches="tight")
        plt.close(fig)
    def lines(ax,names,split=hist,key="s3_pct"):
        for arm in names:
            rows=sorted((r for r in lead if (r["split"],r["policy"],r["arm"])==(split,"final",arm)),key=lambda r:int(r["lead"]))
            assert len(rows)==20
            ax.plot(range(1,21),[float(r[key]) for r in rows],label=arm)
        ax.set_xlabel("Lead")
        ax.set_ylabel("MSE reduction (%)")
        ax.legend(fontsize=8)
    interaction=[r for r in read(out,"history_pairing_interactions") if (r["split"],r["policy"])==(hist,"final")]
    fig,axs=plt.subplots(2,2,figsize=(9,7))
    for ax,r in zip(axs.flat,interaction):
        for label,keys in (("Correct train",("correct_correct","correct_wrong")),
                           ("Shuffled train",("wrong_correct","wrong_wrong"))):
            ax.plot(("Correct eval","Shuffled eval"),[float(r[k]) for k in keys],marker="o",label=label)
        ax.set_title(f'{r["family"]} {r["loss"]}')
        ax.set_ylabel("Physical MSE")
        ax.legend(fontsize=7)
    save(fig,"01_history_pairing")
    fig,axs=plt.subplots(1,2,figsize=(13,5))
    for ax,loss in zip(axs,("MSE","EN")):
        lines(ax,[f"{a}_{loss}" for a in ("I0","IB","ID")]+[f"{a}_shuffled_{loss}" for a in ("IB","ID")])
        ax.set_title(loss)
    save(fig,"02_full_leads")
    fig,ax=plt.subplots(figsize=(11,5))
    rows=[r for r in pair if (r["split"],r["policy"])==(hist,"final") and r["comparison"] in ("P3","shuffled_vs_I0")]
    ax.bar(range(len(rows)),[float(r["effect_pct"]) for r in rows])
    ax.set_xticks(range(len(rows))); ax.set_xticklabels([r["arm"] for r in rows],rotation=45,ha="right")
    ax.set_ylabel("MSE reduction versus I0 (%)")
    save(fig,"03_vs_i0")
    fig,ax=plt.subplots(figsize=(9,5))
    rows=[r for r in pair if (r["split"],r["policy"],r["comparison"])==(hist,"final","P4")]
    x=np.arange(len(rows))
    ax.hlines(x,[float(r["lower_pct"]) for r in rows],[float(r["upper_pct"]) for r in rows])
    ax.scatter([float(r["effect_pct"]) for r in rows],x)
    ax.set_yticks(x); ax.set_yticklabels([r["arm"] for r in rows])
    ax.axvline(0,color=".6"); ax.set_xlabel("Full MSE reduction versus missing (%)")
    save(fig,"04_full_missing")
    fig,ax=plt.subplots(figsize=(9,5))
    rows=[r for r in read(out,"history_gain_decomposition") if
          (r["split"],r["policy"],r["scope"],r["dimension"])==(hist,"final","overall","window")]
    x=np.arange(len(rows))
    for i,(key,sign,label) in enumerate((("C_H",2,"2 C_H"),("U_H",-1,"−U_H"),("G_H",1,"G_H"))):
        ax.bar(x+(i-1)*.25,[sign*float(r[key]) for r in rows],width=.25,label=label)
    ax.set_xticks(x); ax.set_xticklabels([r["arm"] for r in rows]); ax.legend()
    ax.set_ylabel("Physical MSE contribution")
    save(fig,"05_history_gain")
    fig,axs=plt.subplots(2,2,figsize=(11,7))
    strata=read(out,"observable_strata")
    for ax,name in zip(axs.flat,("history_E_rms","history_S3_gain","history_R_E_cosine","current_S3_energy")):
        rows=[r for r in strata if r["split"]==hist and r["feature"]==name]
        for arm in sorted({r["comparison"] for r in rows}):
            r=[v for v in rows if v["comparison"]==arm]
            ax.plot([int(v["tercile"]) for v in r],[float(v["gain"]) for v in r],marker=".",label=arm)
        ax.set_title(name); ax.axhline(0,color=".6"); ax.legend(fontsize=7); ax.set_xlabel("Train-defined tercile")
    save(fig,"06_observable_strata")
    fig,axs=plt.subplots(1,2,figsize=(10,4))
    ridge=read(out,"conditional_history_scores")
    for ax,c in zip(axs,("u","v")):
        for model in ("current","current_R","current_E"):
            ax.plot(range(1,21),[mean(float(r["mse"]) for r in ridge if r["model"]==model and r["channel"]==c and int(r["lead"])==k) for k in range(1,21)],label=model)
        ax.set_title(c); ax.set_xlabel("Lead"); ax.set_ylabel("Out-of-track physical MSE"); ax.legend()
    save(fig,"07_conditional_ridge")
    fig,ax=plt.subplots(figsize=(12,5))
    data=[r for r in scores if r["split"].startswith("holdout_") and r["policy"]=="final" and r["arm"] not in ("Native","S3") and "__" not in r["arm"]]
    names=sorted({r["arm"] for r in data}); splits=sorted({r["split"] for r in data})
    matrix=[[float(lookup[s,"final",a]["s3_pct"]) if (s,"final",a) in lookup else np.nan for a in names] for s in splits]
    limit=np.nanmax(np.abs(matrix))
    image=ax.imshow(matrix,aspect="auto",cmap="RdBu",vmin=-limit,vmax=limit)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names,rotation=48,ha="right",fontsize=8)
    ax.set_yticks(range(len(splits))); ax.set_yticklabels(splits)
    fig.colorbar(image,ax=ax,label="MSE reduction versus S3 (%)")
    save(fig,"08_holdouts")
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    arms=("ID_MSE","ID_TM1","ID_TM1_early")
    training=read(out,"training")
    for arm in arms:
        r=[row for row in training if row["arm"]==arm]
        axs[0].plot([int(row["step"]) for row in r],[float(row["dev_mse"]) for row in r],label=arm)
    axs[0].set_xlabel("Step"); axs[0].set_ylabel("Development physical MSE"); axs[0].legend(fontsize=8)
    lines(axs[1],arms)
    save(fig,"09_early_objective")
    fig,axs=plt.subplots(1,3,figsize=(13,5))
    for ax,(file,label) in zip(axs,(("regimes","re"),("trajectories","trajectory"),("channels","channel"))):
        rows=[r for r in read(out,file) if (r["split"],r["policy"],r["arm"])==(hist,"final",freeze["selection"]["final"])]
        ax.bar(range(len(rows)),[float(r["s3_pct"]) for r in rows])
        ax.set_xticks(range(len(rows))); ax.set_xticklabels([r[label] for r in rows],rotation=35,ha="right",fontsize=7)
        ax.set_title(label)
    save(fig,"10_regime_track_channel")
    fig,axs=plt.subplots(1,2,figsize=(13,5))
    cost=read(out,"inference_cost")
    axs[0].barh([r["arm"]+"/"+r["mode"] for r in cost],[float(r["ms_median"]) for r in cost])
    axs[0].set_xlabel("Median end-to-end latency (ms)")
    train=[r for r in read(out,"training_cost") if r["arm"] in plan["new"]]
    axs[1].barh([r["arm"] for r in train],[float(r["seconds"])/60 for r in train])
    axs[1].set_xlabel("Own training minutes")
    save(fig,"11_cost")
    fields=read(out,"fixed_fields")
    names=list(dict.fromkeys(r["field"] for r in fields))
    fig,axs=plt.subplots(2,len(names),figsize=(19,5))
    limit=max(abs(float(r["value"])) for r in fields)
    for c,axes in zip(("u","v"),axs):
        for arm,ax in zip(names,axes):
            image=np.full((32,64),np.nan)
            for r in fields:
                if r["channel"]==c and r["field"]==arm:
                    image[int(r["y"]),int(r["x"])]=float(r["value"])
            assert np.isfinite(image).all()
            im=ax.imshow(image,cmap="RdBu_r",vmin=-limit,vmax=limit)
            ax.set_title(f"{c} {arm}"); ax.set_axis_off()
    fig.subplots_adjust(right=.92)
    cax=fig.add_axes([.94,.2,.012,.6])
    fig.colorbar(im,cax=cax,label="Physical residual")
    fig.savefig(figures/"12_fixed_fields.png",dpi=170,bbox_inches="tight")
    plt.close(fig)
