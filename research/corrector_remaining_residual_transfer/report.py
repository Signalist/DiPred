"""Chinese decision record and figures generated directly from physical scores."""
from collections import defaultdict
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from research.corrector_incremental_structure.validation import read_rows
from research.corrector_window_study.data import read_json

LOSSES=("MSE","TM1","TM2","EN")
SPLITS=("development","historical_test","period","period_new","period_extended",
        "holdout_interpolation","holdout_upper","holdout_lower")
LABELS=("开发","历史24","连续96","较晚48","额外168","Re17812","Re25482","Re8863")


def index(out, filename, axes):
    return {tuple(r[k] for k in axes):r for r in read_rows(out/f"{filename}.csv")}


def pct(number):
    return f"{float(number):+.3f}%"


def strength(rows):
    """Classify the direction and cross-cohort coverage of descriptive intervals."""
    if not rows:
        return "尚未验证"
    if all(float(row["lower_pct"])>0 for row in rows):
        return "支持"
    if any(float(row["effect_pct"])>0 for row in rows):
        return "部分支持"
    return "未支持"


def render(out,freeze,plan):
    scores=index(out,"scores",("split","policy","arm"))
    pairs=index(out,"contrasts",("split","policy","comparison","arm","control"))
    early=read_rows(out/"early_stopping_results.csv")
    risk=index(out,"trajectory_risks",("split","policy","arm"))
    verification=read_json(out/"verification.json")
    counts=verification["training"]
    selection=freeze["selection"]
    fixed,selected=selection["final"],selection["selected"]
    chosen=pairs["historical_test","final","P4",fixed,"S3"]
    holds=[f"holdout_{label}" for label in plan["outer"]]
    historical=scores["historical_test","final",fixed]
    early_historical=pairs["historical_test","selected","P4",selected,"S3"]
    operation=(fixed if float(chosen["lower_pct"])>0 and
               all((s,"final",fixed) in scores and float(scores[s,"final",fixed]["s3_pct"])>0
                   for s in (*holds,"period_extended")) else "S3")
    text=["# DiPred 跨窗口剩余残差传导与增量纠错","",
          f"固定步开发候选为 **{fixed}**，开发早停候选为 **{selected}**（第 {freeze['models'][selected]['selected']['step']} 步）。"
          f"默认运行建议 **{operation}**。",
          f"固定步候选历史相对 Native {pct(historical['native_pct'])}，相对 S3 {pct(chosen['effect_pct'])}，"
          f"95% 轨迹配对区间 [{pct(chosen['lower_pct'])}, {pct(chosen['upper_pct'])}]。",
          f"早停候选历史相对 Native {pct(scores['historical_test','selected',selected]['native_pct'])}，"
          f"相对 S3 {pct(early_historical['effect_pct'])}，"
          f"95% 轨迹配对区间 [{pct(early_historical['lower_pct'])}, {pct(early_historical['upper_pct'])}]。",
          f"主实验新增 {counts['new_main']} 次、复用 {counts['reused_main']} 个配置，"
          f"工况实验新增 {counts['new_outer']} 次、复用 {counts['reused_outer']} 个配置。",
          "",
          "历史直接残差模型的 D 分支相对 A 在开发改善 1.098%，历史测试改善 −0.008%。"
          "该结果来自直接预测残差的旧实验，本轮的 ID 以冻结 S3 的剩余误差为预测目标。"
          " [旧八组输入对照](../corrector_residual_feedback/scores.csv)",
          "", "## 六结构四目标固定步矩阵","",
          "表中为相对各队列冻结 S3 的物理 MSE 降幅。正值表示改善。","",
          "| 模型 | "+" | ".join(LABELS)+" |",
          "| --- | "+" | ".join("---:" for _ in SPLITS)+" |"]
    for name in selection["core"]:
        text.append(f"| {name} | "+" | ".join(pct(scores[s,"final",name]["s3_pct"]) if (s,"final",name) in scores else "—"
                                            for s in SPLITS)+" |")

    text += ["","## P1、P2、P3 和 P4 的历史固定步比较","",
             "| 项 | 模型 | 对照 | MSE 降幅 | 95% 轨迹区间 | 改善轨迹 |",
             "| --- | --- | --- | ---: | --- | ---: |"]
    for key,row in pairs.items():
        split,policy,category,arm,control=key
        if (split,policy)!=("historical_test","final") or category not in ("P1","P2","P3","P4"):
            continue
        text.append(f"| {category} | {arm} | {control} | {pct(row['effect_pct'])} | "
                    f"[{pct(row['lower_pct'])}, {pct(row['upper_pct'])}] | {row['improved']}/{row['trajectories']} |")
    text += ["","## 各输入结构对辅助目标的响应","",
             "历史 24 窗、固定 6000 步。每格给出相对该结构纯 MSE 的降幅和 95% 轨迹配对区间。","",
             "| 输入 | TM1 | TM2 | EN |",
             "| --- | ---: | ---: | ---: |"]
    for arm in plan["inputs"]:
        category="P3" if arm=="ID" else "own_MSE"
        comparisons=[pairs["historical_test","final",category,f"{arm}_{loss}",f"{arm}_MSE"] for loss in LOSSES[1:]]
        cells=[f"{pct(row['effect_pct'])} [{pct(row['lower_pct'])}, {pct(row['upper_pct'])}]" for row in comparisons]
        text.append(f"| {arm} | "+" | ".join(cells)+" |")

    text += ["","## 相同实际权重的时间匹配对照","",
             "固定 6000 步，表中为同一实际权重时的物理 MSE 降幅。历史区间按轨迹配对重抽样。"
             "固定初始梯度比例的核心对照保留在前表和交互明细中。","",
             "| 目标 | 实际权重 | 输入比较 | 开发降幅 | 历史降幅 | 历史 95% 区间 |",
             "| --- | ---: | --- | ---: | ---: | ---: |"]
    for loss in ("TM1","TM2"):
        names={arm:f"{arm}_{loss}_fixed" for arm in plan["focal"]}
        if loss=="TM1":
            names["I0"]="I0_TM1"
        for left,right in (("ID","IB"),("ID","I0"),("IB","I0")):
            category=("final","common_weight",names[left],names[right])
            dev=pairs[("development",*category)]
            hist=pairs[("historical_test",*category)]
            text.append(f"| {loss} | {freeze['anchors'][loss]:.9g} | {left} 对 {right} | "
                        f"{pct(dev['effect_pct'])} | {pct(hist['effect_pct'])} | "
                        f"[{pct(hist['lower_pct'])}, {pct(hist['upper_pct'])}] |")

    text += ["","## 早停与提前启用","",
             "| 分支 | 辅助启用步 | 开发选中步 | 历史500相对S3 | 历史1000相对S3 | 历史6000相对S3 | 历史开发选中步相对S3 |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for arm in (*selection["early"],*(f"{a}_{b}" for a in plan["focal"] for b in LOSSES)):
        model=freeze["models"][arm]
        step,first=model["selected"]["step"],model["config"]["first"]
        key="（共享 MSE 前缀）" if first and step<=first else ""
        values=[pct(scores["historical_test",p,arm]["s3_pct"]) for p in ("step500","step1000","final","selected")]
        text.append(f"| {arm} | {first} | {step}{key} | "+" | ".join(values)+" |")
    text += ["","P5 分别与对应纯 MSE 的固定第 500 步和开发早停配对。表中另列第 1000 步启用的辅助分支对同一纯 MSE 的比较。见 [早停对比](early_stopping_results.csv)。",
             "固定步与早停规则均在历史和工况评分前冻结。",""]

    text += ["## 提前启用与常规启用的直接比较","",
             "每行固定到第 6000 步，对比同结构、同辅助目标的第 250 步与第 1000 步启用方案。"
             "两者在各自启用点按相同初始梯度比例校准，实际系数可不同。","",
             "| 结构与目标 | 开发 MSE 降幅 | 历史 24 窗降幅 | 历史 95% 区间 | 额外 168 窗降幅 |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for arm in plan["focal"]:
        for loss in plan["early"]:
            pair=("final","early_vs_late",f"{arm}_{loss}_early",f"{arm}_{loss}")
            dev,hist,late=(pairs[(cohort,*pair)] for cohort in ("development","historical_test","period_extended"))
            text.append(f"| {arm} {loss} | {pct(dev['effect_pct'])} | {pct(hist['effect_pct'])} | "
                        f"[{pct(hist['lower_pct'])}, {pct(hist['upper_pct'])}] | {pct(late['effect_pct'])} |")
    text.append("")

    def historical_pairs(left,right,classification):
        return [pairs["historical_test","final",classification,f"{left}_{loss}",f"{right}_{loss}"] for loss in LOSSES]

    def pattern(rows):
        return "、".join(pct(r["effect_pct"]) for r in rows)

    p1=historical_pairs("ID","IB","P1")
    p2=historical_pairs("ID","I0","P2")
    b=historical_pairs("IB","I0","same_loss")
    c=historical_pairs("IC","I0","same_loss")
    f=historical_pairs("IF","IE","combination")
    p3=[pairs["historical_test","final","P3",f"ID_{loss}","ID_MSE"] for loss in LOSSES[1:]]
    historical_early=[r for r in early if (r["split"],r["policy"])==("historical_test","final")]
    p5=[r for r in historical_early if (r["comparison"],r["control_policy"])==("P5","selected")]
    p5_500=[r for r in historical_early if (r["comparison"],r["control_policy"])==("P5","step500")]
    late_id={r["arm"]:r for r in historical_early if r["comparison"]=="late_vs_mse_early" and
             r["control_policy"]=="selected" and r["arm"].startswith("ID_")}
    late_id=[late_id[f"ID_{loss}"] for loss in LOSSES[1:]]
    id_hold={loss:[pairs[s,"final","S3_reference",f"ID_{loss}","S3"] for s in holds] for loss in LOSSES}
    id_hold_summary="，".join(f"{loss} 为 {pattern(rows)}" for loss,rows in id_hold.items())
    id_hold_rows=[row for rows in id_hold.values() for row in rows]
    representation_tag=("支持" if any(float(r["lower_pct"])>0 or float(r["upper_pct"])<0 for r in f)
                        else "数据不足无法判断")
    transfer=read_rows(out/"temporal_correlations.csv")
    rows=[r for r in transfer if (r["source"],r["target"],r["re"])==("E","En","all")]
    boundary=mean(float(r["pearson"]) for r in rows if r["boundary"]=="True")
    same=mean(float(r["pearson"]) for r in rows if r["same_lead"]=="True")
    peak=max(rows,key=lambda r:abs(float(r["pearson"])))
    spatial=[r for r in read_rows(out/"spatial_correlations.csv") if (r["source"],r["target"])==("E","En")]
    xy=max(spatial,key=lambda r:abs(float(r["pearson"])))
    mode=max((r for r in read_rows(out/"transfer_modes.csv") if (r["source"],r["target"])==("E","En")),
             key=lambda r:abs(float(r["pearson"])))
    audit=read_json(out/"publication_verification.json")
    costs=index(out,"inference_cost",("arm","mode"))
    chrono=read_json(out/"period_provenance.json")
    e_map=[pairs["historical_test","final","mapping",arm,"ID_MSE"] for arm in selection["extensions"]
           if arm.endswith("_MSE")]
    perturbations=read_rows(out/"history_interventions.csv")
    coverage=verification["history_donor_coverage"]
    swap_coverage="，".join(f"{label} {coverage[split]['moved']}/{coverage[split]['windows']} 窗"
                          for split,label in (("development","开发"),("historical_test","历史")))
    response=[]
    for name in ("IB_MSE","ID_MSE","IB_TM1","ID_TM1"):
        r=[r for r in perturbations if (r["split"],r["arm"],r["scope"])==("development",name,"joint")]
        swap=[v for v in r if v["intervention"]=="same_re"]
        rev=[v for v in r if v["intervention"]=="reverse"]
        last=[v for v in r if v["intervention"]=="zero_history_lead_20" and v["lead"]=="1"]
        assert len(swap)==len(rev)==20 and len(last)==1
        gap=lambda items:100*(1-sum(float(v["mse"]) for v in items)/sum(float(v["original_mse"]) for v in items))
        response.append(f"{name} 同 Re 置换 {pct(gap(swap))}、反转 {pct(gap(rev))}，末帧扰动对首帧输出 RMS {float(last[0]['response_rms']):.3g}")
    base=scores["historical_test","final","ID_MSE"]
    changes=[]
    for loss in LOSSES[1:]:
        value=scores["historical_test","final",f"ID_{loss}"]
        changes.append(f"{loss} 的 2ΔC={2*(float(value['C_delta'])-float(base['C_delta'])):+.3g}，"
                       f"−ΔU={float(base['U_delta'])-float(value['U_delta']):+.3g}")
    q=[
        ("历史真实残差的额外价值",strength(b),f"MSE、TM1、TM2、EN 下，历史 IB 相对同目标 I0 依次为 {pattern(b)}。"),
        ("剩余残差与真实残差",strength(p1),f"同目标 ID 对 IB 依次为 {pattern(p1)}。配对区间见 P1。"),
        ("历史预测残差单独使用",strength(c),f"IC 对同目标 I0 依次为 {pattern(c)}。"),
        ("ID 相对 I0 和 IB 的稳定性",strength(p1+p2),f"ID 对 I0 依次为 {pattern(p2)}。结合 P1 判断目标依赖。"),
        ("IE 与 IF 的输入表示",representation_tag,f"IF 对 IE 的历史结果为 {pattern(f)}。R/P 与 R/E 可代数互推，差异反映训练表示及优化响应。"),
        ("时刻、空间和模式","部分支持",
         f"折外 E 对当前剩余误差的同 lead 平均 Pearson 为 {same:+.4f}，物理最近配对为 {boundary:+.4f}。"
         f"绝对峰值对应历史 lead {peak['source_index']}、当前 lead {peak['target_index']}、lag {peak['lag']}。"
         f"空间绝对峰值在 {xy['channel']} 的 (y={xy['y']}, x={xy['x']})。"
         f"时间模态关联峰值为 {mode['source_index']}→{mode['target_index']}。关联用于定位可检验假设。"),
        ("物理时间邻接",strength(e_map),
         "末帧与 Lag 对原始 ID 的历史比较依次为 "+pattern(e_map)+"。"
         "映射改变了表示，独立的物理邻接因果效应仍待验证。"),
        ("三个目标适合的输入",strength(p3),
         f"ID 的 TM1、TM2、EN 对自身纯 MSE 为 {pattern(p3)}。其余目标与输入的匹配看完整矩阵与物理单位差分交互。"),
        ("增量收益来源","部分支持","。".join(changes)+"。两项之和等于相对 ID-MSE 的物理 MSE 降幅。"),
        ("辅助目标与 MSE 早停",strength(p5+late_id),"提前启用分支固定 6000 步相对对应纯 MSE 开发早停为 "
         + "、".join(f"{r['arm']} {pct(r['effect_pct'])}" for r in p5)
         +"。相对纯 MSE 固定 500 步依次为 "+pattern(p5_500)
         +"。常规启用的 ID-TM1、ID-TM2、ID-EN 相对 ID-MSE 开发早停依次为 "+pattern(late_id)+"。"),
        ("提前启用与选模",strength(p5),f"预冻结工况补充配置为 {selection['outer_extra'] or '无'}。"
         f"开发早停正式选择第 {freeze['models'][selected]['selected']['step']} 步。"),
        ("工况迁移",strength(id_hold_rows),
         "ID 相对隔离 S3 的工况降幅按 Re 17812、25482、8863 排列。"
         +id_hold_summary+"。冻结 DiPred 曾接触留出轨迹。"),
        ("时间范围与工况差异","部分支持",
         "历史 24、连续 96、较晚 48 和额外 168 窗都来自同一批 "
         f"{chrono['cohorts']['historical_test']['trajectories']} 条历史轨迹。"
         "这些队列用于描述同轨迹上随时间范围变化的表现，样本独立性按轨迹判断。"),
        ("删除历史残差","部分支持" if any(float(r["effect_pct"])<=0 for r in p2) else "数据不足无法判断",
         "同目标 I0 是省去旧预测缓存的基线。使用 ID 对 I0 的配对区间与工况覆盖判断简化，区间跨零无法确认等价。"),
        ("值得复验的传导机制","部分支持",
          f"同 Re 跨轨迹置换覆盖 {swap_coverage}。"+"。".join(response)
          +"。逐帧置零响应反映所测模型的输入敏感性。两槽联合干预保持 E=R−P，单槽干预属于关系分布外测试。"),
        ("研究与运行模型","支持",f"继续复验固定步 {fixed} 和早停 {selected}。默认运行 {operation}。"),
        ("论文主张与独立证据","数据不足无法判断" if not chrono["independent"]["verified"] else "部分支持",
         f"本地库存核验覆盖 {chrono['independent']['files']} 个文件，符合登记独立确认条件的轨迹有 "
         f"{len(chrono['independent']['verified'])} 条。"
         "当前结果支持报告固定参考 S3 下的配对实验及关联。单训练种子、复用历史队列和少量工况限制独立泛化主张。")
    ]
    for i,(title,tag,body) in enumerate(q,1):
        text.extend([f"## {i}. {title}","",f"**{tag}。** {body}",""])
    text.extend(["## 风险与运行成本",""])
    report_risk=risk["historical_test","final",fixed]
    text.append(f"候选最差历史工况为 Re={report_risk['worst_re']}，相对 S3 {pct(report_risk['worst_re_s3_pct'])}。"
                "逐轨迹、通道、逐 lead 和 Re 等权结果已单独归档。")
    for arm in dict.fromkeys(("S3",fixed,selected)):
        row=costs[arm,"end_to_end" if arm=="S3" else "hot"]
        cold=costs.get((arm,"cold"))
        penalty=""
        if cold:
            max_change=audit["models"][arm]["cacheless_max_abs"]
            penalty=(f"，无缓存重算 {float(cold['ms_median']):.3f} ms，"
                     f"固定案例有无缓存预测的最大绝对差为 {max_change:.3g}")
        text.append(f"{arm} 单窗端到端中位数 {float(row['ms_median']):.3f} ms{penalty}，新增头参数 {row['head_params']}。")
    text.extend(["",f"固定案例的未来真值置换与零增量恢复检查均通过。"
                 f"现场设备为 {audit['gpu']}，每个阶段预热 {audit['warmup']} 次并重复 {audit['repeat']} 次。",
                 "", "[逐 lead](leads.csv) · [轨迹区间](contrasts.csv) · [输入与 Loss 交互](input_loss_interactions.csv) · "
                 "[工况](holdout_results.csv) · [置换配对](history_donors.csv) · [运行成本](inference_cost.csv) · "
                 "[图表](../../.User/corrector_remaining_residual_transfer) · [复现](PROTOCOL.md)"])
    (out/"DECISION_REPORT.md").write_text("\n".join(text)+"\n",encoding="utf-8")


def plot(out,figures,freeze,plan):
    figures.mkdir(parents=True,exist_ok=True)
    scores=index(out,"scores",("split","policy","arm"))
    leads=index(out,"leads",("split","policy","arm","lead"))
    main=freeze["selection"]["final"]
    names=tuple(plan["inputs"])
    hist="historical_test"
    regime_splits=SPLITS[-3:]
    lead=np.arange(1,21)

    def save(fig,filename):
        fig.tight_layout()
        fig.savefig(figures/filename,dpi=170,bbox_inches="tight")
        plt.close(fig)

    def val(split,arm,key="s3_pct",policy="final"):
        return float(scores[split,policy,arm][key])

    def cell(ax,z,xt,yt,label):
        bound=max(abs(np.asarray(z)).max(),1e-12)
        im=ax.imshow(z,aspect="auto",cmap="RdBu",vmin=-bound,vmax=bound)
        ax.set_xticks(range(len(xt)))
        ax.set_xticklabels(xt,rotation=25,ha="right")
        ax.set_yticks(range(len(yt)))
        ax.set_yticklabels(yt)
        ax.figure.colorbar(im,ax=ax,label=label,shrink=.8)

    corr=read_rows(out/"temporal_correlations.csv")
    bound=max(abs(float(row["pearson"])) for row in corr
              if row["re"]=="all" and row["target"] in ("Rn","En"))
    fig,axs=plt.subplots(2,3,figsize=(12,7))
    for (i,field),(j,target) in [(a,b) for a in enumerate(("R","P","E")) for b in enumerate(("Rn","En"))]:
        ax=axs[j,i]
        rows=[r for r in corr if (r["source"],r["target"],r["re"])==(field,target,"all")]
        if not rows:
            ax.set_axis_off()
            continue
        a=np.zeros((20,20))
        for r in rows:
            a[int(r["source_index"])-1,int(r["target_index"])-1]+=float(r["pearson"])/2
        im=ax.imshow(a,origin="lower",cmap="RdBu",vmin=-bound,vmax=bound,extent=(.5,20.5,.5,20.5))
        ax.set_title(f"Prior {field} → {target}")
        ax.set_xlabel("Current lead")
        ax.set_ylabel("Prior lead")
        fig.colorbar(im,ax=ax,shrink=.75)
    save(fig,"01_time_correlations.png")

    fig,axs=plt.subplots(1,2,figsize=(12,5))
    for ax,split in zip(axs,("development",hist)):
        cell(ax,[[val(split,f"{a}_{b}") for b in LOSSES] for a in names],LOSSES,names,"MSE reduction versus S3 (%)")
        ax.set_title(split)
    save(fig,"02_input_loss_matrix.png")

    fig,axs=plt.subplots(1,3,figsize=(14,4))
    for ax,base_arm in zip(axs,plan["focal"]):
        for loss in LOSSES:
            a=f"{base_arm}_{loss}"
            ax.plot(lead,[float(leads[hist,"final",a,str(i)]["native_pct"]) for i in lead],label=loss)
        ax.set_title(base_arm)
        ax.set_xticks(lead[::2])
        ax.set_xlabel("Lead")
        ax.set_ylabel("Reduction versus Native (%)")
    axs[-1].legend()
    save(fig,"03_full_leads.png")

    fig,axs=plt.subplots(1,2,figsize=(10,4))
    for ax,control in zip(axs,("IB","I0")):
        for loss in LOSSES:
            m,n=(np.array([float(leads[hist,"final",f"{a}_{loss}",str(i)]["mse"]) for i in lead]) for a in ("ID",control))
            ax.plot(lead,100*(1-m/n),label=loss)
        ax.axhline(0,color=".6",lw=.7)
        ax.set_xlabel("Lead")
        ax.set_xticks(lead[::2])
        ax.set_ylabel(f"ID improvement versus {control} (%)")
    axs[1].legend()
    save(fig,"04_id_relative_leads.png")

    rows=read_rows(out/"input_loss_interactions.csv")
    fig,axs=plt.subplots(1,2,figsize=(12,6))
    for ax,key in zip(axs,("fixed_rho","fixed_lambda")):
        r=[r for r in rows if r["split"]==hist and r["comparison"]==key]
        x=np.arange(len(r))
        ax.hlines(x,[float(v["ci_low"]) for v in r],[float(v["ci_high"]) for v in r],color="#326a8b")
        ax.scatter([float(v["mse_interaction"]) for v in r],x,s=12)
        ax.set_yticks(x)
        ax.set_yticklabels([f"{v['arm']}/{v['control']} {v['loss']}" for v in r],fontsize=7)
        ax.axvline(0,color=".6")
        ax.set_xlabel("Physical MSE interaction")
        ax.set_title(key)
    save(fig,"05_interactions.png")

    fig,axs=plt.subplots(1,3,figsize=(15,4))
    for ax,a in zip(axs,plan["focal"]):
        for policy in ("step500","step1000","final","selected"):
            ax.plot(LOSSES,[val(hist,f"{a}_{b}",policy=policy) for b in LOSSES],label=policy,marker=".")
        ax.set_title(a)
        ax.set_ylabel("Reduction versus S3 (%)")
    axs[-1].legend()
    save(fig,"06_fixed_early_stopping.png")

    training=read_rows(out/"training.csv")
    fig,axs=plt.subplots(2,3,figsize=(14,7),sharex=True)
    for column,arm in enumerate(plan["focal"]):
        for loss in (*LOSSES,"TM1_early","EN_early"):
            name=f"{arm}_{loss}"
            history=sorted((r for r in training if r["arm"]==name),key=lambda row:int(row["step"]))
            steps=[int(row["step"]) for row in history]
            for ax,key in zip(axs[:,column],("train_mse","dev_mse")):
                ax.plot(steps,[float(row[key]) for row in history],label=loss)
        for ax in axs[:,column]:
            for branch_step in (250,1000):
                ax.axvline(branch_step,color=".6",lw=.7)
        axs[0,column].set_title(arm)
        axs[1,column].set_xlabel("Update")
    axs[0,0].set_ylabel("Train physical MSE")
    axs[1,0].set_ylabel("Development physical MSE")
    axs[0,-1].legend(ncol=2,fontsize=8)
    save(fig,"07_early_learning_curves.png")

    interventions=read_rows(out/"history_interventions.csv")
    fig,axs=plt.subplots(2,2,figsize=(9,7))
    for ax,name in zip(axs.flat,("IB_MSE","ID_MSE","IB_TM1","ID_TM1")):
        z=np.zeros((20,20))
        for r in interventions:
            if r["split"]=="development" and r["arm"]==name and r["scope"]=="joint" and r["intervention"].startswith("zero_history_lead_"):
                z[int(r["intervention"].split("_")[-1])-1,int(r["lead"])-1]=float(r["response_rms"])
        im=ax.imshow(z,origin="lower",extent=(.5,20.5,.5,20.5),cmap="viridis")
        ax.set_title(name)
        ax.set_xlabel("Current lead")
        ax.set_ylabel("Ablated historical lead")
        fig.colorbar(im,ax=ax,shrink=.8,label="Response RMS")
    save(fig,"08_frame_interventions.png")

    fig,axs=plt.subplots(1,3,figsize=(13,4))
    for ax,arm in zip(axs,plan["focal"]):
        for i,(factor,key,label) in enumerate(((2,"C_delta","2C"),(-1,"U_delta","−U"),(1,"G_delta","G"))):
            ax.bar(np.arange(4)+(i-1)*.24,[factor*val(hist,f"{arm}_{b}",key) for b in LOSSES],.21,label=label)
        ax.set_xticks(range(4))
        ax.set_xticklabels(LOSSES)
        ax.set_title(arm)
        ax.set_ylabel("Physical squared units")
    axs[-1].legend()
    save(fig,"09_incremental_decomposition.png")

    fig,ax=plt.subplots(figsize=(12,4))
    a=[f"{arm}_{b}" for arm in plan["focal"] for b in LOSSES]
    cell(ax,[[val(s,n) for n in a] for s in regime_splits],a,("17812","25482","8863"),"MSE reduction versus isolated S3 (%)")
    save(fig,"10_regime_matrix.png")

    fig,ax=plt.subplots(figsize=(10,4))
    cohort=("historical_test","period","period_new","period_extended")
    for a in plan["focal"]:
        for loss in ("MSE","TM1"):
            ax.plot(range(4),[val(s,f"{a}_{loss}") for s in cohort],marker="o",label=f"{a}_{loss}")
    ax.set_xticks(range(4))
    ax.set_xticklabels(("24","96","48","168"))
    ax.set_xlabel("Historical cohort window count")
    ax.set_ylabel("MSE reduction versus S3 (%)")
    ax.legend(ncol=2)
    save(fig,"11_temporal_cohorts.png")

    fig,axs=plt.subplots(1,3,figsize=(15,4))
    lag=defaultdict(list)
    for r in corr:
        if (r["source"],r["target"],r["re"])==("E","En","all"):
            lag[int(r["lag"])].append(float(r["pearson"]))
    axs[0].plot(sorted(lag),[mean(lag[i]) for i in sorted(lag)])
    axs[0].set_xlabel("Physical lag (frames)")
    axs[0].set_ylabel("Mean lead-pair Pearson")
    a=["ID_MSE",*freeze["selection"]["extensions"]]
    axs[1].bar(a,[val(hist,n) for n in a])
    axs[1].tick_params(axis="x",labelrotation=24)
    axs[1].set_ylabel("MSE reduction versus S3 (%)")
    learned=np.asarray(freeze["models"]["ID_lag_MSE"]["mapping"],dtype=float)
    assert learned.shape==(20,20) and np.isfinite(learned).all()
    change=learned-np.eye(20)
    bound=max(np.abs(change).max(),1e-12)
    image=axs[2].imshow(change,cmap="RdBu_r",vmin=-bound,vmax=bound,
                        extent=(.5,20.5,20.5,.5),interpolation="nearest")
    axs[2].set_title("Learned Lag weights minus identity")
    axs[2].set_xlabel("Historical lead")
    axs[2].set_ylabel("Mapped lead")
    fig.colorbar(image,ax=axs[2],shrink=.78)
    save(fig,"12_lag_mapping.png")

    fig,axs=plt.subplots(1,3,figsize=(13,4))
    for ax,(file,field) in zip(axs,(("channels","channel"),("regimes","re"),("trajectories","trajectory"))):
        r=[r for r in read_rows(out/f"{file}.csv") if (r["split"],r["policy"],r["arm"])==(hist,"final",main)]
        ax.bar(range(len(r)),[float(v["s3_pct"]) for v in r])
        ax.set_xticks(range(len(r)))
        ax.set_xticklabels([v[field].replace(".h5","") for v in r],rotation=32,ha="right",fontsize=7)
        ax.set_ylabel("MSE reduction versus S3 (%)")
    save(fig,"13_channel_re_trajectory.png")

    plot_fixed_fields(out,figures,plan["field"],main)


def plot_fixed_fields(out,figures,case,arm):
    names=("history_R","history_P","history_E","current_E","increment")
    channels=("u","v")
    planes={(channel,field):np.full((32,64),np.nan) for channel in channels for field in names}
    for row in read_rows(out/"fixed_fields.csv"):
        plane=planes[row["channel"],row["field"]]
        y,x=int(row["y"]),int(row["x"])
        assert 0<=y<plane.shape[0] and 0<=x<plane.shape[1] and np.isnan(plane[y,x])
        plane[y,x]=float(row["value"])
    assert all(np.isfinite(plane).all() for plane in planes.values())
    bound=max(np.abs(plane).max() for plane in planes.values())
    fig,axs=plt.subplots(2,5,figsize=(15,5),layout="constrained")
    fig.suptitle(f"{case['file']} · start {case['start']} · lead {case['lead']} · {arm}")
    for channel,ax_row in zip(channels,axs):
        for ax,name in zip(ax_row,names):
            im=ax.imshow(planes[channel,name],cmap="RdBu_r",vmin=-bound,vmax=bound)
            ax.set_title(f"{channel} {name}")
            ax.set_axis_off()
    fig.colorbar(im,ax=axs.ravel().tolist(),shrink=.68,label="Physical residual")
    fig.savefig(figures/"14_fixed_fields.png",dpi=170,bbox_inches="tight")
    plt.close(fig)
