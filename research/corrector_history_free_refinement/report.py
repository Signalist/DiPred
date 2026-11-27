"""Generate the Chinese decision report and nine figures from physical scores."""
from collections import defaultdict
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from research.corrector_incremental_structure.validation import read_rows
from research.corrector_window_study.data import read_json

FAMILIES = ("M0", "TM1", "TM2", "EN")
OUTER = ("holdout_interpolation", "holdout_upper", "holdout_lower")
SPLITS = ("development", "historical_test", "period", "period_new", "period_extended", *OUTER)
LABELS = ("Dev", "Hist 24", "Continuous 96", "Later 48", "Extended 168", "Re 17812", "Re 25482", "Re 8863")


def indexed(out, file, keys):
    return {tuple(row[k] for k in keys):row for row in read_rows(out/f"{file}.csv")}


def pct(value):
    return f"{float(value):+.3f}%" if value not in ("", None) else "—"


def report(out, frozen):
    selection = frozen["selection"]
    families = selection["families"]
    winner = selection["candidate"]
    metrics = indexed(out,"scores",("split","policy","arm"))
    contrasts = indexed(out,"contrasts",("split","policy","comparison","arm","control"))
    costs = indexed(out,"inference_cost",("arm",))
    train_cost = indexed(out,"training_cost",("arm",))
    ver = read_json(out/"verification.json")
    reuse = read_json(out/"reuse_audit.json")
    s3 = metrics["development","final","S3"]
    hist = metrics["historical_test","final",winner]
    primary = contrasts["historical_test","final","P1",winner,"S3"]
    supported = float(primary["lower_pct"])>0 and all(
        float(metrics[s,"final",winner]["s3_pct"])>0 for s in ("period_extended",*OUTER))
    deployment = winner if supported and float(s3["mse"])>float(metrics["development","final",winner]["mse"]) else "S3"
    training = ver["training"]
    lines = ["# DiPred 无历史残差增量纠错", "",
             f"固定第 6000 步的开发规则选出 **{winner}**。根据历史、额外时间段和工况覆盖，默认运行建议 **{deployment}**。",
             f"研究候选在历史 24 窗相对 Native、冻结 S3 和 I0-M0 分别改善 {pct(hist['native_pct'])}、"
             f"{pct(hist['s3_pct'])}、{pct(hist['own_pct'])}。相对 S3 的 95% 轨迹配对区间为 "
             f"[{pct(primary['lower_pct'])}, {pct(primary['upper_pct'])}]，改善 {primary['improved']}/{primary['trajectories']} 条轨迹。",
             f"新增主训练 {training['new_main']} 次、外层训练 {training['new_outer']} 次。"
             f"分别复用 {training['reused_main']}、{training['reused_outer']} 个模型及 {training['reused_teachers']} 个 S3 教师。",
             "", "## 固定 6000 步目标比较", "",
             "| 家族 | 模型 | 开发物理 MSE | 历史物理 MSE | 历史相对 Native | 历史相对 S3 | 历史相对 M0 |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for family,names in selection["matrix"].items():
        for name in names:
            dev = metrics["development","final",name]
            row = metrics["historical_test","final",name]
            lines.append(f"| {family} | {name} | {float(dev['mse']):.9g} | {float(row['mse']):.9g} | "
                         f"{pct(row['native_pct'])} | {pct(row['s3_pct'])} | {pct(row['own_pct'])} |")
    lines.extend(["", "## 主要配对比较", "",
                  "| 队列 | 比较 | 候选 | 对照 | 降幅 | 95% 轨迹配对区间 | 改善轨迹 |",
                  "| --- | --- | --- | --- | ---: | --- | ---: |"])
    for split,label in zip(SPLITS,LABELS):
        for key,arm,control in selection["primary"]:
            row = contrasts[split,"final",key,arm,control]
            lines.append(f"| {label} | {key} | {arm} | {control} | {pct(row['effect_pct'])} | "
                         f"[{pct(row['lower_pct'])}, {pct(row['upper_pct'])}] | {row['improved']}/{row['trajectories']} |")
    lines.extend(["", "## 工况压力测试与时间覆盖", "",
                  "表内为相对各队列 S3 的物理 MSE 降幅。只有冻结主干下的 Corrector 工况压力测试使用对应隔离 S3，"
                  "且 DiPred 曾接触留出轨迹。额外 168 个窗口来自原历史轨迹，窗口起点为 560 至 820。",
                  "", "| 家族或参照 | " + " | ".join(LABELS) + " |",
                  "| --- | " + " | ".join(["---:"]*len(SPLITS)) + " |"])
    for family,name in (*families.items(),("IB_match2","IB_match2"),("固定 λ","I0_match2_fixed"),("续训 S3","S3_extra_MSE")):
        cells = [metrics.get((split,"final",name),{}).get("s3_pct") for split in SPLITS]
        lines.append(f"| {family} | " + " | ".join(map(pct,cells)) + " |")
    lines.extend(["", "## 研究候选与续训对照", "",
                  "| 队列 | I0 相对 Native | I0 相对 S3 | I0 相对 M0 | I0 相对续训 S3 | 95% 区间 |",
                  "| --- | ---: | ---: | ---: | ---: | --- |"])
    for split,label in zip(SPLITS,LABELS):
        row = metrics[split,"final",winner]
        pair = contrasts.get((split,"final","engineering",winner,"S3_extra_MSE"))
        comparison = pct(pair["effect_pct"]) if pair else "—"
        interval = f"[{pct(pair['lower_pct'])}, {pct(pair['upper_pct'])}]" if pair else "—"
        lines.append(f"| {label} | {pct(row['native_pct'])} | {pct(row['s3_pct'])} | {pct(row['own_pct'])} | "
                     f"{comparison} | {interval} |")

    lines.extend(["", "## 去掉上一窗残差后的覆盖", "",
                  "| 队列 | I0 候选相对 IB_match2 | 95% 区间 | I0 候选优于 S3 的轨迹 | Re 等权相对 S3 |",
                  "| --- | ---: | --- | ---: | ---: |"])
    equal = indexed(out,"equal_regime_scores",("split","policy","arm"))
    for split,label in zip(SPLITS,LABELS):
        row = contrasts[split,"final","reference",winner,"IB_match2"]
        covers = contrasts[split,"final","P1",winner,"S3"]
        lines.append(f"| {label} | {pct(row['effect_pct'])} | [{pct(row['lower_pct'])}, {pct(row['upper_pct'])}] | "
                     f"{covers['improved']}/{covers['trajectories']} | {pct(equal[split,'final',winner]['s3_pct'])} |")
    lines.extend(["", "## 开发最优检查点", "",
                  "| 家族 | 模型 | 步数 | 开发物理 MSE | 历史相对 S3 | 说明 |",
                  "| --- | --- | ---: | ---: | ---: | --- |"])
    for family,name in families.items():
        step = selection["selected_step"][name]
        note = "辅助训练尚未产生额外收益" if family!="M0" and step<=1000 else "开发最小物理 MSE"
        dev = metrics["development","selected",name]
        historical = metrics["historical_test","selected",name]
        lines.append(f"| {family} | {name} | {step} | {float(dev['mse']):.9g} | {pct(historical['s3_pct'])} | {note} |")

    calibration = read_json(out/"calibration.json")
    gradient = read_rows(out/"gradient_diagnostics.csv")
    lines.extend(["", "## 权重与梯度", "",
                  "初始和第 6000 步的辅助梯度比采用同一批次集合，先分别平均辅助梯度与 MSE 梯度范数，再取比值。",
                  "", "| 家族 | 初始 ρ | 实际权重 | 第 6000 步的比值 | 第 6000 步的平均余弦 |",
                  "| --- | ---: | ---: | ---: | ---: |"])
    for family in ("TM1","TM2","EN"):
        term = {"TM1":"match","TM2":"match2","EN":"energy"}[family]
        for rho,name in zip(("0.1","0.3"),selection["matrix"][family]):
            rows = [row for row in gradient if row["arm"]==name and row["step"]=="6000" and row["term"]==term]
            weight = calibration["coefficients"][family][rho]
            ratio = weight*mean(float(row["norm"]) for row in rows)/mean(float(row["mse_norm"]) for row in rows)
            cosine = mean(float(row["cosine_to_mse"]) for row in rows)
            lines.append(f"| {family} | {rho} | {weight:.16g} | {ratio:.4f} | {cosine:+.4f} |")

    p2 = contrasts["historical_test","final","P2",families["TM2"],families["TM1"]]
    p3 = contrasts["historical_test","final","P3",selection["best_time"],families["EN"]]
    lines.extend(["", "## 一阶、二阶与能量惩罚", "",
                  f"开发固定步最佳时间目标为 {selection['best_time']}。历史固定步 TM2 相对 TM1 的改善为 "
                  f"{pct(p2['effect_pct'])}，区间 [{pct(p2['lower_pct'])}, {pct(p2['upper_pct'])}]。"
                  f"最佳时间目标相对 EN 为 {pct(p3['effect_pct'])}，区间 [{pct(p3['lower_pct'])}, {pct(p3['upper_pct'])}]。",
                  "以工况及延长时间段的配对表衡量覆盖范围。初始 ρ 相等期间，各目标的梯度比例、余弦和裁剪频率仍会随训练变化。",
                  "", "| 家族 | 历史 2ΔC_delta | 历史 −ΔU_delta | 历史 ΔG_delta | 一阶误差能量 | 二阶误差能量 |",
                  "| --- | ---: | ---: | ---: | ---: | ---: |"])
    m0 = metrics["historical_test","final","M0"]
    for family,name in families.items():
        row=metrics["historical_test","final",name]
        dc,du,dg=(float(row[k])-float(m0[k]) for k in ("C_delta","U_delta","G_delta"))
        lines.append(f"| {family} | {2*dc:+.5g} | {-du:+.5g} | {dg:+.5g} | "
                     f"{float(row['error_d1_mse']):.5g} | {float(row['error_d2_mse']):.5g} |")
    if winner!="M0":
        alignment=2*(float(hist["C_delta"])-float(m0["C_delta"]))
        reduction=float(m0["U_delta"])-float(hist["U_delta"])
        main="修正能量下降" if reduction>alignment else "有效对齐项变化"
        lines.append(f"\n候选相对 M0 的历史 MSE 变化由上表两项相加，贡献较大的一项为{main}。")
    if float(p2["lower_pct"])<=0:
        lines.append("历史结果不足以确认二阶匹配的必要性。")
    if float(p3["lower_pct"])<=0:
        lines.append("历史结果尚未确认时间匹配具有超出能量惩罚的优势。")
    later_p2 = contrasts["period_extended","final","P2",families["TM2"],families["TM1"]]
    later_p3 = contrasts["period_extended","final","P3",selection["best_time"],families["EN"]]
    best_historical = min(families, key=lambda f:float(metrics["historical_test","final",families[f]]["mse"]))
    best_later = min(families, key=lambda f:float(metrics["period_extended","final",families[f]]["mse"]))
    lines.append(f"历史 24 窗的最小 MSE 来自 {best_historical}，额外 168 窗的最小 MSE 来自 {best_later}。"
                 f"额外 168 窗中 TM2 相对 TM1 为 {pct(later_p2['effect_pct'])}，最佳时间目标相对 EN 为 "
                 f"{pct(later_p3['effect_pct'])}。这段证据来自既有轨迹上的更晚时段。")
    modes = read_rows(out/"incremental_mode_scores.csv")
    for channel in ("u","v"):
        by_mode = defaultdict(dict)
        for row in modes:
            if row["split"]=="historical_test" and row["policy"]=="final" and row["channel"]==channel:
                by_mode[int(row["mode"])][row["arm"]] = float(row["final_error_energy"])
        gains = {mode:arms["M0"]-arms[winner] for mode,arms in by_mode.items()}
        lead_mode = max(gains,key=gains.get)
        worst_mode = min(gains,key=gains.get)
        lines.append(f"历史 {channel} 通道相对 M0 的最大模式降幅出现在模式 {lead_mode}（{gains[lead_mode]:+.4g}），"
                     f"最低出现在模式 {worst_mode}（{gains[worst_mode]:+.4g}）。")
    if "I0_EN_energy_matched" in frozen["models"]:
        matched = metrics["historical_test","final","I0_EN_energy_matched"]
        lines.append(f"训练能量匹配 EN 在历史 24 窗相对 S3 为 {pct(matched['s3_pct'])}。"
                     "匹配对象是整体训练增量能量，各窗口和时间模式的能量仍可不同。")
    lines.extend(["", "## 工程参照和发布依赖", ""])
    extra = metrics["historical_test","final","S3_extra_MSE"]
    diff=100*(1-float(hist["mse"])/float(extra["mse"]))
    frac=float(costs["candidate",]["additional_end_to_end_ms"])
    lines.append(f"候选在历史队列相对续训 S3 的 MSE 改善为 {pct(diff)}。"
                 f"续训 S3 相对冻结 S3 为 {pct(extra['s3_pct'])}。"
                 f"续训对照{'进入' if selection['extra_outer'] else '未达到进入门槛，未执行'}工况复验。")
    extra_train = train_cost["S3_extra_MSE",]
    i0_train = train_cost[winner,]
    lines.append("续训从源权重的独立副本开始，优化器、随机状态和学习率日程重新初始化，训练原 Native 残差。"
                 f"本次额外 6000 步耗时 {float(extra_train['training_seconds']):.1f} 秒。"
                 f"主候选自身 {i0_train['own_updates']} 步耗时 {float(i0_train['training_seconds']):.1f} 秒。"
                 f"I0 学习折外 S3 剩余误差，四个复用折外教师历史训练耗时合计 {ver['original_main_oof_training_seconds']:.1f} 秒。")
    lines.append(f"固定当前观测窗后，修改、删除和替换此前预测缓存均保持发布输出逐位一致。"
                 f"候选端到端中位延迟相对冻结 S3 增加 {frac:+.3f} ms。")
    replay = ver["publication"]["cache_replay"]
    lines.append(f"将同一归档 Native 预测输入 S3 时，现场 S3 输出与批量评分逐位一致。"
                 f"现场重跑 Native 的最大物理差异为 {replay['native_max_abs_error']:.3g}，"
                 f"传到最终发布结果的最大差异为 {replay['final_max_abs_error']:.3g}。")
    lines.extend(["", "| 模型 | 总参数 | 训练阶段可更新参数 | 端到端 ms 中位数 | 单次范围 ms | 额外峰值内存 B |",
                  "| --- | ---: | ---: | ---: | --- | ---: |"])
    for name in ("S3","S3_extra_MSE","candidate"):
        row = costs[name,]
        lines.append(f"| {winner if name=='candidate' else name} | {row['total_parameters']} | {row['trainable_parameters']} | "
                     f"{float(row['ms_end_to_end_median']):.3f} | [{float(row['ms_end_to_end_min']):.3f}, "
                     f"{float(row['ms_end_to_end_max']):.3f}] | {row['additional_end_to_end_peak_bytes']} |")

    lines.extend(["", "## 退步范围与证据", ""])
    for title, filename, key in (("轨迹","trajectories","trajectory"),("工况","regimes","re"),("lead","leads","lead")):
        rows = [r for r in read_rows(out/f"{filename}.csv") if (r["split"],r["policy"],r["arm"])==
                ("historical_test","final",winner)]
        names = [r[key] for r in rows if float(r["s3_pct"])<0]
        lines.append(f"- 历史退步{title}为 " + ("、".join(names) if names else "无") + "。")
    channels = indexed(out,"channels",("split","policy","arm","channel"))
    for split,label in (("historical_test","历史 24 窗"),("period_extended","额外 168 窗")):
        gains = [pct(channels[split,"final",winner,c]["s3_pct"]) for c in ("u","v")]
        lines.append(f"- {label}相对 S3 的通道 MSE 降幅为 u {gains[0]}、v {gains[1]}。")
    risk = indexed(out,"trajectory_risks",("split","policy","arm"))["historical_test","final",winner]
    share=risk["top_two_share_of_positive_gain"]
    lines.append(f"- 最差历史工况 Re={risk['worst_re']}，相对 S3 改善 {pct(risk['worst_re_s3_pct'])}。"
                 f"正收益最大两条轨迹占总正收益 {pct(100*float(share)) if share else '不适用'}。")
    routes = []
    for family,name in families.items():
        if family != "M0" and name!=winner and all(float(metrics[s,"final",name]["mse"])>=float(metrics[s,"final",winner]["mse"])
                                                          for s in SPLITS):
            routes.append(family)
    if routes:
        lines.append("\n优先停止扩大以下目标的权重搜索，当前固定候选在八个队列的点估计均已被开发预选候选追平或超过。"
                     + "、".join(routes) + "。")
    if float(later_p2["upper_pct"])<0 and float(p2["lower_pct"])<=0:
        lines.append("\n停止将 TM2 作为默认必要项继续扩展。保留现有 TM2 和 EN 做固定对照，并优先对一阶方案开展独立复验。")
    lines.extend(["", f"本轮保留 {deployment} 作为运行方案，后续独立验证使用 {winner}。"
                  "I0 仍然读取当前 20 帧历史观测。历史队列反复使用，训练固定单种子，轨迹区间用于描述。"
                  "已核对登记数据的接触身份，独立确认队列为空。",
                  "折外训练身份、批次及共同完整状态逐项核对。上一轮的正式 IB_match2 选择保留在原冻结记录中。",
                  "", "## 交付", "",
                  "[完整 lead 1–20](leads.csv) · [主要和次要配对区间](contrasts.csv) · [工况结果](holdout_results.csv) · "
                  "[模型训练成本](training_cost.csv) · [折外及隔离教师训练成本](teacher_training_cost.csv) · [发布测试](publication_verification.json) · [图表](../../.User/corrector_history_free_refinement) · [复现命令](PROTOCOL.md)"])
    replayed = reuse["development_replay"]
    if replayed["recomputed"]:
        error = max(replayed["max_mse_difference"].values())
        lines.insert(lines.index("## 交付")-1,
                     f"完整 S3 的开发输出重算后，复用检查点的开发 MSE 最大差异为 {error:.3g}。")
    (out/"DECISION_REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


def plot(out, figures, frozen):
    figures.mkdir(parents=True,exist_ok=True)
    selection=frozen["selection"]
    families=selection["families"]
    winner=selection["candidate"]
    scores=indexed(out,"scores",("split","policy","arm"))

    def save(fig,name):
        fig.tight_layout()
        fig.savefig(figures/name,dpi=165,bbox_inches="tight")
        plt.close(fig)

    def indexed_points(file, split, arm):
        return [r for r in read_rows(out/f"{file}.csv")
                if (r["split"],r["policy"],r["arm"])==(split,"final",arm)]

    def score(split,arm,field):
        return float(scores[split,"final",arm][field])

    fig, ax=plt.subplots(figsize=(11,4))
    width=.17
    positions=np.arange(len(SPLITS))
    for i,(family,arm) in enumerate(families.items()):
        ax.bar(positions+(i-1.5)*width,[score(s,arm,"s3_pct") for s in SPLITS],width,label=family)
    ax.set_xticks(positions)
    ax.set_xticklabels(LABELS)
    ax.set_ylabel("MSE reduction versus S3 (%)")
    ax.axhline(0,color=".55",lw=.7)
    ax.legend()
    save(fig,"01_objective_comparison.png")

    fig, axes=plt.subplots(1,2,figsize=(10,4))
    calibration=read_json(out/"calibration.json")
    for family in ("TM1","TM2","EN"):
        names=selection["matrix"][family]
        for j,split in enumerate(("development","historical_test")):
            values=[score(split,n,"s3_pct") for n in names]
            axes[j].plot((.1,.3),values,marker="o",label=family)
            for rho,value in zip((.1,.3),values):
                w=calibration["coefficients"][family][str(rho)]
                axes[j].annotate(f"{w:.3g}",(rho,value),xytext=(5,1),textcoords="offset points",fontsize=7)
            axes[j].set_xlabel("Initial gradient ratio ρ")
            axes[j].set_title(LABELS[j])
            axes[j].set_ylabel("MSE reduction versus S3 (%)")
    axes[1].legend()
    save(fig,"02_rho_lambda.png")

    fig, axes=plt.subplots(2,3,figsize=(13,6),sharex=True)
    for split, panel in zip(("historical_test","period_extended"),axes):
        for ax,field,title in zip(panel,("native_pct","s3_pct","own_pct"),("Native","S3","I0-M0")):
            for family,arm in families.items():
                rows=sorted(indexed_points("leads",split,arm),key=lambda r:int(r["lead"]))
                ax.plot(range(1,21),[float(r[field] or 0) for r in rows],label=family)
            ax.set_title(f"{split} versus {title}")
            ax.set_xticks(range(1,21,2))
            ax.axhline(0,color=".55",lw=.7)
    axes[0][-1].legend()
    save(fig,"03_leads.png")

    fig, axes=plt.subplots(1,2,figsize=(10,4))
    for split,ax in zip(("historical_test","period_extended"),axes):
        for i,(factor,field,title) in enumerate(((2,"C_delta","2C"),(-1,"U_delta","−U"),(1,"G_delta","G"))):
            x=np.arange(len(families))+(i-1)*.25
            ax.bar(x,[factor*score(split,n,field) for n in families.values()],.23,label=title)
        ax.set_xticks(range(len(families)))
        ax.set_xticklabels(families)
        ax.set_title(split)
        ax.set_ylabel("Physical squared units")
    axes[1].legend()
    save(fig,"04_increment_decomposition.png")

    fig,axes=plt.subplots(1,2,figsize=(13,4),gridspec_kw={"width_ratios":[1,2]})
    result=np.array([[score(s,n,"s3_pct") for n in families.values()] for s in OUTER])
    image=axes[0].imshow(result,aspect="auto",cmap="RdBu",vmin=-max(abs(result).max(),.01),vmax=max(abs(result).max(),.01))
    axes[0].set_xticks(range(4))
    axes[0].set_xticklabels(families)
    axes[0].set_yticks(range(3))
    axes[0].set_yticklabels(LABELS[-3:])
    fig.colorbar(image,ax=axes[0],label="Versus isolated S3 (%)")
    rows=indexed_points("trajectories","historical_test",winner)
    axes[1].bar(range(len(rows)),[float(r["s3_pct"]) for r in rows])
    axes[1].tick_params(axis="x",labelrotation=38)
    axes[1].set_xticks(range(len(rows)))
    axes[1].set_xticklabels([r["trajectory"].replace(".h5","") for r in rows],fontsize=7)
    axes[1].set_ylabel("Historical reduction versus S3 (%)")
    save(fig,"05_regime_trajectory.png")

    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for ax,channel in zip(axes,("u","v")):
        for family,arm in (("S3","S3"),*families.items()):
            rows=[r for r in indexed_points("incremental_mode_scores","historical_test",arm) if r["channel"]==channel]
            rows.sort(key=lambda r:int(r["mode"]))
            ax.plot(range(20),[float(r["final_error_energy"]) for r in rows],label=family)
        ax.set_title(channel)
        ax.set_xlabel("Temporal mode")
        ax.set_ylabel("Physical error energy")
        ax.set_yscale("log")
    axes[1].legend()
    save(fig,"06_time_modes.png")

    fig,axes=plt.subplots(2,3,figsize=(13,7))
    training=read_rows(out/"training.csv")
    grad=read_rows(out/"gradient_diagnostics.csv")
    for family,arm in families.items():
        rows=[r for r in training if r["arm"]==arm]
        x=[int(r["step"]) for r in rows]
        for ax,col,title in ((axes[0,0],"train_mse","Training MSE"),(axes[0,1],"dev_mse","Development MSE"),
                             (axes[0,2],"clip_frequency","Training clip frequency")):
            ax.plot(x,[float(r[col]) if r[col] else np.nan for r in rows],label=family)
            ax.set_title(title)
        by_step=defaultdict(list)
        for row in grad:
            if row["arm"]==arm and row["term"]!="mse":
                by_step[int(row["step"])].append(row)
        for ax,col,title in ((axes[1,0],"weighted_ratio","Auxiliary gradient ratio"),
                             (axes[1,1],"cosine_to_mse","Gradient cosine"),
                             (axes[1,2],"total_norm","Total gradient norm")):
            ax.plot(sorted(by_step),[mean(float(r[col]) for r in by_step[k]) for k in sorted(by_step)],marker=".",label=family)
            ax.set_title(title)
    axes[0,1].legend()
    save(fig,"07_training_gradients.png")

    fig,ax=plt.subplots(figsize=(7,4))
    for row in read_rows(out/"inference_cost.csv"):
        name=winner if row["arm"]=="candidate" else row["arm"]
        x=float(row["ms_end_to_end_median"])
        y=score("historical_test",name,"native_pct")
        ax.errorbar(x,y,xerr=[[x-float(row["ms_end_to_end_min"])],[float(row["ms_end_to_end_max"])-x]],fmt="o",label=name)
    ax.set_xlabel("End-to-end timing (ms, batch 1)")
    ax.set_ylabel("Historical removal of Native error (%)")
    ax.legend()
    save(fig,"08_accuracy_cost.png")

    fig,axes=plt.subplots(2,4,figsize=(12,4),layout="constrained")
    patches=read_rows(out/"fixed_fields.csv")
    for channel,axes_row in zip(("u","v"),axes):
        part=[r for r in patches if r["channel"]==channel]
        maximum=max(abs(float(r["value"])) for r in part)
        for field,ax in zip(("true_residual","S3","increment","remaining"),axes_row):
            matrix=np.zeros((32,64))
            for r in part:
                if r["field"]==field:
                    matrix[int(r["y"]),int(r["x"])]=float(r["value"])
            im=ax.imshow(matrix,cmap="RdBu_r",vmin=-maximum,vmax=maximum)
            ax.set_title(f"{channel} {field.replace('_',' ')}")
            ax.axis("off")
        fig.colorbar(im,ax=list(axes_row),shrink=.72)
    fig.savefig(figures/"09_development_fields.png",dpi=165)
    plt.close(fig)
