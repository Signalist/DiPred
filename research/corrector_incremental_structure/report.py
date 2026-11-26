"""Write the Chinese decision report using the frozen physical comparisons."""
from statistics import mean

from research.corrector_window_study.data import read_json

from .validation import read_rows


def number(v):
    return f"{float(v):.5g}"


def pct(v):
    return f"{float(v):+.3f}%"


def classify(values):
    positive=sum(x>0 for x in values)
    if positive==len(values):
        return "支持"
    if positive:
        return "部分支持"
    return "未支持"


def report(out,figures):
    freeze=read_json(out/"FROZEN.json")
    selection=freeze["selection"]
    plan=read_json(out/"experiment_plan.json")
    verify=read_json(out/"verification.json")
    model=selection["best_new"]
    final=selection["overall"]
    chosen=selection["best_MSE_input"]
    tables=read_rows(out/"scores.csv")
    scores={(r["split"],r["policy"],r["arm"]):r for r in tables}
    pairs=read_rows(out/"contrasts.csv")
    dev,history="development","historical_test"
    primary=[r for r in pairs if r["comparison"] in ("P1","P2","P3","P4") and r["policy"]=="selected"
             and r["split"] in (dev,history)]
    train=read_json(out/"teacher_train.json")["distribution"]
    costs={r["arm"]:r for r in read_rows(out/"inference_cost.csv")}
    lineage=read_json(out/"outer_lineage.json")["regimes"]
    q=read_json(out/"feedback_training_selection.json")
    step=freeze["models"][model]["selected"]["step"]
    interval={(r["split"],r["policy"],r["comparison"],r["arm"],r["control"]):r for r in pairs}

    def contrast(split,a,b,label,policy="selected"):
        return interval[split,policy,label,a,b]

    def subject(r):
        return f"{r['arm']} 对 {r['control']} {pct(r['effect_pct'])}"

    def improvement(split,a,b,policy="selected"):
        x=scores[split,policy,a]
        y=scores[split,policy,b]
        return 100*(1-float(x["mse"])/float(y["mse"]))

    def describe(split,arm,policy):
        s=scores[split,policy,arm]
        return (f"{arm} MSE {number(s['mse'])}，相对 Native {pct(s['native_pct'])}，"
                f"相对 S3 {pct(s['s3_pct'])}" + (f"，相对自身对照 {pct(s['own_pct'])}" if s["own_pct"] else ""))

    p1=contrast(history,model,"S3","P1")
    p1dev=contrast(dev,model,"S3","P1")
    p4=[r for r in primary if r["comparison"]=="P4" and r["split"]==dev]
    p4h=[r for r in primary if r["comparison"]=="P4" and r["split"]==history]
    p2=[r for r in primary if r["comparison"]=="P2" and r["split"]==history]
    p3=[r for r in primary if r["comparison"]=="P3" and r["split"]==history]
    cost=costs[final]
    core=freeze["counts"]["core"]
    outer=sum(len(r["candidates"]) for r in lineage.values())
    calibration_archive=read_json(out/"calibrated_state_archive.json")
    retained=sum(len(r["retained"]) for r in calibration_archive.values())
    unavailable=sum(len(r["unavailable"]) for r in calibration_archive.values())
    out_text=[
        "# DiPred 残差增量、时间结构和反馈一致性",
        "",
        f"按登记的开发集主规则选定 **{final}**。新候选 **{model}** 使用第 {step} 步权重。"
        f"开发主规则所选配置共有 {int(cost['total_parameters']):,} 个参数，"
        f"本轮头部有 {int(cost['trainable_parameters']):,} 个可训练参数。"
        f"单窗端到端中位延迟为 {float(cost['ms_end_to_end']):.2f} ms，"
        "计入当前发布中实际调用的四次 DiPred 去噪和 S3，已经发布的历史状态从缓存读取。",
        "",
        f"开发所选端点为 {describe(dev,model,'selected')}。历史 24 窗上为 {describe(history,model,'selected')}。"
        f"历史队列相对 S3 的配对区间为 [{pct(p1['lower_pct'])}, {pct(p1['upper_pct'])}]，"
        f"{p1['improved']}/{p1['trajectories']} 条轨迹改善。固定第 6000 步时，"
        f"{describe(history,model,'final')}。",
        "",
        f"登记的 {core} 个核心配置和 {outer} 个工况复验均完成。复用了主训练的四个 OOF 教师、"
        "一套完整训练 S3、三个工况各五个隔离 S3、一套直接 E 的第 1000 步全状态、"
        "对应 K=20 基线以及已发布的单路径 Native 缓存。历史 B/E 与时间目标保留原权重和选法。"
        "各配置的实际更新量、监督窗口暴露量与耗时见训练记录。结构模型归档各登记步的全状态，"
        f"直接 E 校准臂保留共有的第 1000 步全状态、全套权重检查点及 {retained} 个分支全状态。"
        f"原训练器轮替临时日志后，仍有 {unavailable} 个分支全状态未归档，清单见 `calibrated_state_archive.json`。",
        "",
        "## 主要比较",
        "",
        "| 队列 | 比较 | 候选对照 | 改善 | 95% 配对区间 | 改善轨迹 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for r in primary:
        out_text.append(f"| {r['split']} | {r['comparison']} | {r['arm']} / {r['control']} | "
                        f"{pct(r['effect_pct'])} | [{pct(r['lower_pct'])}, {pct(r['upper_pct'])}] | "
                        f"{r['improved']}/{r['trajectories']} |")
    out_text.extend([
        "",
        f"主比较每个完整队列有 {sum(len(x) for x in selection['primary'].values())} 项。"
        f"原始对比表共 {len(pairs)} 行，涵盖不同队列、两套权重口径及次要比较。"
        f"按原轨迹配对 bootstrap {plan['bootstrap']:,} 次。区间用于描述，未进行显著性检验。"
        "经开发选模后的区间以及已复用的历史评价均无独立确认含义。",
        "",
        "## 1. 相近初始梯度影响下的时间结构",
        "",
        f"开发结果{classify([float(r['effect_pct']) for r in p4])}正常 TM2。四项分别为"
        f" {'、'.join(map(subject,p4))}。历史对应比较为 {'、'.join(map(subject,p4h))}。"
        "匹配的约束是起点处八批平均梯度范数，随训练变化的方向、裁剪、逐 lead 权重与路径仍可不同。"
        "初始范数匹配及所有后续探针见梯度表。",
        "",
        "## 2. 历史信息对 S3 剩余误差的作用",
        "",
        f"历史输入的增益为{'、'.join(map(subject,p2))}，整体方向为{classify([float(r['effect_pct']) for r in p2])}。"
        f"{describe(history,'IB','selected')}。{describe(history,'IE','selected')}。"
        f"OOF 当前输出的物理 RMS 为 {number(train['used']['prediction']['rms'])}，完整训练教师为 "
        f"{number(train['full']['prediction']['rms'])}。两者的剩余误差 RMS 分别为 "
        f"{number(train['used']['remaining']['rms'])} 和 {number(train['full']['remaining']['rms'])}。"
        "冻结 S3 当前输出由已观测输入和 Native 决定，属于已有输入的表征。",
        "",
        "## 3. 增量与同输入直接预测",
        "",
        f"历史队列中，逐组的点估计为 {'、'.join(map(subject,p3))}。"
        f"三项配对区间有 {sum(float(r['lower_pct'])<=0<=float(r['upper_pct']) for r in p3)} 项包含零。"
        "三对 CNN 的初始权重和批次相同。D 的零输出从 Native 起步，I 的零增量从 S3 起步。",
        "",
        "## 4. 增量机制",
        "",
    ])
    best=scores[history,"selected",model]
    ct,ut,gt=(float(best[k]) for k in ("C_delta","U_delta","G_delta"))
    global_cos=best["global_delta_cosine"]
    out_text.append(
        f"{model} 的历史增量分解为 Cδ {number(ct)}、Uδ {number(ut)}、Gδ {number(gt)}。"
        f"对齐项 2Cδ {'超过' if 2*ct>ut else '低于或等于'}新增输出能量 Uδ。"
        f"逐窗余弦均值为 {number(best['delta_cosine'])}，"
        f"轨迹加权整场余弦为 {number(global_cos) if global_cos else '未定义'}。"
        f"总修正分解 C、U、G 依次为 {number(best['C'])}、{number(best['correction_energy'])}、{number(best['G'])}。"
        "这两套分解衡量已发布修正的对齐与能量，对形成该修正的学习机制仍需结合配对对照。"
    )
    own=next((r for r in pairs if r["split"]==history and r["policy"]=="selected" and r["arm"]==model
              and r["comparison"]=="S1_own"),None)
    if own:
        ref=scores[history,"selected",own["control"]]
        shift_c=2*(ct-float(ref["C_delta"]))
        shift_u=float(ref["U_delta"])-ut
        out_text.append(f"相对 {own['control']}，变化来自对齐的部分为 {number(shift_c)}，"
                        f"来自新增修正能量变化的部分为 {number(shift_u)}。")
    out_text.extend(["","## 5. 低维表示和快速时间变化",""])
    ranks=selection["representation"]
    train_fits={r["arm"]:r for r in read_rows(out/"training.csv") if r["step"]=="6000"}
    mode_source=read_rows(out/"incremental_mode_scores.csv")
    rank_results=[]
    for arm in ranks:
        anchor=arm.split("_")[0]
        rank=int(arm.split("_K")[1])
        d=improvement(dev,arm,anchor,"final")
        h=improvement(history,arm,anchor,"final")
        rank_results.append(d)
        tr=100*(1-float(train_fits[arm]["train_mse"])/float(train_fits[anchor]["train_mse"]))
        old,new=(sum(float(r["final_error_energy"]) for r in mode_source if r["split"]==dev
                     and r["policy"]=="final" and r["arm"]==m and int(r["mode"])>=rank)
                 for m in (anchor,arm))
        out_text.append(f"- {arm} 相对 K=20 的训练、开发、历史 MSE 改善分别为 {pct(tr)}、{pct(d)}、{pct(h)}。"
                        f"K 以上模式的开发误差能量和由 {number(old)} 变为 {number(new)}。")
    out_text.append(f"开发固定步的比较结果{classify(rank_results)}低维限制。"
                    "高变化模态的标签能量与实际预测误差列在模式表中。标签投影诊断只给出固定子空间的表示误差，增量模型的基座保留 S3 全频段输出。")
    out_text.extend(["","## 6. 相同 lead 与物理时间邻接",""])
    rel=read_rows(out/"history_relationships.csv")
    for label in ("native_residual","s3_remaining"):
        by=[r for r in rel if r["target"]==label]
        same=mean(abs(float(r["pearson"])) for r in by if r["lead"]==r["source"])
        adjacent=next(float(r["pearson"]) for r in by if r["lag"]=="1")
        out_text.append(f"- {label} 的同 lead 平均绝对 Pearson 为 {number(same)}，物理间隔 1 帧的单对 Pearson 为 {number(adjacent)}。")
    observed=read_rows(out/"history_interventions.csv")
    for arm in ("IB","IB_lag","IB_dual"):
        rows=[r for r in observed if r["model"]==arm and r["policy"]=="selected"]
        same=mean(float(r["response_rms"]) for r in rows if r["source"]==r["lead"])
        adjacent=next(float(r["response_rms"]) for r in rows if r["physical_lag"]=="1")
        out_text.append(f"- {arm} 的逐帧置零干预中，同 lead 的平均响应 RMS 为 {number(same)}，"
                        f"物理间隔 1 帧的响应 RMS 为 {number(adjacent)}。")
    out_text.append(f"开发集上 Last、Reverse 和 Lag 相对 Original 的历史残差组织效果依次为 "
                    f"{'、'.join(pct(improvement(dev,'IB_'+a,'IB')) for a in ('last','reverse','lag'))}。"
                    "相关性矩阵、原始输入逐帧置零干预和配对 MSE 衡量不同问题。")
    out_text.extend(["","## 7. Dual 条件对照",""])
    dual=[improvement(dev,"IB_dual","IB_"+arm) for arm in ("lag","last","reverse","free")]
    out_text.append(f"开发结果{classify(dual)} Dual 优势。相对 Lag、Last、Reverse、Free 的改善依次为 "
                    f"{'、'.join(map(pct,dual))}。物理 lag 由 i、j 导出，结果限定于 20 帧窗口和当前发布步长。")
    out_text.extend(["","## 8. 自身反馈训练和运行",""])
    interaction_rows={r["objective"]:r for r in read_rows(out/"feedback_interactions.csv") if r["split"]=="period"}
    published=read_rows(out/"feedback_rollout.csv")
    for key,parent in (("MSE","IE"),("aux",q["auxiliary"])):
        a,b=f"Q_{key}_self",f"Q_{key}_ref"
        arms=(b,b+"_self_run",a,a+"_self_run")
        error=[number(scores["period","final",name]["mse"]) for name in arms]
        self_gain=improvement("period",a+"_self_run",b+"_self_run","final")
        contrast_estimate=interaction_rows[key]
        interaction=float(contrast_estimate["interaction"])
        drift=[mean(float(r["state_rms_from_fixed_s3"]) for r in published
                    if r["arm"]==arm and r["run_mode"]=="self" and r["prior_source"]=="published")
               for arm in (b,a)]
        out_text.append(f"- {key} 使用 {parent} 配置。按参考训练固定运行、参考训练自身运行、"
                        f"自身训练固定运行、自身训练自身运行的顺序，物理 MSE 为 {'、'.join(error)}。"
                        f"自身运行中改用自身训练的 MSE 改善为 {pct(self_gain)}，"
                        f"训练与运行模式的差异差为 {number(interaction)}，"
                        f"配对区间为 [{number(contrast_estimate['lower'])}, {number(contrast_estimate['upper'])}]。"
                        f"联结窗口的历史反馈状态相对固定 S3 的平均 RMS 为 {number(drift[0])} 和 {number(drift[1])}，"
                        "依次对应参考训练与自身训练。"
                        + ("交互区间包含零，失配缓解仍待确认。" if
                           float(contrast_estimate["lower"])<=0<=float(contrast_estimate["upper"]) else
                           f"该配对结果{classify([self_gain,interaction])}自身训练缓解运行失配。"))
    out_text.append(f"每条训练链监督三个窗口，反馈张量 detach。每个序列分支实际训练 "
                    "5,000 次更新、15,000 次小批前向、5,000 次反向和 120,000 个窗口暴露，另借用 1,000 步单窗前缀。"
                    "历史评价为 12 条轨迹，各连续 8 次发布，每次均有新观测。")
    family_controls={**{arm:arm.split("_")[0] for arm in ranks},
                     **{f"IB_{k}":"IB" for k in ("last","reverse","lag","dual","free")}}
    consistent=[arm for arm,baseline in family_controls.items()
                if improvement(dev,arm,baseline)>0 and improvement(history,arm,baseline)>0]
    stop_candidates=[arm for arm,baseline in family_controls.items()
                     if improvement(dev,arm,baseline)<=0 and improvement(history,arm,baseline)<=0]
    continued=(
        f"优先用独立轨迹确认 {model} 相对 S3 的效果。" if float(p1["lower_pct"])>0 else
        f"{model} 保留为待复验候选，现有历史区间尚不足以支持替换 S3。" if float(p1["effect_pct"])>0 else
        f"暂停推广 {model}，历史队列相对 S3 的物理 MSE 改善未转正。"
    )
    out_text.extend([
        "",
        "## 9. 开发候选是否超过冻结 S3",
        "",
        f"开发和历史相对 S3 的改善点估计分别为 {pct(p1dev['effect_pct'])} 和 {pct(p1['effect_pct'])}。"
        f"{continued}第 0 步增量为零，选择第 0 步时记为保持 S3。",
        "",
        "## 10. 工况与时间迁移",
        "",
        f"| 留出 Re | 超过该工况 S3 的配置 | 最低 MSE 配置 | 最低 MSE 配置相对 S3 | {model} 相对 S3 |",
        "| --- | --- | --- | --- | --- |",
    ])
    for regime,entry in lineage.items():
        split="holdout_"+regime
        options=[scores[split,"final",a] for a in selection["outer"]]
        top=min(options,key=lambda r:float(r["mse"]))
        good=[r["arm"] for r in options if float(r["s3_pct"])>0]
        chosen=scores.get((split,"final",model))
        result=pct(chosen["s3_pct"]) if chosen else "未列入冻结复验配置"
        out_text.append(f"| {entry['phys_re']} | {', '.join(good) or '无'} | {top['arm']} | {pct(top['s3_pct'])} | {result} |")
    lead_rows=[r for r in read_rows(out/"leads.csv") if r["arm"]==model and r["split"]==history and r["policy"]=="selected"]
    gain=[float(r["s3_pct"]) for r in lead_rows]
    tracks=[float(r["s3_pct"]) for r in read_rows(out/"regimes.csv")
            if r["arm"]==model and r["split"]==history and r["policy"]=="selected"]
    risk=next(r for r in read_rows(out/"trajectory_risks.csv")
              if r["arm"]==model and r["split"]==history and r["policy"]=="selected")
    share=risk["top_two_share_of_positive_gain"]
    concentration=(f"正收益最大的两条轨迹贡献了总正收益的 {pct(100*float(share))}。" if share else "历史轨迹的增量收益均未转正。")
    out_text.extend([
        "",
        f"历史 24 窗上有 {sum(v>0 for v in gain)}/20 个 lead 的聚合点估计优于 S3，"
        f"范围为 [{pct(min(gain))}, {pct(max(gain))}]，最差历史 Re 为 {pct(min(tracks))}。"
        f"{concentration}"
        f"预选候选在 96 窗历史连发队列的固定步改善为 {pct(scores['period','final',model]['s3_pct'])}，"
        f"较晚的 {verify['period']['later_windows']} 窗为 {pct(scores['period_new','final',model]['s3_pct'])}。"
        "留出组对 Corrector 与其教师隔离，冻结 DiPred 的原训练已接触这些轨迹。"
        "各 Re 与逐轨迹收益记录在原表。",
        "",
        "## 11. 机制证据的边界",
        "",
        "共享起点和配对批次控制了初始化与取样差异。等范数校准只控制起点处的平均幅度，"
        "后续梯度方向与裁剪可以分离。EN 的惩罚对象为新增修正，历史直接 E 的 EN 惩罚总修正。"
        "模式投影检验固定表示限制。时序映射含不同参数量，位置表达范围及有限样本所得的优化差异仍可能影响结果。",
        "",
        "## 12. 路线取舍",
        "",
        "后续机制确认应保持同输入 D 和初始等梯度对照。",
        "",
        f"按开发选定检查点，两队列相对各自对照均改善的投影或时间混合配置为 {', '.join(consistent) or '无'}。"
        f"两队列的点估计均无改善时，降低配置的复验优先级。涉及 {', '.join(stop_candidates) or '无'}。"
        "第 5 节另外给出了投影模型的固定步对照。",
        "",
        "## 13. 论文主张与证据等级",
        "",
        "本轮能报告单训练种子、固定主干、固定采样路径下的配对机制与工程结果。"
        "独立确认清单为空，跨初始化稳健性、未见主干工况泛化和长程运行稳定性均数据不足。"
        "开发选模后的区间仅作描述。历史队列的正向点估计不能替代独立复现。",
        "",
        "## 14. 更简单的选择",
        "",
        f"S3 的端到端中位延迟为 {float(costs['S3']['ms_end_to_end']):.2f} ms，"
        f"开发主规则所选 {final} 的端到端中位延迟为 {float(cost['ms_end_to_end']):.2f} ms。"
        f"历史 {describe(history,'S3','selected')}。"
        + ("在独立复验前保留 S3 作为运行默认项。" if float(p1["lower_pct"])<=0 else
           "S3 继续作为后续独立复验的基线。"),
        "",
        "## 产物",
        "",
        "- `scores.csv`、`contrasts.csv`、`lead_primary_contrasts.csv`、`leads.csv`、`channels.csv`、"
        "`trajectories.csv` 和 `regimes.csv` 提供物理指标、轨迹覆盖和逐帧结果。",
        "- `matched_gradients.csv`、`history_interventions.csv`、`temporal_mode_scores.csv`、"
        "`incremental_mode_scores.csv`、`oracle_modes.csv` 和 `time_maps.csv` 记录机制。",
        "- `feedback_rollout.csv`、`feedback_interactions.csv` 和 `lead_windows.csv.gz` 提供反馈状态、"
        "交叉实验的配对区间与逐窗逐帧误差。",
        "- `FROZEN.json`、`outer_lineage.json`、`period_provenance.json`、`verification.json` 与 "
        "`artifact_audit.json` 绑定配置、缓存、检查点和评价。",
        f"- 图表存放于 `{figures.name}`，复现命令见 `PROTOCOL.md`。",
        "",
    ])
    (out/"DECISION_REPORT.md").write_text("\n".join(out_text),encoding="utf-8")
