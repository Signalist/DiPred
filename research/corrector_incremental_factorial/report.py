"""Render the Chinese decision record from signed numerical tables."""
from statistics import mean

from research.corrector_incremental_structure.validation import read_rows
from research.corrector_window_study.data import read_json

from .core import BASE, FACTORIAL
from .plot import DEV, HIST, REGIONS, RES


def percent(value):
    return f"{float(value):+.3f}%"


def describe(effect):
    return "支持" if float(effect["lower_pct"])>0 else (
        "部分支持" if float(effect["effect_pct"])>0 else "未支持")


def effect_status(effect):
    if float(effect["lower"]) > 0:
        return "支持"
    return "部分支持" if float(effect["mse_reduction"]) > 0 else "未支持"


def report(out, freeze):
    selection=freeze["selection"]
    candidate=selection["best_new"]
    records={(r["split"],r["policy"],r["arm"]):r for r in read_rows(out/"scores.csv")}
    comparisons={(r["split"],r["policy"],r["comparison"],r["arm"],r["control"]):r
                 for r in read_rows(out/"contrasts.csv")}
    effects=read_rows(out/"factorial_effects.csv")+read_rows(out/"factorial_interactions.csv")
    hist_effects={r["effect"]:r for r in effects if r["split"]==HIST and r["policy"]=="final"}
    assembly=read_json(out/"verification.json")
    cal=read_json(out/"initial_calibration.json")
    costs={r["arm"]:r for r in read_rows(out/"inference_cost.csv")}
    leads=read_rows(out/"leads.csv")
    regimes=read_rows(out/"regimes.csv")
    trajectories=read_rows(out/"trajectories.csv")
    intervention=read_rows(out/"history_interventions.csv")
    re_equal={(r["split"],r["policy"],r["arm"]):r for r in read_rows(out/"equal_regime_scores.csv")}
    risk=next(r for r in read_rows(out/"trajectory_risks.csv")
              if r["split"]==HIST and r["policy"]=="selected" and r["arm"]==candidate)
    p4=comparisons[HIST,"selected","P4",candidate,"S3"]
    deployment=candidate if selection["overall"]==candidate and float(p4["lower_pct"])>0 else "S3"

    def pair(a,b,label,split=HIST,policy="final"):
        return comparisons.get((split,policy,label,a,b)) or comparisons.get((split,policy,"own_mse",a,b))

    def interval(r):
        return f"{percent(r['effect_pct'])}，区间 [{percent(r['lower_pct'])}, {percent(r['upper_pct'])}]"

    def one(label, body):
        lines.extend(["", label, "", body])

    def change(a, b, field, split=HIST):
        return float(records[split,"final",a][field])-float(records[split,"final",b][field])

    def decomposition(base, split):
        aux = base + "_match2"
        return (2*change(aux, base, "C_delta", split), -change(aux, base, "U_delta", split),
                change(aux, base, "G_delta", split))

    def rank_gain(a, b, split):
        return 100*change(b, a, "mse", split)/float(records[split, "final", b]["mse"])

    def stress_series(a, b, label):
        rows = [pair(a, b, label, split) for split in REGIONS]
        return "、".join(f"Re={re} {interval(row)}（{row['improved']}/{row['trajectories']} 条轨迹改善）"
                        for re, row in zip(RES, rows))

    def conditional_lines(label, pairs):
        return "。".join(f"{a} 对 {b}，{describe(row)}历史优势，{interval(row)}"
                        for a, b in pairs for row in (pair(a, b, label),))

    lines=["# DiPred 增量纠错的全因子消融、机制归因与工况验证","",
           f"开发主规则选择 **{selection['overall']}**。新候选为 **{candidate}**，"
           f"开发选中第 {freeze['models'][candidate]['selected']['step']} 步。实际运行建议 **{deployment}**。",
           f"候选在历史 24 窗相对 S3 的改善为 {interval(p4)}，"
           f"{p4['improved']}/{p4['trajectories']} 条轨迹改善。",
           f"在同一历史队列及检查点口径下，候选相对 Native 改善 {percent(records[HIST,'selected',candidate]['native_pct'])}，"
           f"Re 等权相对 S3 改善 {percent(re_equal[HIST,'selected',candidate]['s3_pct'])}。",
           f"全因子固定 λ 为 {freeze['anchor']:.16g}。本轮新增 "
           f"{len([r for r in freeze['models'].values() if r['directory'].endswith('_factorial')])} 个主队列训练和 15 个工况训练，"
           "复用五个主队列全因子配置、三个工况各三个全因子配置及 20 个 S3 教师。历史完整状态的已知缺口见复用审计。",
           "", "## 八组全因子固定 6000 步","",
           "| 模型 | 开发物理 MSE | 历史物理 MSE | 历史相对 Native | 历史相对 S3 | 历史相对自身 MSE |",
           "| --- | --- | --- | --- | --- | --- |"]
    for name in FACTORIAL:
        d=records[DEV,"final",name]
        h=records[HIST,"final",name]
        own=percent(h["own_pct"]) if h["own_pct"] else "—"
        lines.append(f"| {name} | {float(d['mse']):.8g} | {float(h['mse']):.8g} | "
                     f"{percent(h['native_pct'])} | {percent(h['s3_pct'])} | {own} |")
    lines.extend(["", "## 冻结主干条件下的 Corrector 工况压力测试","",
                  "表内为相对各工况的隔离 S3 的物理 MSE 改善。",
                  "", "| 模型 | Re=17812 | Re=25482 | Re=8863 |", "| --- | --- | --- | --- |"])
    for name in FACTORIAL:
        lines.append("| "+name+" | "+" | ".join(percent(records[s,"final",name]["s3_pct"]) for s in REGIONS)+" |")

    lines.extend(["", "| Re | 模型 | 相对 Native | 相对自身 MSE | 相对无历史模型 | 相对 S3 改善轨迹 |",
                  "| --- | --- | --- | --- | --- | --- |"])
    for split, re in zip(REGIONS, RES):
        for name in FACTORIAL:
            row = records[split, "final", name]
            own = percent(row["own_pct"]) if row["own_pct"] else "—"
            no_history = name.replace("B", "0")
            history = percent(pair(name, no_history, "history", split)["effect_pct"]) if name != no_history else "—"
            gain = pair(name, "S3", "s3", split)
            lines.append(f"| {re} | {name} | {percent(row['native_pct'])} | {own} | {history} | "
                         f"{gain['improved']}/{gain['trajectories']} |")

    lines.extend(["", "## 主效应与交互","",
                  "下表为历史固定步轨迹等权物理 MSE 对比。正值代表误差下降或相应条件下误差下降增加。",
                  "", "| 项目 | MSE 下降量 | 95% 配对区间 | 相当于 S3 MSE |", "| --- | --- | --- | --- |"])
    for k,r in hist_effects.items():
        if r["kind"]=="factorial":
            lines.append(f"| {k} | {float(r['mse_reduction']):+.4g} | "
                         f"[{float(r['lower']):+.4g}, {float(r['upper']):+.4g}] | {percent(r['pct_of_s3'])} |")
    h=pair("IB_match2","I0_match2","P1")
    i=pair("IB_match2","DB_match2","P2")
    one("## 1. I0_match2 能否达到 IB_match2", f"{describe(h)} IB_match2 的额外优势。历史 {interval(h)}。"
        + stress_series("IB_match2", "I0_match2", "P1") + "。")
    one("## 2. 真实历史残差的独立价值",
        conditional_lines("history", [(a+"B"+t,a+"0"+t) for a in ("I","D") for t in ("","_match2")]) + "。")
    hi=hist_effects["history*match2|increment=1"]
    one("## 3. 历史与 TM2 的交互",
        f"{effect_status(hi)}增量条件下的正向交互。历史与 TM2 差分为 {float(hi['mse_reduction']):+.5g}，"
        f"区间 [{float(hi['lower']):+.5g}, {float(hi['upper']):+.5g}]。该量等于启用 TM2 与纯 MSE 时历史收益之差。")
    one("## 4. 增量与直接预测", f"{describe(i)} 含历史且启用 TM2 时的增量优势。历史 {interval(i)}。"
        "参数化对照同时改变输出锚点、监督目标和初始预测。" +
        conditional_lines("increment", [("I"+s+t,"D"+s+t) for s in ("0","B") for t in ("","_match2")]) + "。" +
        stress_series("IB_match2", "DB_match2", "P2") + "。")
    for n,k,title in ((5,"increment*match2","TM2 对增量结构的依赖"),
                      (6,"history*match2","TM2 对历史真实残差的依赖")):
        e=hist_effects[k]
        one(f"## {n}. {title}",f"{effect_status(e)}正向交互。物理误差交互 {float(e['mse_reduction']):+.5g}，"
            f"区间 [{float(e['lower']):+.5g}, {float(e['upper']):+.5g}]。")
    lines.extend(["", conditional_lines("P3", [(a+"_match2", a) for a in BASE]) + "。"])
    strongest=max((hist_effects[k] for k in ("increment","history","match2")),key=lambda r:abs(float(r["mse_reduction"])))
    one("## 7. 最大主效应",f"历史固定步中，按 MSE 下降量绝对值排序的首项为 {strongest['effect']}，"
        f"符号和数值为 {float(strongest['mse_reduction']):+.5g}。")
    calibration=[]
    for a in BASE:
        other=a+"_match2_rho"
        if (HIST,"final",other) in records:
            fixed = records[HIST, "final", a+"_match2"]
            matched = records[HIST, "final", other]
            sign_changed = (float(fixed["own_pct"]) > 0) != (float(matched["own_pct"]) > 0)
            calibration.append(f"{a} 的固定 λ 与等初始梯度强度收益分别为 {percent(fixed['own_pct'])}、"
                               f"{percent(matched['own_pct'])}（{'符号改变' if sign_changed else '符号一致'}）")
    one("## 8. 初始梯度强度匹配", "固定 λ 的初始比例为 " +
        "、".join(f"{k} {v:.4f}" for k,v in cal["actual_ratios"].items())+"。"+
        ("校准臂单列。"+ "、".join(calibration)+"。" if calibration else "未触发比率校准补训。"))
    time_orders = []
    for a in ("I0", "IB"):
        if (HIST, "final", a+"_match") not in records:
            time_orders.append(f"尚未验证 {a} 的匹配阶数对照，该路线未达到开发扩展门槛。")
            continue
        for control, label in (("_match", "TM1"), ("_energy", "EN")):
            if (HIST, "final", a+control) in records:
                values = [percent(rank_gain(a+"_match2", a+control, split))
                          for split in (DEV, HIST, "period", "period_new")]
                time_orders.append(f"固定步 {a} 的 TM2 相对 {label} 在开发、历史 24 窗、连续 96 窗和较晚 48 窗的 MSE 改善依次为 "
                                   + "、".join(values) + "。")
    one("## 9. 一阶、二阶与能量惩罚", "".join(time_orders))
    detail = ["| 模型 | 2ΔC_delta | −ΔU_delta | ΔG_delta |", "| --- | --- | --- | --- |"]
    for base in BASE:
        alignment, energy, net = decomposition(base, HIST)
        detail.append(f"| {base} | {alignment:+.4g} | {energy:+.4g} | {net:+.4g} |")
    one("## 10. 修正能量的作用", "表内是各模型启用 TM2 后的历史固定步变化。"
        "−ΔU_delta 为正时，额外修正能量下降。ΔG_delta 直接对应物理 MSE 的下降。\n\n"+"\n".join(detail))
    detail = ["| Re | 模型 | 2ΔC_delta | −ΔU_delta | ΔG_delta |", "| --- | --- | --- | --- | --- |"]
    for re, split in zip(RES, REGIONS):
        for base in BASE:
            alignment, energy, net = decomposition(base, split)
            detail.append(f"| {re} | {base} | {alignment:+.4g} | {energy:+.4g} | {net:+.4g} |")
    improved = [base for base in BASE if decomposition(base,HIST)[0]>0 and decomposition(base,HIST)[2]>0]
    one("## 11. 有效增量内积", ("部分支持对齐项增强并降低 MSE，历史固定步符合该模式的结构为 " + "、".join(improved)+"。"
        if improved else "未支持在历史固定步同时提高有效对齐和减少 MSE。") + "各隔离工况的变化如下。\n\n"+"\n".join(detail))
    diagnostics = ("zero", "same_re", "reverse", "retain_last_5", "retain_last_10", "mask_last_5", "mask_last_10", "spatial")
    sample_history_gain = {}
    input_table = ["| 队列 | 模型 | 置零 | 同 Re 置换 | 反转 | 仅末 5 帧 | 仅末 10 帧 | 屏蔽末 5 帧 | 屏蔽末 10 帧 | 空间置换 |",
                   "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for split, label in ((DEV, "开发"), (HIST, "历史")):
        for a in ("IB", "IB_match2"):
            rows = {k:mean(float(r["mse"]) for r in intervention if r["split"]==split and r["arm"]==a and r["intervention"]==k)
                    for k in ("original", *diagnostics)}
            sample_history_gain[split, a] = 100*(1-rows["original"]/rows["same_re"])
            values = " | ".join(percent(100*(1-rows[k]/rows["original"])) for k in diagnostics)
            input_table.append(f"| {label} | {a} | {values} |")
    stronger_history = all(sample_history_gain[s,"IB_match2"]>max(0.,sample_history_gain[s,"IB"])
                           for s in (DEV,HIST))
    matched_history = ("部分支持 IB_match2 更有效利用对应样本的历史残差。恢复原输入相对同 Re 置换的收益在两个队列均为正，且大于 IB。"
                       if stronger_history else "同 Re 置换结果未支持 IB_match2 在两个队列都比 IB 更有效地利用对应样本的历史残差。")
    one("## 12. 实际利用了哪些历史输入", matched_history +
        "表内是每项干预相对原输入的 MSE 改善。置换和屏蔽可能改变输入分布，重新训练的历史通道消融见第 2 节。"
        "单帧屏蔽的输出响应见图 10。\n\n"+"\n".join(input_table))
    one("## 13. 工况复现","IB_match2 相对各自 S3 为 " + stress_series("IB_match2", "S3", "s3") + "。"
        "冻结 DiPred 曾训练过这些留出轨迹。")
    one("## 14. 上边缘与 Native",f"Re={RES[1]} 的 IB_match2 相对 Native 为 "
        f"{percent(records[REGIONS[1],'final','IB_match2']['native_pct'])}。同工况各配置见原表。")
    negative=[]
    for title,rows,key in (("轨迹",trajectories,"trajectory"),("Re",regimes,"re"),("lead",leads,"lead")):
        bad=[r[key] for r in rows if r["arm"]==candidate and r["split"]==HIST and r["policy"]=="selected" and float(r["s3_pct"])<0]
        negative.append(title+" "+("、".join(bad) or "无"))
    worst = min((r for r in regimes if r["arm"]==candidate and r["split"]==HIST and r["policy"]=="selected"),
                key=lambda r:float(r["s3_pct"]))
    one("## 15. 退步与集中度","历史开发预选模型相对 S3 退步的 "+ "，".join(negative)+"。"+
        f"相对 S3 的改善最低的是 Re={worst['re']}（{percent(worst['s3_pct'])}）。"+
        (f"正收益最大的两条轨迹占比为 {percent(100*float(risk['top_two_share_of_positive_gain']))}。"
         if risk["top_two_share_of_positive_gain"] else "没有历史正收益轨迹。"))
    one("## 16. 最佳模型与 S3",f"{describe(p4)}历史改善。{interval(p4)}。"
        f"连续 96 窗和较晚 48 窗的固定步改善分别为 {percent(records['period','final',candidate]['s3_pct'])}、"
        f"{percent(records['period_new','final',candidate]['s3_pct'])}。两者为历史评价队列。")
    simplified = pair("IB_match2", "I0_match2", "P1")
    if float(simplified["lower_pct"])>0:
        simplification = "历史固定步比较支持保留 TM2 模型中的真实历史残差。"
    else:
        simplification = "历史固定步比较尚不足以确认移除历史残差后的效果，工况配对结果见第 1 节。"
    one("## 17. 简化与成本", f"默认运行建议为 {deployment}。{simplification}"
        f"候选的总参数 {int(costs[candidate]['total_parameters']):,}，新增可训练参数 {int(costs[candidate]['trainable_parameters']):,}。"
        f"`I0_match2` 的新增可训练参数为 {int(costs['I0_match2']['trainable_parameters']):,}。"
        f"候选和 S3 单窗完整延迟分别为 {float(costs[candidate]['ms_end_to_end']):.2f} ms、"
        f"{float(costs['S3']['ms_end_to_end']):.2f} ms，含 Native 四次去噪及实际 S3 调用。")
    one("## 18. 证据范围",
        f"{effect_status(hist_effects['increment'])}增量参数化在历史固定步的主效应。"
        f"{effect_status(hi)}增量条件下历史残差与 TM2 的正向交互。"
        "这些对比描述本训练协议下的设计差异。配对 bootstrap 以轨迹为单位抽样 10,000 次。"
        "单种子、开发选模和复用历史评价限制确认性。"+
        ("独立确认数据不足，无法判断独立迁移。" if not assembly["independent"]["verified"] else "独立队列身份见验证记录。"))
    one("## 19. 下一步", f"以 {candidate} 为研究候选、保留 S3 作为基准，用独立轨迹复验 P1–P4。"
        "同步记录不同工况的增量对齐与能量变化，并保持共同 λ 和对应纯 MSE 对照。")
    controls = [a for a in ("IB_diag","IB_perm") if (HIST,"final",a) in records]
    structure = [f"正常 TM2 相对 {a} 的开发和历史固定步改善分别为 "
                 + "、".join(percent(rank_gain("IB_match2",a,s)) for s in (DEV,HIST)) for a in controls]
    if controls and all(rank_gain("IB_match2",a,s)>0 for a in controls for s in (DEV,HIST)):
        adjacency = "部分支持正常时间目标在当前开发和历史比较中的优势。物理邻接需要独立数据复验。"
    elif controls:
        adjacency = "未支持正常时间邻接持续领先。将 TM2 作为结构化误差加权或正则化研究。"
    else:
        adjacency = "正常时间邻接的控制实验尚未执行。"
    one("## 20. 降低优先级的方向", "。".join(structure)+("。" if structure else "")+adjacency+
        "保持 IE 及自身反馈为历史参照。扩展的训练依据和停止点见记录。")
    one("## 历史 24 窗逐 lead","| lead | 候选相对 Native | 候选相对 S3 |\n| --- | --- | --- |\n"+
        "\n".join(f"| {r['lead']} | {percent(r['native_pct'])} | {percent(r['s3_pct'])} |"
                  for r in sorted((r for r in leads if r["split"]==HIST and r["policy"]=="selected" and r["arm"]==candidate),
                                  key=lambda r:int(r["lead"]))))
    one("## 产物","[完整逐 lead 结果](leads.csv)覆盖各队列逐帧物理 MSE 和纠错率。"
        "[配对对比](contrasts.csv)和[工况结果](holdout_results.csv)包含具体区间和误差。"
        "[图表](../../.User/corrector_incremental_factorial)以实测数据绘制。"
        "[复现命令](PROTOCOL.md)和[复用审计](reuse_audit.json)记录原始来源及上一轮 14 个校准实验的中间状态缺口。")
    (out/"DECISION_REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
