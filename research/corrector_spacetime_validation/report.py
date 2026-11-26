"""Generate the Chinese mechanism decision report without reading raw targets."""
from collections import Counter
from os.path import relpath
from statistics import mean

from research.corrector.run import sha
from research.corrector_window_study.data import read_json

from .plot import read


def report(out,figures):
    f = read_json(out/"FROZEN.json")
    v = read_json(out/"verification.json")
    assert v["passed"] and v["freeze_sha256"]==sha(out/"FROZEN.json")
    choices = f["selection"]["choices"]
    all_scores = read(out/"scores.csv")
    lookup = {(r["split"],r["policy"],r["arm"]):r for r in all_scores}
    contrasts = read(out/"contrasts.csv")
    sensitivity = read(out/"selection_sensitivity.csv")
    histories = read(out/"historical_review.csv")
    energy = read(out/"energy_matching.csv")
    modes = read(out/"temporal_mode_scores.csv")
    main_splits = ("development","historical_test")
    e,b = choices["E"],choices["B"]
    winner = f["selection"]["global_rule_A"]["selected"]

    def number(split,model,key="mse",policy="selected"):
        value = lookup[split,policy,model][key]
        return float(value) if value else None

    def effect(split,a,b,policy="selected"):
        return 100*(1-number(split,a,policy=policy)/number(split,b,policy=policy))

    def pct(value):
        if value is None:
            return "—"
        return "≈0.000%" if abs(value)<.0005 else f"{value:+.3f}%"

    def support(values,limited=False):
        if all(x>0 for x in values) and not limited:
            return "支持（限所列队列）"
        if any(x>0 for x in values):
            return "部分支持"
        return "未支持"

    def pair(split,a,c,policy="selected"):
        return next(x for x in contrasts if (x["split"],x["policy"],x["arm"],x["control"])==(split,policy,a,c))

    def link(name):
        return f"[{name}]({name})"

    lines = ["# DiPred 时空残差约束机制与泛化验证",""]
    add = lines.append
    new_outer = len(list((out/"fits").glob("outer_*.json")))
    add(f"本轮新增 **{f['core_new_fits']} 组核心及能量对照训练**、**{new_outer} 组工况留出训练**。"
        f"Rule-A 按开发集冻结的全局候选为 **{winner}**。"
        f"主要机制候选为 **{e['TM2']}**。全部验证以冻结记录 `{v['freeze_sha256']}` 为准。")
    add("")
    add("## 数学含义")
    add("")
    add("最终预测与真值之差除以公共残差尺度 s_R 即 e，其中 r = R̂ / s_R，y = R / s_R，e = r − y。"
        "线性差分算子满足 D₁r − D₁y = D₁e 和 D₂r − D₂y = D₂e。普通二阶平滑惩罚预测曲率 D₂r，"
        "真实二阶匹配惩罚误差曲率 D₂e。")
    add("")
    add("`L = eᵀ A e`，其中 `A = I/20 + λ D₂ᵀD₂/18`。批次、通道和空间维度仍按原实现等权平均。"
        "给定输入 x，条件均值为 μ = E[y | x]，固定正定 A 下的条件风险为 "
        "`(r − μ)ᵀ A (r − μ) + tr(A Cov(y | x))`。"
        "第二项与预测 r 无关，因而无限表达能力下的最优解仍是条件均值。"
        "在有限模型中，二次型改变了误差权重、梯度及优化路径。")
    add("")
    add("输出能量训练目标 `mean((r − y)²) + β mean(r²)` 等于 "
        "`(1 + β) mean((r − y/(1 + β))²) + β mean(y²)/(1 + β)`。"
        "它对收缩真值重新训练，并改变目标和梯度尺度。发布修正强度固定为 1。"
        "E 指模型分支，上一窗剩余残差记为 Eₙ₋₁，当前修正能量记为 U。")
    add("")
    add("## 主要比较")
    add("")
    add("| 比较 | 6000 步开发 | 6000 步历史测试 | 历史测试描述性区间及改善轨迹 | 开发选中检查点的历史测试 | 判定 |")
    add("|---|---:|---:|---|---:|---|")
    for name,(a,c) in f["selection"]["primary"].items():
        early = effect("development",a,c,"final")
        held = effect("historical_test",a,c,"final")
        deployed = effect("historical_test",a,c)
        policy = "selected" if name=="P2_U_selected" else "final"
        ct=pair("historical_test",a,c,policy)
        lo,hi=(float(ct[k]) for k in ("lower_pct","upper_pct"))
        limited=lo<=0 or 2*int(ct["improved"])<=int(ct["trajectories"])
        if name.startswith("P2_U_"):
            policy = name.rsplit("_",1)[1]
            match = next(r for r in energy if r["branch"]=="E" and r["policy"]==policy and r["model"]==c)
            judgment = ("尚未验证（训练 U 未达 5% 容差）" if abs(float(match["relative_gap"]))>.05
                        else support((early,held,deployed),limited))
        else:
            judgment = support((early,held,deployed),limited)
        ci=f"[{pct(lo)}, {pct(hi)}]，{ct['improved']}/{ct['trajectories']} 条（{policy}）"
        add(f"| {name} · {a} vs {c} | {pct(early)} | {pct(held)} | {ci} | {pct(deployed)} | {judgment} |")
    add("")
    add("三项主要假设对应五个具名对照行，P2 的能量匹配按两个检查点口径分别保留。"
        "其余为次要比较。"
        f"配对表共 {len(contrasts)} 行，{len(set(r['comparison'] for r in contrasts))} 种比较标签，含各队列及检查点口径。"
        "区间由 10,000 次轨迹配对重抽样得到。开发选择和多个 lead 比较使区间只能作描述性解释。")
    add("")
    add("## 时间结构与幅度机制")
    add("")
    for br in ("B","E"):
        m = choices[br]["TM2"]
        for policy in ("selected","final"):
            used = choices[br][f"EN_train_U_{policy}"]
            row = next(r for r in energy if r["branch"]==br and r["policy"]==policy and r["model"]==used)
            gap = 100*float(row["relative_gap"])
            adjustment = "达到" if abs(gap)<=5 else "超出"
            add(f"- {br} 的 {policy} 口径采用 {used}。训练集 U 相对 {m} 偏差 {pct(gap)}，{adjustment} 5% 容差。"
                f"TM2 相对该 EN 的开发与历史测试 MSE 改善分别为 "
                f"{pct(effect('development',m,used,policy))}、{pct(effect('historical_test',m,used,policy))}。")
    anchor = "historical_E_ext_match2"
    for arm,hypothesis in (("E_DIAG","跨时刻交叉项的作用"),("E_PERM","正确时间邻接的作用")):
        gains = [effect(s,anchor,arm,"final") for s in main_splits]
        add(f"- {hypothesis}。正常 TM2 相对 {arm} 的固定步开发和历史测试 MSE 改善为 "
            f"{pct(gains[0])}、{pct(gains[1])}，{support(gains,limited=True)}。三者 λ 相同。"
            "初始和训练中的实际梯度尺度见梯度明细。")
    grad=read(out/"gradient_diagnostics.csv")
    actual=[]
    for identity in (anchor,"E_DIAG","E_PERM"):
        label=f["models"][identity]["name"]
        at=[r for r in grad if r["model"]==label and r["step"]=="1000" and r["other"]=="mse"]
        ratio=mean(float(r["weighted_norm"]) for r in at if r["term"]!="mse")/mean(
            float(r["gradient_norm"]) for r in at if r["term"]=="mse")
        actual.append(f"{identity} {ratio:.2%}")
    add("- 八批训练探针测得三种同 λ 目标在共同检查点的辅助/MSE 梯度范数比分别为 "
        +"、".join(actual)+"。")
    add("- 对角和置换对照的实际梯度远大于正常 TM2。结果支持优先使用当前正常目标，"
        "交叉项或真实时间邻接的独立贡献仍与梯度强度混杂。")

    add("")
    add("时间分解使用有效二阶差分矩阵的正交本征向量。前两模式为常数与线性趋势张成的零空间，"
        "三段模式在冻结前定义。能量比 (U/E[R^2]) 先聚合二阶矩；逐窗口 RMS 比的平均则先各自开方。"
        "全局内积 (C) 未中心化，Pearson 使用中心化协方差并按方差归一化。"
        "全局余弦由聚合后的 C、U 和真值能量计算，另单独报告逐窗口余弦均值。")
    add("")
    mode_scores = {(r["split"],r["policy"],r["arm"],r["channel"],int(r["mode"])):r for r in modes}
    for split in main_splits:
        m,q = e["TM2"],e["EN_train_U_final"]
        for channel in ("u","v"):
            by_metric={}
            for kind in ("squared_error","predicted"):
                changes = []
                for lo,hi in ((0,6),(6,13),(13,20)):
                    a,b0 = (mean(float(mode_scores[split,"final",n,channel,j][kind]) for j in range(lo,hi))
                            for n in (m,q))
                    changes.append(pct(100*(1-a/b0)))
                by_metric[kind]="、".join(changes)
            add(f"- {split} · {channel} 通道，E-TM2 相对等能量 EN 的低、中、高模式误差改善为 "
                f"{by_metric['squared_error']}。对应预测能量降幅为 {by_metric['predicted']}。")
    region_index={(r["split"],r["policy"],r["arm"],r["channel"],r["region"]):r
                  for r in read(out/"spatial_region_scores.csv")}
    channel_index={(r["split"],r["policy"],r["arm"],r["channel"]):r for r in read(out/"channels.csv")}
    for split in main_splits:
        for channel in ("u","v"):
            a=e["TM2"]
            own=channel_index[split,"selected",a,channel]
            base=channel_index[split,"selected","E_M0",channel]
            tails=[region_index[split,"selected",k,channel,"tail"] for k in (a,"E_M0")]
            local_gain=float(tails[1]["mse_contribution"])-float(tails[0]["mse_contribution"])
            total=float(base["mse"])-float(own["mse"])
            add(f"- {split} · {channel} 通道，E-TM2 相对 E-M0 的物理 MSE 改善 {pct(float(own['own_pct']))}，"
                f"空间梯度匹配误差变化 {float(own['spatial_match'])-float(base['spatial_match']):+.4g}。"
                f"高误差区覆盖 {float(tails[0]['coverage']):.2%}，贡献的 MSE 降幅为 {local_gain:+.4g}，"
                f"全通道 MSE 降幅为 {total:+.4g}。")
    for split in main_splits:
        a=e["TM2"]
        c="E_M0"
        C=2*(number(split,a,"C")-number(split,c,"C"))
        U=number(split,a,"correction_energy")-number(split,c,"correction_energy")
        d=number(split,a,"global_cosine")-number(split,c,"global_cosine")
        g=number(split,a,"G")-number(split,c,"G")
        add(f"- {split} · E-TM2 相对 E-M0 的 Δ2C={C:+.3g}，ΔU={U:+.3g}，ΔG={g:+.3g}，"
            f"全局方向余弦变化 {d:+.4f}。")
    add("")
    add("## 跨分支及联合归因")
    add("")
    for br in ("S3","B","E"):
        for kind in ("EN","TM1","TM2"):
            before,after=f"{br}_M0",f"{br}_{kind}_0.3"
            changes=[number(s,before,policy="final")-number(s,after,policy="final") for s in main_splits]
            add(f"- {br} {kind} 固定 6000 步的实际物理 MSE 降幅（ρ=0.3），开发 {changes[0]:+.4g}，历史测试 {changes[1]:+.4g}。")
    add("")
    add("分支内部配对控制了输入、布局、步数及初始化。跨分支差异同时包含输入和模型条件，"
        "因此 E 的新增历史预测残差是否单独造成效应差异仍为部分支持。")
    for br in ("B","E"):
        for split in main_splits:
            a=effect(split,f"{br}_J4",f"{br}_JOINT_original","final")
            c=effect(split,f"{br}_J5",f"{br}_J4","final")
            add(f"- {br} · {split}，保持四项组成调权后改善 {pct(a)}，"
                f"在相同四权重上增加 MATCH 后再改变 {pct(c)}。")
    inter = read(out/"joint_interaction.csv")
    for split in main_splits:
        excess = mean(float(r["interaction_mse_improvement"]) for r in inter if r["split"]==split and r["policy"]=="final")
        add(f"- E 的 MAG × MATCH 配对交互在 {split} 为 {excess:+.4g} 原尺度 MSE；"
            "正值表示联合的净改善超过两单项净改善之和。")
    add("")
    add("## 开发集选模及历史失败")
    add("")
    legacy="historical_B_ACC_raw_1"
    for split in main_splits:
        for policy in ("selected","final"):
            ct=pair(split,legacy,"B_M0",policy)
            add(f"- 旧冻结 B-ACC · {split} · {policy}，相对自身纯 MSE 改善 {pct(float(ct['effect_pct']))}，"
                f"{ct['improved']}/{ct['trajectories']} 条轨迹改善。")
    previous = [r for r in histories if r["split"]=="development" and r["policy"]=="selected"]
    add(f"- 旧研究有 {sum(r['shared_prefix']=='True' for r in previous)} 个开发选中点处于共享 MSE 前缀。"
        "旧结果与本轮重算的历史锚点身份均已保留。")
    for a,plain in ((legacy,"B_M0"),(e["TM2"],"E_M0")):
        all_tracks=read(out/"trajectories.csv")
        native = {r["trajectory"]:r for r in all_tracks if r["split"]=="development" and r["policy"]=="selected"
                  and r["arm"]==plain}
        deltas=[(float(native[r["trajectory"]]["mse"])-float(r["mse"]),r["trajectory"],r["re"])
                for r in all_tracks if r["split"]=="development" and r["policy"]=="selected" and r["arm"]==a]
        positive=sorted((v for v in deltas if v[0]>0),reverse=True)
        top=positive[:2]
        if top:
            ratio=sum(t[0] for t in top)/sum(t[0] for t in positive)
            add(f"- {a} 的开发正向降幅中，最大两条轨迹贡献 {ratio:.1%}，"
                + "、".join(f"{t[1]}（Re {float(t[2]):g}）" for t in top)+"。")
    for br in ("B","E","S3"):
        for scope in (("core","expanded") if br=="S3" else ("historical_grid","core","expanded")):
            rs=[r for r in sensitivity if r["branch"]==br and r["scope"]==scope and r["policy"]=="selected"
                and r["kind"]=="trajectory" and r["rule"]=="A"]
            count=Counter(r["picked"] for r in rs)
            add(f"- {br} · {scope} 的 Rule-A 留一轨迹选择频率为 "
                + "、".join(f"{k} {n}/12" for k,n in count.most_common())
                + f"。留出改善率的均值为 {pct(mean(float(r['held_vs_plain']) for r in rs))}。")
    add("")
    add("开发留一和影子规则用于敏感性诊断。Rule-A 仍为原定主规则，Rule-B、Rule-C 的模型已在验证前冻结。")
    add("")
    add("## 工况迁移、时段扩展与连续反馈")
    add("")
    hold=read_json(out/"outer_split_manifest.json")
    for label,split in hold.items():
        key=f"holdout_{label}"
        add(f"- Re {split['re']:g} 留出 {len(split['held_files'])} 条轨迹。B-ACC、B-MAG、B-TM2、E-EN、E-TM1、E-TM2"
            " 相对各自纯 MSE 的冻结步改善依次为 "
            + "、".join(pct(number(key,n,"own_pct","final")) for n in ("B_ACC","B_MAG","B_TM2","E_EN","E_TM1","E_TM2"))+"。")
    add(f"冻结主干的训练清单覆盖三组全部留出轨迹。每组 4 个 OOF 参考和 1 个完整子训练 S3 的拟合身份均与外层隔离。"
        f"沿用旧测试的连续 {v['period_windows']} 个窗口，其中 {v['new_period_windows']} 窗位于旧窗口研究的最晚起点之后。"
        "这些是同轨迹时段扩展。依据本地同任务清单，没有满足独立确认条件的合法轨迹。")
    add("")
    for kind in ("M0","EN","TM1","TM2"):
        m=e[kind]
        d=effect("period",f"self_{kind}",m,"final")
        native=number("period",f"self_{kind}","native_pct","final")
        add(f"- E-{kind} 自身反馈相对固定参考的物理 MSE 改善为 {pct(d)}，相对 Native 为 {pct(native)}。"
            "主干在每次发布时用新到达的观测窗口。")
    add("")
    add("## 保留建议与限制")
    add("")
    add("| 数据 | 候选 | 物理 MSE | 相对 Native | 相对 S3 | 相对自身纯 MSE |")
    add("|---|---|---:|---:|---:|---:|")
    for split in (*main_splits,"period_new"):
        policy="final" if split=="period_new" else "selected"
        for arm in dict.fromkeys((winner,b["TM2"],e["TM2"])):
            if (split,policy,arm) not in lookup:
                continue
            add(f"| {split} | {arm} | {number(split,arm,policy=policy):.9g} | "
                + " | ".join(pct(number(split,arm,k,policy)) for k in ("native_pct","s3_pct","own_pct"))+" |")
    add("")
    gains=[number(s,e["TM2"],"own_pct") for s in main_splits]
    history_ct=pair("historical_test",e["TM2"],"E_M0")
    limited=float(history_ct["lower_pct"])<=0 or 2*int(history_ct["improved"])<=int(history_ct["trajectories"])
    outer_gains=[number(f"holdout_{s}","E_TM2","own_pct","final") for s in hold]
    outer_pairs=[pair(f"holdout_{label}","E_TM2","E_MSE","final") for label in hold]
    outer_limited=any(float(r["lower_pct"])<=0 or 2*int(r["improved"])<=int(r["trajectories"]) for r in outer_pairs)
    add(f"E-TM2 的主队列结果为{support(gains,limited)}，工况留出为{support(outer_gains,outer_limited)}。"
        "保留冻结主规则结果及 E-TM2 作为预先指定的机制候选。"
        "正式泛化结论仍需独立同任务轨迹。历史测试反复使用、旧开发参与选择、单训练种子、"
        "主干见过外层轨迹和逐帧多重比较，均限制了论文中的强主张。")
    add("")
    add("## 产物")
    add("")
    names=("PROTOCOL.md","hypotheses.json","experiment_plan.json","data_roles.json","sources.json","reuse_audit.json",
           "FROZEN.json","selections.json","scores.csv","leads.csv","channels.csv","trajectories.csv","regimes.csv",
           "contrasts.csv","lead_primary_contrasts.csv","verification.json","artifact_audit.json","reference_lineage.json","exploration_log.md")
    for name in names:
        add("- "+link(name))
    for path in sorted(figures.glob("*.png")):
        add(f"- [图 {path.stem[:2]}](<{relpath(path,out)}>)")
    (out/"DECISION_REPORT.md").write_text("\n".join(lines)+"\n")

    opts = ["# 有限扩展与停止记录","","主矩阵之外的 EN 追加权重只使用训练集 U。"]
    for p in sorted(out.glob("*energy_match_*_plan.json")):
        item=read_json(p)
        opts.append(f"- {p.stem} 使用 β={item['weight']:.8g}，目标训练 U={item['target']:.8g}。")
    opts.extend(["","附加权重扰动、启用时机、缩放和历史输入组织未加入训练。"
                 "现有梯度、等能量、对角及置换对照构成本轮预先安排的机制矩阵。"
                 "验证冻结后未追加优化。","",
                 "- [能量匹配记录](energy_matching.csv)","- [时段身份清单](period_plan.json)"])
    (out/"exploration_log.md").write_text("\n".join(opts)+"\n")
