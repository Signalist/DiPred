"""Write the Chinese decision record from the audited, frozen study outputs."""
from os.path import relpath
from statistics import mean

from research.corrector.run import sha
from research.corrector_window_study.data import read_json

from .data import BRANCHES
from .plot import LABELS,index,read_rows


def write_report(out,figures):
    frozen = read_json(out/"FROZEN.json")
    audit = read_json(out/"artifact_audit.json")
    assert audit["passed"] and audit["freeze_sha256"]==sha(out/"FROZEN.json")
    choice = frozen["selections"]
    decisions = read_json(out/"stage_decisions.json")
    scores = index(read_rows(out/"scores.csv"),("split","policy","arm"))
    contrasts = index(read_rows(out/"contrasts.csv"),("split","policy","arm","control"))
    lead = index(read_rows(out/"leads.csv"),("split","policy","arm","lead"))
    trials = read_rows(out/"loss_search.csv")
    diagnostics = read_rows(out/"gradient_diagnostics.csv")
    conflicts = read_rows(out/"gradient_conflicts.csv")
    weighted = read_rows(out/"weighted_gradients.csv")
    exploration = read_json(out/"exploration_results.json")
    splits = (("development","开发"),("test","历史测试"))
    models = choice["models"]
    global_best = choice["global_best"]
    final = dict.fromkeys(("Native","S3","B_MSE","E_MSE",*choice["best"].values()))
    if choice["best_extension"]:
        final[choice["best_extension"]] = None

    def score(arm,split="development",metric="native_pct",policy="selected"):
        value = scores[split,policy,arm][metric]
        return float(value) if value else None

    def pct(x):
        return "—" if x is None else f"{x:+.3f}%"

    def own(arm,b,split="development",policy="selected"):
        return 100*(1-score(arm,split,"mse",policy)/score(f"{b}_MSE",split,"mse",policy))

    def evidence(dev,test):
        if dev>0 and test>0:
            return "两组评分均支持该对照下的改善"
        if dev>0:
            return "开发集支持改善，历史测试未复现"
        if test>0:
            return "历史测试有改善，开发集缺少支持"
        return "两组评分均缺少改善证据"

    def rho_control(branch,loss):
        core = {"relation":"direction","match2":"acceleration"}.get(loss)
        return next((v["run"] for v in trials if v["branch"]==branch and v["loss"]==core
                     and v["stage"]=="balanced" and float(v["coordinate"])==.3),None)

    def single(b,k):
        return choice["single"][b][k]["best"]

    def by_dev(names):
        return min(names,key=lambda a:(score(a,metric="mse"),a))

    def line(text=""):
        lines.append(text)

    lines = ["# B/E 残差修正器时空 Loss 研究结论",""]
    line(f"共完成 **{decisions['core_fits']} 个核心训练臂**和 **{decisions['extension_fits']} 个扩展训练臂**。"
         f"开发集冻结的全局候选为 **{global_best}**，B 分支为 **{choice['best']['B']}**，"
         f"E 分支为 **{choice['best']['E']}**。选模使用开发集 20 帧物理 MSE，轨迹等权。"
         f"历史测试沿用 12 条已参与研究的轨迹。冻结记录 SHA256 为 `{audit['freeze_sha256']}`。")
    line()
    line("## 1. 基线复现")
    line()
    line(f"B 和 E 均在原批次计划和输入上重放 6,000 步。B 开发/历史测试 Native 修正率为 "
         f"{pct(score('B_MSE'))}/{pct(score('B_MSE','test'))}，E 为 "
         f"{pct(score('E_MSE'))}/{pct(score('E_MSE','test'))}。"
         "训练轨迹的检查点权重、开发 MSE 和两分支第 1,000 步的优化器状态经过审计。")
    line()
    line("## 2–4. B/E 最有效的单项及最优系数")
    line()
    line("| 分支 | 约束 | 最优 λ | 选中模型 | 开发相对自身 MSE | 历史测试相对自身 MSE |")
    line("|---|---|---:|---|---:|---:|")
    for b in BRANCHES:
        for loss,label in LABELS.items():
            picked = choice["single"][b][loss]
            a = picked["best"]
            line(f"| {b} | {label} | {picked['weight']:.6g} | {a} | {pct(own(a,b))} | {pct(own(a,b,'test'))} |")
    line()
    for b in BRANCHES:
        best = by_dev([single(b,k) for k in LABELS])
        line(f"{b} 的最佳单项选择为 **{best}**，开发相对自身 MSE 改善 {pct(own(best,b))}。")
    line()
    line("## 5. 梯度尺度、方向和训练阶段")
    line()
    line("| 分支 | Loss | 0.1×原梯度 / MSE | 校准至 ρ=0.1 的 λ | 校准至 ρ=0.3 的 λ | 与 MSE 余弦 1k / 3k / 6k |")
    line("|---|---|---:|---:|---:|---:|")
    for b in BRANCHES:
        subset = [r for r in diagnostics if r["run"]==f"{b}_MSE" and r["step"]=="1000"]
        base = mean(float(r["gradient_l2"]) for r in subset if r["loss"]=="mse")
        for loss,label in LABELS.items():
            pack = [r for r in subset if r["loss"]==loss]
            ratio = 100*mean(float(r["fixed_0_1_gradient"]) for r in pack)/base
            ranges = [mean(float(r["cosine"]) for r in conflicts if r["run"]==f"{b}_MSE"
                           and r["loss"]==loss and r["other"]=="mse" and int(r["step"])==step)
                      for step in (1000,3000,6000)]
            line(f"| {b} | {label} | {ratio:.2f}% | {float(pack[0]['rho_0_1_weight']):.5g} | "
                 f"{float(pack[0]['rho_0_3_weight']):.5g} | {' / '.join(f'{v:+.3f}' for v in ranges)} |")
    line()
    for b in BRANCHES:
        for k in ("velocity","acceleration","match"):
            raw = choice["single"][b][k]["raw"]
            calibrated = choice["single"][b][k]["balanced"]
            line(f"- {b} {LABELS[k]} 的 0.1、最佳原权重、最佳校准候选相对自身的开发增益依次为 "
                 f"{pct(own(choice['fixed'][b][LABELS[k]],b))}、{pct(own(raw,b))}、{pct(own(calibrated,b))}。")
    line()
    for b in BRANCHES:
        arm = choice["best"][b]
        actual = [r for r in weighted if r["run"]==arm and r["loss"]=="mse"]
        if actual:
            ratios = [100*mean(float(r["auxiliary_ratio"]) for r in actual if int(r["step"])==step)
                      for step in (2000,4000,6000)]
            line(f"{arm} 在 2k/4k/6k 的实际联合梯度占比为 "
                 + "、".join(f"{v:.2f}%" for v in ratios) + "。")
    line()
    line("## 6–8. 平滑、差分匹配和真实方向关系")
    line()
    for b in BRANCHES:
        rankings = [single(b,k) for k in ("velocity","acceleration","match")]
        change = [own(a,b) for a in rankings]
        line(f"- {b} 的一阶平滑、二阶平滑、真实一阶差分匹配，相对纯 MSE 的最佳开发增益分别为 "
             + "、".join(pct(v) for v in change) + "。"
             + f"匹配相对最佳平滑方案的开发增益差为 {change[2]-max(change[:2]):+.3f} 个百分点。")
        for hypothesis in ("relation","match2"):
            run = next((r for r in exploration if r["branch"]==b and r["loss"]==hypothesis and r["executed"]),None)
            if run:
                a,c = (run[k] for k in ("run","control"))
                dev,test = (100*(1-score(a,s,"mse")/score(c,s,"mse")) for s,_ in splits)
                line(f"- {b} {hypothesis} 相对 {c} 的开发/历史测试改善为 {pct(dev)}/{pct(test)}。")
                same = rho_control(b,hypothesis)
                if same and same!=c:
                    dev,test = (100*(1-score(a,s,"mse")/score(same,s,"mse")) for s,_ in splits)
                    line(f"  相对同样以 ρ=0.3 校准的 {same}，分别为 {pct(dev)}/{pct(test)}。")
    line()
    line("## 9–10. 联合与历史输入")
    line()
    line("| 分支 | 原联合 | 调参联合 | 梯度平衡联合 | 最佳两项联合 | 最佳单项 | 最佳联合相对单项 |")
    line("|---|---:|---:|---:|---:|---:|---:|")
    for b in BRANCHES:
        joint = choice["joint"][b]
        best = by_dev([single(b,k) for k in LABELS])
        vals = [pct(own(joint[k],b)) for k in ("original","tuned","balanced","top2")]
        line(f"| {b} | {' | '.join(vals)} | {pct(own(best,b))} | "
             f"{pct(100*(1-score(joint['best'],metric='mse')/score(best,metric='mse')))} |")
    line()
    multi = [r for r in exploration if r["loss"].startswith("omit_") and r["executed"]]
    if multi:
        for x in multi:
            line(f"- {x['run']} 相对 {x['control']} 的开发 MSE 改善 "
                 f"{pct(100*(1-score(x['run'],metric='mse')/score(x['control'],metric='mse')))}。")
    else:
        line("逐项移除的触发条件是联合优于最佳单项、相对本分支纯 MSE 超过 0.5%，且至少 8/12 条开发轨迹改善。此次没有触发。")
    line()
    line("B 与 E 的各自对照隔离了辅助 Loss 的作用。两分支都含上一窗真实残差，E 额外使用冻结的上一窗预测残差。"
         "同一 Loss 在两分支的增益差可以描述这项额外历史输入下的响应差异。"
         "本设计缺少关闭真实历史残差且接受同样 Loss 搜索的对照，互补性的解释限于这两条分支。")
    line()
    line("| 约束 | E 开发相对自身增益减去 B（百分点） |")
    line("|---|---:|")
    for k,label in LABELS.items():
        line(f"| {label} | {own(single('E',k),'E')-own(single('B',k),'B'):+.3f} |")
    line()
    line("## 11. 最终精度与代价")
    line()
    line("| 数据 | 模型 | 物理 MSE | MAE | 通道标准化 MSE | Native | S3 | 自身 MSE |")
    line("|---|---|---:|---:|---:|---:|---:|---:|")
    for s,label in splits:
        for a in final:
            physical,std,mae = (score(a,s,k) for k in ("mse","standardized_mse","mae"))
            rate = [pct(score(a,s,k)) for k in ("native_pct","s3_pct","own_pct")]
            line(f"| {label} | {a} | {physical:.9g} | {mae:.7g} | {std:.7g} | {' | '.join(rate)} |")
    line()
    costs = {r["arm"]:r for r in read_rows(out/"inference_cost.csv")}
    line("| 模型 | 修正器参数量 | batch 8 推理 ms/窗 | 本次训练秒数 |")
    line("|---|---:|---:|---:|")
    for a in final:
        r = costs[a]
        train_cost = f"{float(r['train_seconds']):.1f}" if r["train_seconds"] else "沿用冻结"
        line(f"| {a} | {r['parameters']} | {float(r['ms_per_window_batch8']):.3f} | {train_cost} |")
    line()
    line("## 12. 完整 20 帧修正率")
    line()
    for s,label in splits:
        line(f"### {label}（相对 Native）")
        line()
        line(f"| Lead | {' | '.join(final)} |")
        line("|---:"*(1+len(final))+"|")
        for t in range(1,21):
            values = [pct(float(lead[s,"selected",a,str(t)]["native_pct"])) for a in final]
            line(f"| {t} | {' | '.join(values)} |")
        line()
    line("## 13–15. 收益分解、轨迹与过度平滑")
    line()
    line("时序一阶和二阶量是预测残差的离散差分。若预测差分能量收缩，同时差分匹配误差升高，"
         "则平滑损失可能压低了真值需要的变化。整体和逐帧的收益按 G = 2C − U 核验。")
    line()
    for s,label in splits:
        for b in BRANCHES:
            a,base = choice["best"][b],f"{b}_MSE"
            delta = lambda k: score(a,s,k)-score(base,s,k)
            r = contrasts[s,"selected",a,base]
            relative = [float(lead[s,"selected",a,str(t)]["own_pct"]) for t in range(1,21)]
            low,high = min(range(20),key=relative.__getitem__),max(range(20),key=relative.__getitem__)
            ratio = [score(a,s,f"{k}_predicted_energy")/score(a,s,f"{k}_true_energy") for k in ("first","second")]
            match = [delta(f"{k}_difference_match") for k in ("first","second")]
            offsets = [2*delta("C"),-delta("correction_energy")]
            if delta("G")>0:
                main = "有效内积 2C 增加" if offsets[0]>=offsets[1] else "修正能量 U 降低"
                takeaway = f"两项贡献中，{main}对 ΔG 的贡献较大。"
            else:
                takeaway = ""
            contracted = [name for name,v,error in zip(("一阶","二阶"),ratio,match) if v<1 and error>0]
            if contracted:
                takeaway += f"{'、'.join(contracted)}差分的能量低于真值且匹配变差，有过度平滑信号。"
            line(f"- {label} {b} 相对 {base}，Δ2C 为 {2*delta('C'):+.3g}，ΔU 为 "
                 f"{delta('correction_energy'):+.3g}，ΔG 为 {delta('G'):+.3g}，"
                 f"真值方向余弦变化 {delta('cosine'):+.4f}，RMS 比变化 {delta('rms_ratio'):+.4f}。"
                 f"相邻方向关系匹配误差变化 {delta('adjacent_direction_match'):+.3g}，"
                 f"相邻幅值关系匹配误差变化 {delta('adjacent_magnitude_match'):+.3g}，"
                 f"空间梯度匹配误差变化 {delta('spatial_match'):+.3g}。"
                 f"一阶、二阶预测差分相对真实能量为 {ratio[0]:.3f}、{ratio[1]:.3f}，"
                 f"差分匹配误差变化 {match[0]:+.3g}、{match[1]:+.3g}。"
                 f"首帧 {pct(relative[0])}、末帧 {pct(relative[-1])}，"
                 f"最弱 lead {low+1}（{pct(relative[low])}），最强 lead {high+1}（{pct(relative[high])}）。"
                 f"{r['improved']}/{r['trajectories']} 条改善，95% 配对区间 "
                 f"[{pct(float(r['lower_pct']))}, {pct(float(r['upper_pct']))}]。{takeaway}")
    line()
    regimes = read_rows(out/"regimes.csv")
    channels = index(read_rows(out/"channels.csv"),("split","policy","arm","channel"))
    for s,label in splits:
        for b in BRANCHES:
            a = choice["best"][b]
            rr = sorted((float(r["re"]),float(r["own_pct"])) for r in regimes
                        if r["split"]==s and r["policy"]=="selected" and r["arm"]==a)
            uv = [pct(float(channels[s,"selected",a,c]["own_pct"])) for c in ("u","v")]
            line(f"- {label} {b} 分工况相对自身 MSE 改善为 "
                 + "、".join(f"Re {re:g} {pct(v)}" for re,v in rr) + f"。u/v 通道分别为 {uv[0]}、{uv[1]}。")
    line()
    line("## 16–18. 扩展、负结果与证据强度")
    line()
    for b in BRANCHES:
        a = choice["best"][b]
        dev,test = own(a,b),own(a,b,"test")
        rejected = [label for k,label in LABELS.items() if choice["single"][b][k]["weight"]==0]
        line(f"- {b} 单项最优 λ 为 0 的约束为 {'、'.join(rejected) or '无'}。"
             f"冻结候选相对自身纯 MSE 的开发/历史测试增益为 {pct(dev)}/{pct(test)}。")
    line()
    supports = [b for b in BRANCHES if own(choice["best"][b],b)>0 and own(choice["best"][b],b,"test")>0]
    if supports:
        line(f"{'、'.join(supports)} 在两套评分中都观察到正向增益。"
             "原轨迹配对区间描述这些既有轨迹上的差异。独立泛化与跨训练种子稳健性仍需验证。")
    else:
        line("采用相同输入和训练预算的对照后，没有分支在开发和历史测试两套评分中均体现正向增益。")
    line()
    line("## 19–20. 未解决的问题与下一步")
    line()
    line("同一训练种子和既往使用过的历史测试轨迹限制了外推。"
         "此前的边缘工况迁移失败依然成立，本轮沿用观察到的历史窗、冻结参考和既有工况取样。"
         "长期自身反馈、超出训练 Re 范围的迁移，以及同一约束在关闭历史残差输入时的贡献仍未测量。"
         "下一步优先在身份独立的轨迹上比较冻结候选与各自纯 MSE，覆盖 Re 边缘和前几帧。"
         "如果继续检验历史输入与约束的交互，可在同一预算下增加关闭历史输入的匹配对照。")
    line()
    line("## 材料")
    line()
    line("- [固定协议与复现命令](PROTOCOL.md)")
    line("- [数值审计](artifact_audit.json) · [独立复算](verification.json) · [冻结配置和检查点](FROZEN.json)")
    line("- [完整权重搜索](loss_search.csv) · [联合搜索](joint_search.csv) · [训练中实际梯度](weighted_gradients.csv)")
    line("- [扩展实验记录](exploration_log.md) · [配对区间](contrasts.csv)")
    line("- [全部模型得分](scores.csv) · [逐帧](leads.csv) · [逐轨迹](trajectories.csv) · [分工况](regimes.csv) · [分通道](channels.csv)")
    for name in sorted(audit["figure_sha256"]):
        line(f"- [图 {int(name[:2])}](<{relpath(figures/name,out)}>)")
    (out/"DECISION_REPORT.md").write_text("\n".join(lines)+"\n")

    notes = ["# 自主扩展实验记录","",
             "方向关系匹配检验强制相邻方向一致是否抹去了真实空间变化。"
             "二阶真值差分匹配检验普通二阶平滑是否错误压低实际残差变化。"
             "两项均从相同的 1,000 步状态出发，以 ρ=0.3 的实测梯度校准固定权重。","",
             "| 分支 | 假设 | 对照 | 运行 | 开发相对对照 | 历史测试相对对照 | 判断 | 冻结分支最优采用 |",
             "|---|---|---|---|---:|---:|---|---|"]
    for r in exploration:
        if r["executed"]:
            a,c = r["run"],r["control"]
            d,t = (100*(1-score(a,s,"mse")/score(c,s,"mse")) for s,_ in splits)
            adopted = "是" if choice["best"][r["branch"]]==a else "否"
            notes.append(f"| {r['branch']} | {r['hypothesis']} | {c} | {a} | {pct(d)} | {pct(t)} | {evidence(d,t)} | {adopted} |")
        else:
            notes.append(f"| {r['branch']} | {r['hypothesis']} | {r['control']} | {r['status']} | — | — | 未执行 | 否 |")
    notes += ["","首帧的预测和真值使用同一个历史残差相减，历史项相消，目标就是首帧 MSE。"
               "因此未将该表达式加入训练。完整扩展计划和停止决定见 "
               "[exploration_plan.json](exploration_plan.json) 与 [stage_decisions.json](stage_decisions.json)。"]
    (out/"exploration_log.md").write_text("\n".join(notes)+"\n")
