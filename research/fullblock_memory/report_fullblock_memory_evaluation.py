"""Audit information controls and keep full primary/strong-reference failures."""

import json

import numpy as np
import torch

from clean_reliability_common import ROOT, configure, digest, dump, manifest
from experiment_fullblock_memory_evaluation import OUT, TRAIN, MEAN, NAMES, MODES, controller, Provider
from experiment_reliability_transport_e12 import normalization, history, target
from experiment_temporal_error_uncertainty import pack, normalized
from pilot_fullblock_memory_risk import OUT as PILOT
from experiment_fullblock_memory_risk import SOURCE
from conditional_online_risk import risk_features
from audit_fourcall_uq_extensions import audit, regression_tables, test_macro, REFERENCE


def controls_audit():
    protocol=json.loads((OUT/"protocol.json").read_text())
    assert all(digest(ROOT/path)==sha for path,sha in protocol["artifacts"].items())
    source=json.loads((SOURCE/"per_case.json").read_text())
    keys=("split","file","start","channel","map","variant")
    reference={tuple(r[k] for k in keys):r for r in source if r["map"]=="expanded_rich"}
    rows=json.loads((OUT/"per_case.json").read_text())
    checked=0
    for row in rows:
        if row["map"]!="frozen_rich":continue
        key=tuple("expanded_rich" if k=="map" else row[k] for k in keys)
        old=reference[key]
        assert all(row[k]==value for k,value in old.items() if isinstance(value,float))
        checked+=1
    assert checked==3744
    initial=torch.load(OUT/"calibration.pt",map_location="cpu",weights_only=False)
    original=torch.load(SOURCE/"calibration.pt",map_location="cpu",weights_only=False)["expanded_rich"]["separable_integral"]
    assert all(torch.equal(value,original[key]) for key,value in initial["frozen_rich"]["separable_integral"].items())
    dump(OUT/"frozen_reference_audit.json",dict(passed=True,reference_rows_exact=checked,
        original_frozen_rich_calibration_exact=True,all_point_predictions_common=True))


@torch.no_grad()
def mean_and_memory_audit(provider):
    frozen=json.loads((MEAN/"head_frozen.json").read_text())
    original=json.loads((TRAIN/"heads_frozen.json").read_text())
    assert frozen["same_minibatches_availability_masks_sha256"]==original["minibatches_and_availability_masks_sha256"]["full"]
    assert digest(MEAN/"mean.pt")==frozen["sha256"]
    packet=torch.load(PILOT/"known/600.pt",map_location="cuda",weights_only=False)
    memory=packet["previous_error"][:2]
    encoded=provider.heads["full"].encode_memory(memory)
    averaged=provider.heads["mean"].encode_memory(memory)
    assert torch.equal(averaged,encoded.mean(2,keepdim=True).expand_as(encoded))
    assert torch.equal(averaged[:,:,0],averaged[:,:,-1])
    numpy=np.log1p(abs(memory.cpu().numpy().astype(np.float64))/provider.scale.cpu().numpy()).clip(0,5)
    numpy=np.broadcast_to(numpy.mean(2,keepdims=True),numpy.shape)
    error=float(abs(averaged.cpu().numpy()-numpy).max())
    assert error<1e-6
    for split,start in (("test",20),("test",40)):
        current=(history(split,start)-provider.norm["mean"])/provider.norm["std"]
        from experiment_direct_error_uncertainty import cached
        previous=cached(split,start-20)["prediction"]
        previous_truth=(target(split,start-20)-provider.norm["mean"])/provider.norm["std"]
        assert torch.equal(current,previous_truth)
        assert torch.equal(current-previous,previous_truth-previous)
    dump(OUT/"memory_and_mean_audit.json",dict(passed=True,mean_control_encoded_mean_preserved=True,
        numpy_mean_encoding_max_error=error,availability_and_minibatch_schedule_identical=True,
        actual_current_history_equals_previous_revealed_target=True,
        no_current_future_in_memory=True,all_allocated_parameters=48340,
        limit="Inputrank differs bydesign; meanaveraging or learnedoffdiagonalweights alone is not a transport mechanism proof."))


def group_error(summary,name,mode):
    rows=[r for r in summary if r["map"]==name and r["variant"]==mode and r["split"]=="test"
          and r["period"]=="warm" and r["group"]!="macro"]
    return float(np.mean([abs(r["coverage"]-.9) for r in rows]))


def report():
    summary=json.loads((OUT/"summary.json").read_text())
    references=json.loads((REFERENCE/"summary.json").read_text())+summary
    for run,names in (("tail_scale_risk_adapter",("separate_tails","tied_tails")),
                      ("fourstep_cue_tails",("asym_history",)),
                      ("extended_train_error_adapters",("expanded_separate","expanded_history")),
                      ("fourstep_denoising_risk_cues",("denoising_change",))):
        references += [r for r in json.loads((ROOT/("runs/"+run+"_20261109/summary.json")).read_text()) if r["map"] in names]
    references+=json.loads((SOURCE/"summary.json").read_text())
    references+=json.loads((ROOT/"runs/centered_score_control_followup_20261109/summary.json").read_text())
    references+=json.loads((ROOT/"runs/context_scale_forecasting_20261109/summary.json").read_text())
    refs=(("previous_2d","global_smooth"),("separate_tails","scale_fastcal"),
          ("tied_tails","scale_fastcal"),("tied_tails","memory_dynamic"),
          ("asym_history","memory_dynamic"),("denoising_change","memory_dynamic"),
          ("core_history_old","ordinary_mse"),("core_history_old","context"),
          ("core_separate_all","effective_context"),("expanded_rich","memory_dynamic"),
          ("expanded_rich","scale_fastcal"))
    refs+=tuple((name,mode) for name in NAMES for mode in MODES)
    # Preserve every reference in the immediately preceding complete study,
    # including stronger ranking/coverage controls whose macro score is worse.
    old_counts=json.loads((ROOT/"runs/centered_score_control_followup_20261109/regression_counts.json").read_text())
    previous_refs=tuple(dict.fromkeys((r["reference"],r["baseline_variant"]) for r in old_counts))
    refs=tuple(dict.fromkeys((*previous_refs,*refs)))
    counts=regression_tables(OUT,summary,references,NAMES,MODES,refs)
    values=[]
    lines=["# 完整上一窗口误差：跨lead风险记忆与时间均值对照","",
        "固定原四步99→66→33→0与完整20→20、零初始噪声点预测。已有45060参数GN风险头增加零初始化40→80误差投影，共48340参数，把完整20帧两通道已揭示误差编码为未来每lead的4个特征，再与物理历史投影相加。没有点预测残差修正或额外Diffusion调用。","",
        "none、last、full三头与随后加入的mean控制使用同样819对原训练身份200:600数据、相同扩展核心初始权重、2000次续训、seed42、batch4/优化器/学习率以及逐步完全相同的20%可用性dropout和批次。none去掉误差信息，last只重复最后误差帧，mean保留逐像素log幅度编码的时间均值、去掉时间变化，full保留完整轨迹。分配参数相同，输入信息秩有意不同；不能称为同活跃函数容量或与旧局部适配器的精确算子对照。","",
        "先在44条原训练身份后续600…760检验，last/full均在两种原在线策略下改善评分、top10捕获且覆盖偏差不恶化，完整序列相对末帧也改善。进一步在读开发/测试场值前训练时间均值控制：大部分收益均值也能获得，完整时序的额外增益很小且排序指标并不一致。因此不预设时序迁移机制已成立。","",
        "四个新头全部冻结后，分别用10条独立校准身份重新初始化。评估仍为开发/测试各12轨迹、39完整窗口；上一误差只取当前已知历史减实际保存的上一点预测，当前窗口目标在全部区间发出后才读取。独立校验旧180预测、补齐380预测；新pilot440次原生预测与旧点/冻结风险边界逐位相同。晚期离线训练包本身含标签，但标签严格不进入特征；不宣称那些混合包加载时未包含标签。","",
        "|风险头|在线策略|测试覆盖%|组覆盖偏差|Score↓|宽度|top10能量%|AURC↓|最差帧%|最差区域%|",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name in NAMES:
        for mode in MODES:
            row=test_macro(summary,name,mode)
            error=group_error(summary,name,mode)
            values.append(dict(head=name,policy=mode,mean_absolute_group_coverage_error=error,**row))
            lines.append(f"|{name}|{mode}|{100*row['coverage']:.3f}|{error:.7f}|{row['standardized_interval_score']:.8f}|{row['standardized_width']:.8f}|{100*row['top10_energy']:.3f}|{row['relative_aurc']:.6f}|{100*row['worst_frame_coverage']:.3f}|{100*row['worst_patch_coverage']:.3f}|")
    lines += ["","|候选|策略|参考|改善|持平|退步|","|---|---|---|---:|---:|---:|"]
    for row in counts:
        if row["candidate"] in ("last","mean","full"):
            lines.append("|{candidate}|{variant}|{reference}/{baseline_variant}|{wins}|{ties}|{regressions}|".format(**row))
    lines += ["","所有原8指标、96项、1e-6容差及强参考保留；风险排名来自实际发出区间半宽，raw_为校准前排名。不能把续训/额外分配参数相对冻结核心的收益当成误差信息收益；先比较none，再比较last/mean。全部轨迹历史已用，未证明独立泛化、严格条件覆盖或整段联合覆盖。"]
    result={(row["head"],row["policy"]):row for row in values}
    full=result[("full","memory_dynamic")]
    none=result[("none","memory_dynamic")]
    mean_result=result[("mean","memory_dynamic")]
    last=result[("last","memory_dynamic")]
    previous_summary=json.loads((ROOT/"runs/centered_score_control_followup_20261109/summary.json").read_text())
    previous=test_macro(previous_summary,"core_history_old","ordinary_mse")
    lines += ["","## 可保留的收益与未解决部分","",
        f"当前平均Score记录为full＋memory_dynamic：{full['standardized_interval_score']:.9f}，相对上一均值候选{previous['standardized_interval_score']:.9f}约低{100*(1-full['standardized_interval_score']/previous['standardized_interval_score']):.4f}%。但宽度{full['standardized_width']:.8f}高于旧候选{previous['standardized_width']:.8f}，AURC与部分区域覆盖也较差；相对上一均值候选60改善/0持平/36退步，测试逐轨迹/通道评分17/24改善、7/24变差。没有全线胜出。","",
        f"严格匹配的信息对照中，none续训本身已从冻结核心Score0.634078865降至{none['standardized_interval_score']:.9f}。full进一步降至{full['standardized_interval_score']:.9f}（较none约低{100*(1-full['standardized_interval_score']/none['standardized_interval_score']):.4f}%），相对none为77改善/1持平/18退步，测试逐轨迹/通道评分21/24改善。top10捕获{100*none['top10_energy']:.4f}%→{100*full['top10_energy']:.4f}%，平均绝对组覆盖偏差{none['mean_absolute_group_coverage_error']:.8f}→{full['mean_absolute_group_coverage_error']:.8f}。这些差异支持本配置历史误差信息有额外小幅价值，冻结核心到none的大部分收益仍来自额外续训/架构，不能全算给记忆。","",
        f"完整时序的独有价值未成立：时间均值Score{mean_result['standardized_interval_score']:.9f}已非常接近full；其top10捕获{100*mean_result['top10_energy']:.6f}%还略高于full的{100*full['top10_energy']:.6f}%。末帧Score{last['standardized_interval_score']:.9f}也接近full，热点捕获{100*last['top10_energy']:.4f}%更高。full对mean仍29/96项退步，对last为47改善/49退步，不能把稠密时间权重或训练内优势当成已经验证的误差迁移机制。","",
        "本族相对早期2D参考的最低反例仍4/96，未低于此前另一个固定参考的3/96观察；最低数量记录不刷新。相对上一均值强参考仍36项退步，相对原低反例separate_tails/scale_fastcal仍38项退步。所有39个参考、390组96项比较及反例完整保留，既有低反例默认不因宏Score下降而自动替换。","",
        "实际入口`infer_fullblock_memory_risk.py`：`FullBlockMemoryReliabilityPredictor(geometry,memory_mode='full',strategy='memory_dynamic')`。只有一个48340参数风险头接在冻结DiPred后，每整段20帧真值揭示后保存真实误差；支持none/last/mean/full和两个原在线校准策略，不需要预测缓存或32路径。输入输出均为原数据尺度[B,2,20,32,64]；更换轨迹/顺序需reset，接口核查下一历史与上一完整已揭示块一致。","",
        "full/mean各2策略×4真实窗口×12轨迹共192次原生四步预测逐位重现点预测、区间、半宽、已揭示误差、校准系数以及全部短期deque/矩统计。重复发出、部分反馈、非连续流与reset守卫通过。模型选择仍是历史数据上的探索，不是独立确认。表格Score和Width按训练通道标准差标准化，为无量纲；原数据点MSE没有改变。","",
        "图`memory_information_comparison.png`保留冻结/续训/所有记忆输入；`full_memory_96cells.png`保留三种参考的完整96项；配对轨迹分数为`paired_trajectory_scores.json`。下一阶段检验位置选择性的风险传播：现有记忆内容的直接卷积路径只覆盖小邻域，GroupNorm有全局统计耦合，不能把二者混同为显式远程位置传播。拟冻结当前头，单独比较相同参数的局部/较长程记忆空间算子；这仍是假设，尚未实现或验证，不能预先归因为现有退步原因。"]
    (ROOT/"研究结果_完整窗口风险记忆与时间均值_20261109.md").write_text("\n".join(lines)+"\n")
    dump(OUT/"test_macro.json",values)
    dump(OUT/"goal_audit.json",dict(achieved=False,reason="Matched memory evidence and all strong-reference failures must be retained; no full-line claim from macro score.",
        evidence=["regression_counts.json","counterexamples.csv","memory_and_mean_audit.json","independent_audit.json"]))
    for row in values:
        print(row["head"],row["policy"],"score",row["standardized_interval_score"],"top10",row["top10_energy"],
              "coverage_error",row["mean_absolute_group_coverage_error"],flush=True)


def main():
    configure()
    assert (OUT/"complete.json").exists()
    controls_audit()
    provider=Provider(normalization())
    stored=torch.load(OUT/"calibration.pt",map_location="cuda",weights_only=False)
    initial={name:{mode:stored[name]["separable_integral"] for mode in MODES} for name in NAMES}
    audit(OUT,provider,initial,controller,lambda row:None,feature_builder=risk_features)
    mean_and_memory_audit(provider)
    report()
    print("FULLBLOCK_MEMORY_EVALUATION_AUDIT_PASS",flush=True)


if __name__=="__main__":
    main()
