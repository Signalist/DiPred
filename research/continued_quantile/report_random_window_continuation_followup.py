"""Audit the new continuation control against every previous fixed reference."""

import json

import torch

from clean_reliability_common import ROOT,configure,dump
from experiment_random_window_continuation_followup import OUT,TRAIN,NAMES,MODES,Provider,controller
from experiment_reliability_transport_e12 import normalization
from conditional_online_risk import risk_features
from audit_fourcall_uq_extensions import audit,regression_tables,test_macro


PREVIOUS = ROOT/"runs/risk_continuation_followup_20261109"


def identity_audit():
    old = json.loads((PREVIOUS/"per_case.json").read_text())
    keys = ("split","file","start","channel","variant")
    reference = {tuple(row[key] for key in keys):row for row in old if row["map"]=="quantile"}
    current = json.loads((OUT/"per_case.json").read_text())
    checked = 0
    for row in current:
        if row["map"]=="density_reference":
            baseline = reference[tuple(row[key] for key in keys)]
            assert all(row[key]==value for key,value in baseline.items() if isinstance(value,float))
            checked+=1
    assert checked==3744 and len(current)==14976
    dump(OUT/"reference_audit.json",dict(passed=True,current_frozen_reference_rows_exact=checked,
        all_common_point_rows=len(current),oldreference="quantile/eachoriginalcontroller fromprecedingcompletefollowup"))


def report():
    summary = json.loads((OUT/"summary.json").read_text())
    sources = json.loads((ROOT/"github/DiPred/results/reliability/fullblock_memory/reference_warm_summaries.json").read_text())
    sources += json.loads((PREVIOUS/"summary.json").read_text())
    counts = json.loads((PREVIOUS/"regression_counts.json").read_text())
    previous_refs = tuple(dict.fromkeys((row["reference"],row["baseline_variant"]) for row in counts))
    assert len(previous_refs)==47
    lookup = {}
    for row in sources:
        if row["period"]=="warm" and row["group"]!="macro" and (row["map"],row["variant"]) in previous_refs:
            key = tuple(row[name] for name in ("split","group","channel","map","variant"))
            assert key not in lookup or lookup[key]==row
            lookup[key] = row
    references = list(lookup.values())
    assert len(references)==47*12
    dump(OUT/"previous_reference_warm_summaries.json",references)
    refs = tuple(dict.fromkeys((*previous_refs,*((name,mode) for name in NAMES for mode in MODES))))
    counts = regression_tables(OUT,summary,references+summary,NAMES,MODES,refs)
    assert len(counts)==440 and all(sum(row[key] for key in ("wins","ties","regressions"))==96 for row in counts)
    values = []
    for name in NAMES:
        for mode in MODES:
            row = test_macro(summary,name,mode)
            groups = [r for r in summary if r["map"]==name and r["variant"]==mode and r["split"]=="test"
                      and r["period"]=="warm" and r["group"]!="macro"]
            row["mean_absolute_group_coverage_error"] = sum(abs(r["coverage"]-.9) for r in groups)/len(groups)
            values.append(dict(head=name,policy=mode,**row))
    dump(OUT/"test_macro.json",values)
    selected = (("quantile","memory_dynamic"),("quantile","scale_fastcal"),
                ("core_history_old","ordinary_mse"),("separate_tails","scale_fastcal"),
                ("previous_2d","global_smooth"),("density_quantile","memory_dynamic"))
    lines = ["# 完整密度对照中的纯分位数续训控制后续","",
        "Gaussian和Student-t(df=4)全边际似然目标在原训练身份pilot均未胜过匹配的纯分位数目标，原失败门槛保持失败。纯分位数控制在memory_dynamic下相对冻结当前版本的八项pilot宏指标全部严格改善，因此独立建立本探索性后续；保留四种风险头、两个父控制器、全部强参考及反例，不将续训增益归因于密度假设。","",
        "所有头48340参数，新的三头同初始化、同819对原训练身份200:600真实误差、seed42、同2000次额外随机窗口更新、批次/dropout/AdamW/学习率。Gaussian与Student的整体目标用32个原拟合样本上的初始半宽梯度范数固定到Quantile的100%；同时学习误差位置和尺度，不是上一弱辅助损失。误差位置仅影响区间，不替换固定P。Gaussian/Student只定义逐点边际分布，不是完整20帧联合分布。","",
        "所有权重先冻结，再独立计算10条校准身份的参数。原四步99→66→33→0、完整20→20及点预测不变；每完整20帧揭示后才更新在线状态。开发/测试历史反复使用，为探索性结果，不是独立确认或严格覆盖保证。","",
        "|风险头|控制器|覆盖%|绝对组覆盖偏差↓|Score↓|宽度↓|top10捕获%↑|AURC↓|最差帧%↑|最差区域%↑|",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in values:
        lines.append(f"|{row['head']}|{row['policy']}|{100*row['coverage']:.4f}|{row['mean_absolute_group_coverage_error']:.8f}|{row['standardized_interval_score']:.9f}|{row['standardized_width']:.9f}|{100*row['top10_energy']:.5f}|{row['relative_aurc']:.9f}|{100*row['worst_frame_coverage']:.4f}|{100*row['worst_patch_coverage']:.4f}|")
    lines += ["","|候选|策略|参考|改善|持平|退步|","|---|---|---|---:|---:|---:|"]
    for row in counts:
        if row["candidate"]!="density_reference" and (row["reference"],row["baseline_variant"]) in selected:
            lines.append("|{candidate}|{variant}|{reference}/{baseline_variant}|{wins}|{ties}|{regressions}|".format(**row))
    lines += ["","全部14976条完整窗口记录、3744条冻结当前控制与47个旧参考/440组96项比较保留；原1e-6容差和8指标不变。风险排名仍以实际区间半宽对原P误差衡量。覆盖误差先逐Re/通道取绝对值，再平均；Score与Width按训练通道标准差标准化、无量纲，训练NLL可为负但不用于替代区间评分。当前点MSE保持原值。"]
    (ROOT/"研究结果_完整密度对照与随机窗口续训_20261110.md").write_text("\n".join(lines)+"\n")
    dump(OUT/"goal_audit.json",dict(achieved=False,reason="Full8metric96cellstrong-referencefailures mustbe inspected; nofull-line claimfrommeanScore.",
        original_density_gate_passed=False,all47_previous_reference_pairs_preserved=True,comparisons=len(counts)))
    for row in values:
        print(row["head"],row["policy"],"score",row["standardized_interval_score"],"aurc",row["relative_aurc"],
              "coverage_error",row["mean_absolute_group_coverage_error"],flush=True)


def main():
    configure()
    assert (OUT/"complete.json").exists()
    identity_audit()
    provider = Provider(normalization())
    stored = torch.load(OUT/"calibration.pt",map_location="cuda",weights_only=False)
    initial = {name:{mode:stored[name]["separable_integral"] for mode in MODES} for name in NAMES}
    audit(OUT,provider,initial,controller,lambda row:None,feature_builder=risk_features)
    report()
    print("RANDOM_WINDOW_CONTINUATION_FOLLOWUP_AUDIT_PASS",flush=True)


if __name__=="__main__":
    main()
