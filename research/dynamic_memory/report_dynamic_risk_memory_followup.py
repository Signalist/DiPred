"""Verify the full dynamic-risk follow-up and preserve every strong reference."""

import json

import torch

from clean_reliability_common import ROOT, configure, dump
from experiment_dynamic_risk_memory_followup import OUT, TRAIN, NAMES, MODES, Provider, controller
from experiment_reliability_transport_e12 import normalization
from conditional_online_risk import risk_features
from audit_fourcall_uq_extensions import audit, regression_tables, test_macro


PREVIOUS = ROOT / "runs/random_window_continuation_followup_20261110"


def identity_audit():
    old = json.loads((PREVIOUS / "per_case.json").read_text())
    keys = ("split", "file", "start", "channel", "variant")
    reference = {tuple(row[key] for key in keys): row for row in old if row["map"] == "density_quantile"}
    current = json.loads((OUT / "per_case.json").read_text())
    count = 0
    for row in current:
        if row["map"] == "frozen_best":
            previous = reference[tuple(row[key] for key in keys)]
            assert all(row[key] == value for key, value in previous.items() if isinstance(value, float))
            count += 1
    assert count == 3744 and len(current) == 11232
    dump(OUT / "reference_audit.json", dict(passed=True, frozen_current_reference_rows_exact=count,
                                           common_point_rows=len(current)))


def report():
    summary = json.loads((OUT / "summary.json").read_text())
    previous = json.loads((ROOT / "github/DiPred/results/reliability/continued_quantile/reference_warm_summaries.json").read_text())
    assert len(previous) == 55 * 12
    previous_pairs = tuple(dict.fromkeys((row["map"], row["variant"]) for row in previous))
    assert len(previous_pairs) == 55
    dump(OUT / "previous_reference_warm_summaries.json", previous)
    references = tuple(dict.fromkeys((*previous_pairs, *((name, mode) for name in NAMES for mode in MODES))))
    assert len(references) == 61
    counts = regression_tables(OUT, summary, previous + summary, NAMES, MODES, references)
    assert len(counts) == 366 and all(sum(row[key] for key in ("wins", "ties", "regressions")) == 96 for row in counts)
    values = []
    for name in NAMES:
        for mode in MODES:
            row = test_macro(summary, name, mode)
            groups = [r for r in summary if r["map"] == name and r["variant"] == mode and r["split"] == "test"
                      and r["period"] == "warm" and r["group"] != "macro"]
            row["mean_absolute_group_coverage_error"] = sum(abs(r["coverage"] - .9) for r in groups) / len(groups)
            values.append(dict(head=name, policy=mode, **row))
    dump(OUT / "test_macro.json", values)
    lines = [
        "# 动态误差风险记忆：完整后续与全部反例", "",
        "原机制门槛保持失败：空间迁移没有八项全面超过同容量调幅模块。另一个训练身份观察值得检验：迁移候选在两种固定控制器下，都比冻结当前最好风险头八项严格改善超过1e-6。因此在不重新拟合、不改变门槛结果的情况下独立开展本完整探索性后续，保留调幅和冻结原头，不把收益直接归因于空间迁移。", "",
        "冻结原四步99→66→33→0、完整20→20点预测及48340参数原风险头；只增加10430参数同结构模块，分别预测每个未来lead的4像素有界位移或两组特征有界增益。它们只处理上一完整已揭示误差的投影特征，绝不修改点预测。原819训练对、2000步seed42后冻结，10个独立校准身份初始化保持锁定。当前目标真值仍在全部区间发出后完整20帧揭示，模块及预测器均在线冻结。", "",
        "|风险头|控制器|覆盖%|组覆盖绝对偏差↓|Score↓|Width↓|AURC↓|top10%↑|最差帧%↑|最差区域%↑|",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in values:
        lines.append(
            f"|{row['head']}|{row['policy']}|{100*row['coverage']:.4f}|{row['mean_absolute_group_coverage_error']:.9f}|"
            f"{row['standardized_interval_score']:.9f}|{row['standardized_width']:.9f}|{row['relative_aurc']:.9f}|"
            f"{100*row['top10_energy']:.5f}|{100*row['worst_frame_coverage']:.5f}|{100*row['worst_patch_coverage']:.5f}|"
        )
    selected = (("density_quantile", "memory_dynamic"), ("density_quantile", "scale_fastcal"),
                ("full", "memory_dynamic"), ("quantile", "memory_dynamic"),
                ("core_history_old", "ordinary_mse"), ("separate_tails", "scale_fastcal"),
                ("previous_2d", "global_smooth"), ("modulate", "memory_dynamic"), ("modulate", "scale_fastcal"))
    lines += ["", "|候选|策略|参考|改善|持平|退步|", "|---|---|---|---:|---:|---:|"]
    for row in counts:
        if row["candidate"] != "frozen_best" and (row["reference"], row["baseline_variant"]) in selected:
            lines.append("|{candidate}|{variant}|{reference}/{baseline_variant}|{wins}|{ties}|{regressions}|".format(**row))
    lines += ["", "完整11232条窗口/通道记录、3744条当前冻结控制逐位复现；全部55个旧参考加6个当前控制、366组原96项比较和所有反例保留。平均Score或训练身份八项改善不能替代原完整目标，也不能用早期2D参考数量掩盖强参考退步。最终不作全线成功声明。", "",
        "Score/Width为训练通道标准化无量纲值，组覆盖误差先逐Re/通道取绝对值再平均；风险排名仍用实际发出半宽对原P误差。原尺度点MSE保持不变。4像素位移不是物理流速，误差潜特征并非物理质量，非零位移不能证明误差输运理论。", "",
        "开发/测试和训练后续轨迹长期反复使用，全部为探索性结果，不是独立泛化确认或严格条件/完整20帧联合覆盖。完整记录位于`runs/dynamic_risk_memory_followup_20261110/`；原训练与失败机制门槛位于`runs/dynamic_risk_memory_20261110/`。",
    ]
    (ROOT / "研究结果_动态误差风险记忆完整后续_20261110.md").write_text("\n".join(lines) + "\n")
    dump(OUT / "goal_audit.json", dict(
        achieved=False, original_mechanism_gate_passed=False,
        all55_previous_reference_pairs_preserved=True, comparisons=len(counts),
        reason="Original96-cell scope and every strong reference must improve; no full-line conclusion from mean score or trainpilot.",
    ))
    for row in values:
        print("DYNAMIC_COMPLETE_MACRO", row["head"], row["policy"],
              "score", row["standardized_interval_score"], "width", row["standardized_width"],
              "aurc", row["relative_aurc"], "patch", row["worst_patch_coverage"],
              "group_cov_error", row["mean_absolute_group_coverage_error"], flush=True)


def main():
    configure()
    assert (OUT / "complete.json").exists()
    identity_audit()
    provider = Provider(normalization())
    stored = torch.load(OUT / "calibration.pt", map_location="cuda", weights_only=False)
    initial = {name: {mode: stored[name]["separable_integral"] for mode in MODES} for name in NAMES}
    audit(OUT, provider, initial, controller, lambda row: None, feature_builder=risk_features)
    report()
    print("DYNAMIC_RISK_MEMORY_FOLLOWUP_AUDIT_PASS", flush=True)


if __name__ == "__main__":
    main()
