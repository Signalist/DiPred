"""Record audited progress without replacing full96-cell success by averages."""

import json
from collections import Counter
from pathlib import Path

from clean_reliability_common import ROOT, dump
from experiment_dynamic_risk_memory_followup import OUT


def read(path):
    return json.loads(path.read_text())


def main():
    for filename in ("independent_audit.json", "reference_audit.json", "direct_inference_audit.json",
                     "analysis_verification.json"):
        assert read(OUT / filename)["passed"]
    analysis = read(OUT / "analysis_verification.json")
    assert analysis["comparisons"] == 366 and analysis["cells_per_comparison"] == 96
    assert read(OUT / "direct_inference_audit.json")["actual_forecasts"] == 288
    macros = read(OUT / "test_macro.json")
    counts = read(OUT / "regression_counts.json")
    failures = read(OUT / "counterexamples.json")
    baseline = next(row for row in macros if row["head"] == "frozen_best" and row["policy"] == "memory_dynamic")
    candidate = next(row for row in macros if row["head"] == "warp" and row["policy"] == "memory_dynamic")
    directions = dict(standardized_interval_score=-1, standardized_width=-1, relative_aurc=-1,
                      top10_energy=1, per_frame_top10_energy=1, worst_frame_coverage=1,
                      worst_patch_coverage=1, mean_absolute_group_coverage_error=-1)
    gains = {key: direction * (candidate[key] - baseline[key]) for key, direction in directions.items()}
    assert all(value > 1e-6 for value in gains.values())
    selected = [row for row in counts if row["candidate"] == "warp" and
                (row["reference"], row["baseline_variant"]) in (
                    ("density_quantile", "memory_dynamic"), ("density_quantile", "scale_fastcal"),
                    ("full", "memory_dynamic"), ("core_history_old", "ordinary_mse"),
                    ("separate_tails", "scale_fastcal"), ("previous_2d", "global_smooth"),
                    ("modulate", "memory_dynamic"), ("modulate", "scale_fastcal"))]
    current_failures = [row for row in failures if row["candidate"] == "warp" and
                        row["variant"] == "memory_dynamic" and row["reference"] == "density_quantile"
                        and row["baseline_variant"] == "memory_dynamic"]
    assert len(current_failures) == 12
    relative_gain = 100 * gains["standardized_interval_score"] / baseline["standardized_interval_score"]
    synopsis = dict(
        best_mean_candidate=["warp", "memory_dynamic"], test_warm=candidate,
        previous_best_mean=baseline, test_eight_macro_directional_gains=gains,
        relative_score_improvement_percent=relative_gain,
        paired_test_trajectory_scores=analysis["paired_trajectory_scores"],
        selected_original96cell_comparisons=selected,
        twelve_regressions_against_previous_best=current_failures,
        current_failure_metric_counts=dict(Counter(row["metric"] for row in current_failures)),
        original_mechanism_gate_passed=False, whole_goal_achieved=False,
        caveat="All eight test macro metrics improve, but12 originalcells still regress versus preceding mean candidate and44 versus matched modulation. Reused exploratory identities, not independent statistical confirmation.",
    )
    dump(OUT / "audited_progress.json", synopsis)
    report = ROOT / "研究结果_动态误差风险记忆完整后续_20261110.md"
    marker = "<!-- audited_progress_20261110 -->"
    text = report.read_text()
    if marker not in text:
        text += "\n" + marker + "\n\n## 已审计结论\n\n"
        text += (f"warp/memory_dynamic刷新本地平均评分记录：{baseline['standardized_interval_score']:.9f}→"
                 f"{candidate['standardized_interval_score']:.9f}，约低{relative_gain:.5f}%。测试八项Re/通道等权汇总都严格改善超过1e-6；"
                 f"但原96项比较仍为83改善/1持平/12退步，不能把八项宏平均同时改善当作全线胜出。"
                 f"测试逐轨迹/通道Score为23/24改善，对同容量调幅则16/24改善。\n\n")
        text += ("对固定强参考的退步数量也有小幅减少：对更早发布full/memory_dynamic从20降到18，对旧均值强参考"
                 "core_history_old/ordinary_mse从27降到25；同warp头的scale_fastcal对旧低反例强参考从19降到18。"
                 "早期2D参考下仍是2项，且两个工况相同。相对同容量调幅的memory_dynamic仍44/96项退步，未证明迁移独有优势。\n\n")
        text += (f"剩余12项按指标为{synopsis['current_failure_metric_counts']}，完整逐项数值在audited_progress.json及counterexamples文件中。"
                 "保留所有反例，没有按工况切换策略、重调增益或挑选种子。\n\n")
        text += ("366组96项比较及全部反例已由标准库从11232条窗口记录独立重算；当前冻结控制3744条逐位重现。"
                 "288次实际原四步预测的点、区间、上一误差及全部deque/矩状态逐位一致；重复发出、部分/非有限反馈、"
                 "非连续窗口及reset守卫通过。源码推理入口infer_dynamic_risk_memory.py，"
                 "DynamicRiskMemoryPredictor(geometry, objective='warp', strategy='memory_dynamic')；无需预测缓存或32路采样。\n\n")
        text += ("图：dynamic_memory_96cells.png保留五组完整96项和所有红色退步；dynamic_memory_paired_scores.png展示全部24个测试轨迹/通道，"
                 "精确值分别在selected_primary_cell_values.csv和paired_trajectory_scores.json。图的改进幅度是百分数，整体仅约0.058%，不是百分点或显著性声明。\n\n")
        text += ("本地新候选尚未自动推送GitHub；仓库仍是用户上次要求发布的39d7085版本。后续应先区分方向性对齐与双线性平滑，"
                 "再讨论输运机制贡献。所有历史开发/测试已反复使用，不构成独立确认或严格覆盖。\n")
        report.write_text(text)
    print("AUDITED_DYNAMIC_RISK_PROGRESS", candidate["standardized_interval_score"],
          synopsis["current_failure_metric_counts"], flush=True)


if __name__ == "__main__":
    main()
