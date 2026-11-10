# 分位数续训：完整可靠性证据

四个冻结风险头：上一轮本地分位数参考`density_reference`、额外2000次纯分位数续训`density_quantile`，以及匹配的`density_gaussian`和`density_student4`对照；每个头采用`memory_dynamic`和`scale_fastcal`两种既定在线策略。新推荐方案为`density_quantile/memory_dynamic`。

`per_case.csv`保留14,976条完整窗口/通道记录：开发和测试各12轨迹、39窗口、u/v两通道、四头、两策略。`start=0`是冷启动；主要比较使用`start=20..760`的38个完整反馈后窗口。`group`为Re组，每个Re及通道等权，不能把不同轨迹数量的组直接混算。

`summary.json`保留冷启动/完整/暖启动汇总。`reference_warm_summaries.json`含55个参考组合：全部47个既有强参考及本次八种头/策略。`regression_counts.json`为440组比较，每组均在原八指标×12个开发/测试Re及通道单元上计96项，阈值固定1e-6。`counterexamples.csv`保留全部退步，无筛除。`test_macro.json`中的平均绝对组覆盖偏差先逐组取绝对值，避免正负抵消。

`paired_trajectory_scores.csv`逐测试轨迹/通道比较本次主方案与上次发布及上一轮本地头。`selected_primary_cell_values.csv`和`primary_comparison_cells.png`提供完整96项的可视化与精确数值。统计为历史轨迹重复开发分析，不作独立统计显著性声明。

```bash
python analyze_continued_reliability.py
```

在仓库根目录运行，仅依赖Python标准库，重算所有暖启动组值、440组比较和全部反例，并确认各头的点MSE完全相同。`analysis_verification.json`是执行结果。`independent_audit.json`、`reference_audit.json`和`direct_inference_audit.json`为本地实验实现核验；`portable_inference_audit.json`是仓库独立推理实现的真实窗口核验，不是独立统计确认。

风险评分和区间宽度是训练通道标准化的无量纲值。风险排序以实际发出区间的半宽对原点预测误差衡量。`raw_*`保留校准前排序；主比较不以它们替代实际发出区间。`whole_block_joint_covered`与90%逐点覆盖不是同一个指标。

Gaussian/Student原训练身份pilot门槛失败；完整评估保留它们作为反例。纯分位数控制通过独立定义的训练内后续门槛才进入本次探索性完整评估，不能将密度失败改写为成功。完整说明见[`../../../docs/研究结果_完整密度对照与随机窗口续训_20261110.md`](../../../docs/研究结果_完整密度对照与随机窗口续训_20261110.md)。
