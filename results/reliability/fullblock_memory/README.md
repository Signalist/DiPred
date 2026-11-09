# 完整误差记忆：结果与全部对照

运行`python analyze_memory_reliability.py`可用Python标准库复核本目录。

|文件|内容|
|---|---|
|`per_case.csv`|18,720条记录：开发/测试各12轨迹×39完整窗口×2通道×5风险头×2在线策略，冷启动与退步窗口均保留|
|`summary.json`|原分组及冷/热启动、早/晚窗口汇总|
|`reference_warm_summaries.json`|39个强参考/同族控制的全部12个开发/测试Re及通道分组|
|`regression_counts.json`|390组比较，每组原8指标×12分组=96项，容差1e-6|
|`counterexamples.csv`|全部15,619条退步，含候选、参考、分组、指标、原值及方向化差异|
|`analysis_verification.json`|从CSV重新计算的核验结果|
|`test_macro.json`|Re/通道等权的测试指标，覆盖偏差先取绝对值再平均|
|`paired_trajectory_scores.json`|逐轨迹/通道配对评分，不是独立样本显著性检验|
|`selected_primary_cell_values.json`|三种关键参考的全部288个单元格数值|
|`memory_information_comparison.png`|冻结核心、无记忆、末帧、均值及完整窗口统一比较，坐标经过放大|
|`full_memory_96cells.png`|与无记忆、均值及上一平均评分最佳版本的完整96项，红色退步全部保留|
|`portable_inference_audit.json`|新仓库版本384次真实窗口预测及完整状态与冻结实现逐位相同|
|`direct_inference_audit.json`|原研究入口192次原生推理及反馈守卫审计|
|`independent_audit.json`|点预测不变、整窗因果顺序及NumPy指标复核|
|`memory_and_mean_audit.json`|时间均值控制与上一实际误差输入审计|
|`frozen_reference_audit.json`|3,744条冻结核心记录及初始校准与旧实现相同|

`full/memory_dynamic`平均Score0.625749768、逐点覆盖89.9406%；相对匹配无记忆77改善/1持平/18退步，相对上一平均最优60改善/0持平/36退步，未全面胜出。时间均值和末帧接近完整窗口，未证明误差迁移机制。

主风险排名使用实际发出区间的半宽，`raw_*`只作校准前诊断。`whole_block_joint_covered`是整段全场同时覆盖事件，`window90/window95`是窗口覆盖比例的阈值事件，均不同于逐点边际覆盖。Score/Width按训练通道标准差标准化，无量纲；所有点预测相同。

历史开发/测试轨迹反复使用；结果不是独立外部验证或形式Conformal/联合时空覆盖保证。训练与最终校准身份分离的审计不消除探索性选择偏差。
