# 上下文尺度预测：全部固定策略

两个冻结风险源各评估base、hold、drift、context四种策略，每种1872条开发/测试窗口通道记录。`per_case.csv`保留完整14,976条及实际`log_factor`，`summary.json`保留研究原名；父目录分析CSV以可部署预设名标识。

|可部署预设|研究风险源|父在线策略|
|---|---|---|
|context_mean_score|core_history_old|memory_dynamic|
|context_balanced|core_separate_all|scale_fastcal|

context版本直接用预设名，其他策略在父目录的method列加`_base`、`_hold`、`_drift`后缀。mean_score的hold与base严格恒等，作为负控制保留。

`regression_counts.json`及`counterexamples.csv`完整保留原160组比较，每组96项；包括旧2D、独立/对称尾、历史/去噪提示、训练扩展及自身base/hold/drift强参考。父目录`analyze_reliability.py`另用导出的窗口CSV重算当前4预设×12参考的48组比较。两表覆盖参考范围不同，不能混淆。

当前平均评分候选相对自身base：Score0.628363→0.626549，宽度0.478372→0.476437，35项改善、36持平、25退步；最差帧/区域覆盖小幅降低。其逐Re/通道绝对覆盖偏差平均约0.001162，不能用abs(总平均覆盖−0.9)替代。相对早期2D仍7项退步，原低反例默认仍4项，因此没有宣称全线领先。

`portable_inference_audit.json`核对两个新增预设、全部4策略、4连续窗口、12轨迹，共384次真实预测；修正启用、实际区间覆盖反馈、父状态及尺度记忆与冻结实现逐位一致。其他audit文件保留独立NumPy复算、训练身份隔离、同点预测和全部20帧揭示后更新的检查。这里的审计是实现核验，不是新数据确认。

区间Score/width按训练通道标准差标准化，为无量纲；90%是逐点目标。全局通道尺度修正没有增加空间风险排序信息，不提供整段20帧联合覆盖或严格conformal保证。所有评估轨迹均有历史研究使用记录。
