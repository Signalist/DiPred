# 三组完整证据：16、原28、32-noise

本目录的正式比较仅这三组；不包含新增基线。

- `per_case.json`：全部5,616条开发/测试窗口/通道记录，含冷启动和在线窗口。
- `summary.json`、`test_macro.json`：完整分组与Re/u/v等权汇总。
- `regression_counts.json`：16分别对28与32-noise的两组原8指标×12工况/通道比较，容差1e-6。
- `counterexamples.json/csv`：全部99条退步，不隐藏局部代价。
- `paired_test_scores.json`：全部24条测试轨迹/通道对两个参考的评分配对。
- `analysis_verification.json`：标准库从逐窗口记录重算所有比较的审计。
- `direct_inference_audit.json`：原研究接口72次实际预测与原32路径std的逐位复现。
- `portable_inference_audit.json`：本次仓库导出接口的同样核验，16/28每批4次主干调用，32-noise加共同中心132次。
- `events.jsonl`：16与32-noise的完整窗口因果事件；`original28_events.jsonl`为复用28原结果的事件溯源。每种方法均先发出完整20帧区间，再读取真值和更新。
- `finalized_study.json`、`protocol.json`：来源、限定和用户最终对照范围。

运行 `python analyze_selected16_comparison.py` 仅用标准库重算，不需要数据或权重。16对28为23改善/73退步，对32-noise为70改善/26退步；仍不全面胜出。16风险参数少42.14%，平均Score比28差1.346%，比32-noise好4.994%。

三组共用同一零噪声点预测；32-noise用真实32路径标准差独立校准，不是32路径均值中心。Score/Width为训练通道标准化无量纲值，覆盖是逐点边际量。数据反复使用，结果探索性；没有独立统计或严格/联合覆盖保证。16只是神经头输入，固定在线校准器仍有部分重叠物理特征。
