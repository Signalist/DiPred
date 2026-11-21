# 交付状态

执行完成；稳定增量效果不支持。当前固定配置下局部传播的证据不足，本轮按协议结束，无资源阻碍或等待继续训练步骤。

起始实际HEAD：a7b7f50caf5b75b2945cc6c8f2ce0076dd77c09f；分支 dynamics。工作目录：/cpfs01/projects-HDD/cfff-3469a2cbe57f_HDD/wangxiyun/DiPred+ResCrct/Workspace/github/DiPred。最终交付提交号可用 `git log -1 --format=%H -- results/nonlinear_parameterization_control/STATUS.md` 查询，push成功及远端提交号以最终回复为准。

代码：research/nonlinear_parameterization_control/{experiment,run,tests,analyze,verify}.py。
结果：results/nonlinear_parameterization_control/。

- 协议：PROTOCOL.md/config.json，先于本轮开发评分。旧pilot gate=false和历史结果哈希不变。
- 实现与执行：7项预检通过；9次正式训练9000步；45个固定检查点（含每次零初始），18项科学/发布选择。
- 冻结评价：FROZEN.json先于读取本轮探索测试目标；旧窗口24块、extension240块；6072行块评分及242880行lead/channel评分。
- 审查入口：DECISION_REPORT.md、selection.csv、candidates.csv、summary.csv、paired.csv、paired_seed.csv、per_Re.csv、per_channel.csv、trajectory_differences.csv。
- 数值／泄漏证据：PREFLIGHT.json、tests.log、runs.json、batch_schedule.json、release_events.jsonl、invariance.csv、verify.log。
- 感受野及开销：preflight_sensitivity.csv、sensitivity.csv、sensitivity_maps.pt、training.csv、cost.csv。
- 完整哈希：config.json旧依赖、FROZEN.json检查点、preflight_data.json/evaluation_audit.json缓存、SHA256SUMS交付。
- 实际执行入口：tests.py；run.py train；run.py freeze；run.py evaluate；analyze.py；verify.py。详细命令和故障归档见RUN_NOTES.md、报告。
- 首次统计中零别名舍入符号问题已修复，仅更改analyze/verify；原模型/选择/评分未变，原版和哈希证据归档。

资源不足造成的未完成实验：无。未执行且不属于本轮授权的研究：Diffusion主干、JVP、反馈记忆头、额外配置和续训。仍缺少未参与历史研究的新独立轨迹及更多事件正例轨迹，限制效果解释，但不妨碍本轮按协议结案。已有用户修改及所有旧结果保留。
