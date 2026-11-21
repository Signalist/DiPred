# STATUS — 重启输入归因与冻结采样器 JVP

状态核对时间：2026-11-21T14:57:29.521971+00:00。这是**提交前的执行快照**；最终交付提交号以 Git 中包含本文件的提交和交付消息为准。

## 版本与所在位置

- 执行/核对 HEAD：`5064fa8c483fdc6df8e8d3bfba7a110d388a54bc`；branch：`dynamics`。
- 仓库：`/cpfs01/projects-HDD/cfff-3469a2cbe57f_HDD/wangxiyun/DiPred+ResCrct/Workspace/github/DiPred`。
- 代码：`research/restart_input_attribution/`；结果：`results/restart_input_attribution/`。
- 大缓存：`Workspace/runs/restart_input_attribution/`（绝对路径见cache_manifest.json）。
- 远端实际查询：`5064fa8c483fdc6df8e8d3bfba7a110d388a54bc	refs/heads/dynamics`。此时与本地HEAD一致，ahead/behind为0/0。
- 当前仅一个关联worktree，位于上述仓库；没有reset、clean、强制checkout或强制push。
- 当前任务来自附带goal-objective.md（归档为TASK.md），并遵照后续明确push授权交付。没有重训phase3或重做旧一致性实验。旧phase3状态提交f288056位于本轮基线历史中，本轮不另作该阶段结果判定。

## 完成分类

| 分类 | 实际状态 |
| --- | --- |
| 已实现但未运行 | 无固定清单待运行项 |
| 已运行但未完成统计 | 无 |
| 已完成并通过复核 | 两条训练轨迹预检；288组R/M主控制；288方向JVP与half；逐轨迹统计；输入/噪声/缓存/评分/分解复核 |
| 数据、缓存或算力阻碍 | 无；未恢复缓存，未超过预算 |
| 已完成但尚未提交/远端尚未出现 | 本快照时本轮两个新增目录尚未提交；接下来仅提交这些文件并正常push dynamics，最终远端核对在交付消息中记录 |

## 实际实验与证据

主生成18432条完整20帧路径；全部24条原轨迹、2窗口、2重复、3个k、32成员；JVP=288方向。正式总耗时194.427秒，缓存4951577240bytes，预算2小时/12GiB以内。没有训练、超参搜索或任何网络权重变化。

- `PREFLIGHT.json`、`preflight.log`、`preflight_*.csv`：两条固定训练轨迹，6组32成员复现maxabs=0，12方向/36个有限差分检查全部通过；污染、增量通道梯度与四步标签检查通过。
- `FROZEN.json`：正式评分前协议/配置/代码/预检哈希。
- `RUN_STARTED.json`、`run.log`、`COMPLETE.json`：开始、完整计数、时长、调用成本、显存及缓存字节。
- `plan.json`、`cache_manifest.json`、`events.jsonl`：完整身份、源缓存/新缓存/噪声/输入/输出哈希，以及oracle前缀→保存路径→揭示评分尾段的顺序。
- `quality*.csv`、`lead_channel*.csv`：R/M/S/B及通道、lead、Re、原轨迹评分与配对区间。
- `response*.csv`、`signed_mse*.csv`：全成员/集合均值响应、交叉项、有符号MSE恒等式。
- `jvp*.csv`：固定288方向全幅与半剂量诊断、输入幅度、增益、余弦、绝对/相对误差、原轨迹判定。
- `ANALYSIS.json`、`analyze.log`：完整逐轨迹统计；62/72个轨迹×k通过预先固定全幅标准，0近零方向。
- `VERIFICATION.json`、`verify.log`：24个正式源文件、96个旧缓存和所有新缓存复核通过；评分最大差1.67e-16，代数最大差8.88e-16，JVP缓存指标最大差1.34e-10。
- `ADDITIONAL_AUDIT.json`：另外两条预检训练源文件哈希核验，1728行旧B/S的全部质量指标与原归档逐值一致。
- `DECISION_REPORT.md`：中文结论、数值表、分组例外、可识别性边界及唯一下一轮方法提案（未执行）。

## 实际执行命令与复跑入口

完整主要调用见EXECUTED_COMMANDS.json；所有Python命令均在本仓库、source ../../env.sh 后执行：

```bash
python research/restart_input_attribution/preflight.py
python research/restart_input_attribution/run.py freeze
python research/restart_input_attribution/run.py formal
python research/restart_input_attribution/analyze.py
python research/restart_input_attribution/verify.py
```

输出日志分别保存在本目录。后续复跑必须使用新DIPRED_ATTRIBUTION_OUT与DIPRED_ATTRIBUTION_CACHE目录；完整命令在代码目录README.md。默认入口拒绝覆盖本次已开始/完成的实验。没有重复训练或重复正式采样。

## 剩余工作及限制

固定实验、统计和复核没有剩余项。提交前剩余操作仅为整理Git提交、正常push dynamics及远端HEAD核对。不会启动下一轮研究。

科学结论仅限冻结采样器的离线输入干预。R使用新增真实观测，M不等价于父混合分布；测试仍是探索数据，只有三个Re组。需要新的独立轨迹/Re才可对未来方法做独立确认；本轮不因此增补样本。

## 提交前工作区与worktree原样记录

```text
?? reliability/extreme_predict.py
?? reliability/two_tail.py
?? research/5pde/
?? research/legacy_xpf16_feature_ablation/
?? research/restart_input_attribution/
?? results/reliability/5pde/
?? results/reliability/fair_f_p_pf_xf/
?? results/reliability/fair_f_p_pf_xf_sharedref/
?? results/reliability/fair_x_xp_xpf/
?? results/reliability/legacy_xpf16_feature_ablation/
?? results/residual_dynamics_phase3_followup/cache/
?? results/restart_input_attribution/
```

```text
worktree /cpfs01/projects-HDD/cfff-3469a2cbe57f_HDD/wangxiyun/DiPred+ResCrct/Workspace/github/DiPred
HEAD 5064fa8c483fdc6df8e8d3bfba7a110d388a54bc
branch refs/heads/dynamics

```

所有列出的无关未跟踪文件均保留且不纳入本轮提交。旧研究归档和checkpoint哈希由本轮审计确认未变。
