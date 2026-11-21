# STATUS — 生成历史来源与不确定性条件化

这是提交前完成快照：2026-11-21T17:16:49.679821+00:00。实验执行/核对HEAD=`50c25fdeb797de21703c039aeb13e6167b26bdbd`；branch=`dynamics`。最终交付提交号由包含本文件的Git提交及交付消息记录，push后另查询实际远端HEAD验证。

## 位置与Git状态

仓库：`/cpfs01/projects-HDD/cfff-3469a2cbe57f_HDD/wangxiyun/DiPred+ResCrct/Workspace/github/DiPred`。代码`research/provenance_uncertainty_conditioning/`；有效结果为本目录；有效大缓存`/cpfs01/projects-HDD/cfff-3469a2cbe57f_HDD/wangxiyun/DiPred+ResCrct/Workspace/runs/provenance_uncertainty_conditioning_k2510`。旧原生缓存只读，复用manifest保存准确绝对路径与hash。只有一个worktree：

```text
/cpfs01/projects-HDD/cfff-3469a2cbe57f_HDD/wangxiyun/DiPred+ResCrct/Workspace/github/DiPred  50c25fd [dynamics]
```

提交前远端实际查询：`50c25fdeb797de21703c039aeb13e6167b26bdbd	refs/heads/dynamics`。本轮新目录尚未提交；其他用户未跟踪文件保留，已跟踪旧文件没有修改。没有reset、clean、强制切分支或强推。

## 完成分类

| 分类 | 核对结果 |
| --- | --- |
| 已实现但未运行 | 无固定实验待运行项 |
| 已运行但未完成统计 | 无 |
| 已完成并通过复核 | 有效九次1000步训练、45权重、18选择、528评价组、全部指标/轨迹配对统计、VERIFICATION及DELIVERY_AUDIT |
| 因数据/缓存/算力阻碍 | 无 |
| 已完成尚未提交/远端未出现 | 本快照时是；下一步仅提交本轮两个目录并按任务末尾明确授权正常push dynamics，远端核对在交付消息中记录 |

有效训练严格k2/5/10，共9000更新；三臂/三种子抽样和32成员噪声完全一致。直接k0只做预检、开发保护及正式评价。所有科学/保护选择均为step0，属于负实验结果，不等于执行未完成。

## 关键证据文件

- TASK.md、PROTOCOL.md、config.json、TRAINING_FROZEN.json：任务全文与训练前固定条件/hash。
- PREFLIGHT.json、tests.log、preflight_replay.csv、EVALUATION_TESTS.json、PLAN_CHECKS.json：零初始化、完整CRPS两项/梯度、四步导数、来源/std、future污染、成员与物理时间检查。
- TRAINING_COMPLETE.json、TRAINING_SCOPE_AUDIT.json、training.csv、九个train_C_*.jsonl、weights/45小权重：九次各1000有效更新，未改主干，未加loss/候选或搜参。
- development.csv、development_nonzero_*、development_tradeoffs.png：全部36非零候选及其CRPS/ES/MSE/SMSE/离散度代价，没有用回退隐藏结果。
- selection_candidates.csv、selection.json、FROZEN_SELECTION.json：分别科学/保护规则、全部45候选及18独立选择，冻结早于测试。
- evaluation_plan.json、evaluation_manifest.jsonl、extension_manifest.jsonl、events.jsonl：48历史+480后续组，全部输入/noise/cache/输出hash及先发布后truth事件。
- scores.csv.gz、leads_*.csv.gz、scores_by_*、paired_*、lead_summary.csv：种子/轨迹/Re/通道/lead/k/窗口/repeat/政策/视图结果，原轨迹配对bootstrap。temporal大缓存保留ACF和协方差/相关矩阵。
- ANALYSIS.json、DECISIONS.json、VERIFICATION.json、DELIVERY_AUDIT.json及日志：统计、固定稳定性判定、数值/身份和任务范围复核。
- DECISION_REPORT.md：中文研究报告，明确回答八个问题，区分执行/收益/机制/未识别/独立验证边界。
- RESOURCE_ACCOUNTING.json、EXECUTED_COMMANDS.json：有效/作废批次成本、实际命令；README提供全新输出目录复跑入口。

审计量：528组身份、2576640个lead明细、3402固定子集评分复算，评分最大差0；2580个CRPS轨迹区间重算。

## 作废批次与成本

invalid_direct_training/完整保留错误加入k0训练的先前9000更新、45权重、协议/源码/开发结果及中止评价，明确无效且不进入任何有效选择/统计。PROTOCOL_CORRECTION.md解释只恢复原任务k约束，没有改变架构、lr、训练步数、种子、loss和选择门槛。该批冻结原生extension可按锁定身份hash复用，错误适配器权重和分数不复用。

有效训练/开发851.891秒，正式评价624.183秒；作废训练932.820秒及中止评价成本单列。大缓存总留存28.397GiB，均不进Git。不是单次四步预测的同预算改进：重启仍需父+子共八次网络调用。

## 剩余工作与限制

固定清单、统计、报告和审计无剩余项；提交前只剩Git提交、正常push和远端HEAD核对。本轮不启动下一研究阶段。

经验结果为未获得可选择的开发CRPS增益；正式回退保留原生S/B，来源/不确定性增量机制未证实。历史测试、后续时段及纠错前探索都已使用过，只有三个测试Re组，不能称新盲测。新的独立轨迹/Re仍是未来研究缺口，不阻碍本轮负结果交付。

## 提交前工作区原样记录

```text
?? reliability/extreme_predict.py
?? reliability/two_tail.py
?? research/5pde/
?? research/legacy_xpf16_feature_ablation/
?? research/provenance_uncertainty_conditioning/
?? results/provenance_uncertainty_conditioning/
?? results/reliability/5pde/
?? results/reliability/fair_f_p_pf_xf/
?? results/reliability/fair_f_p_pf_xf_sharedref/
?? results/reliability/fair_x_xp_xpf/
?? results/reliability/legacy_xpf16_feature_ablation/
?? results/residual_dynamics_phase3_followup/cache/
```
