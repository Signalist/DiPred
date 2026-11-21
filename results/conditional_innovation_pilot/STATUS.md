# 条件未解释项小型研究交付状态

本轮有界研究已完成并通过复核。阶段 A 完成六次固定岭回归拟合、开发选模和冻结探索性评价；阶段 B 因预设开发门槛失败跳过，神经训练次数为 0。不存在等待继续训练或统计的任务。阶段 B 的实现和数值测试不代表完成了神经结构实验。

核对时分支为 dynamics，实际 HEAD 为 f2880560fb12191cc8b206444dd78bd5ff49eb58；提交前远端 dynamics 与其一致。唯一 worktree 为 `/cpfs01/projects-HDD/cfff-3469a2cbe57f_HDD/wangxiyun/DiPred+ResCrct/Workspace/github/DiPred`。本文件与其他交付文件同次提交，交付提交号可用 `git log -1 --format=%H -- results/conditional_innovation_pilot/STATUS.md` 查询；实际 push 提交号另由最终回复给出。

代码目录：research/conditional_innovation_pilot/。结果目录：results/conditional_innovation_pilot/。已有未跟踪修改清单保存在 config.json 的 status 字段，保持原状；旧结果、旧权重和主干没有修改。

实际执行入口（仓库根目录，source ../../env.sh 后）：

```bash
python research/conditional_innovation_pilot/tests.py
python -u research/conditional_innovation_pilot/run.py stage_a
python -u research/conditional_innovation_pilot/run.py stage_b
python -u research/conditional_innovation_pilot/run.py freeze
python -u research/conditional_innovation_pilot/run.py evaluate
python research/conditional_innovation_pilot/analyze.py
python research/conditional_innovation_pilot/verify.py
```

日志分别见 tests.log / tests_final.log、stage_A.log、stage_B.log、freeze.log、evaluate.log、analyze.log、verify.log。评价 JSON 序列化和分析语法的两次失败及修复已在 DECISION_REPORT.md 说明，原失败日志保留；未改变候选和冻结选择。最终七项测试与独立证据复核均通过。

审查文件：

- PROTOCOL.md、config.json：先验协议、预算、信息边界、轨迹划分。
- selection_A.json、FROZEN.json、stage_B.json：开发选择、模型哈希、停止依据。
- DECISION_REPORT.md：中文结论、数值、限制和独立目录复跑命令。
- summary.csv、paired.csv、per_channel.csv、per_Re.csv、per_seed.csv：主要结果及原轨迹配对区间。
- ridge_source_*.csv、source_gains.csv：训练／开发源项诊断及相对均值基线的收益。
- per_trajectory.csv、trajectory_differences.csv、per_lead_channel.csv、per_Re_lead_channel.csv：逐级代价。
- scores.csv、detail.csv：2640 行块评分、105600 行细分评分。
- weights/*.pt：六个岭回归 checkpoint 和一个训练均值文件。
- cache_hashes.json、evaluation_audit.json、verification.json、invariance.csv、SHA256SUMS：证据核验。

没有资源阻碍导致的未完成步骤。跨机器复跑需要恢复报告列明的原始数据、大型缓存及既有冻结权重，这些不随本次 Git 提交上传。未来证据缺口是未参与历史研究的独立轨迹；本轮没有 T_local 对 D/T_diag 的实证比较，不据此宣称所有非线性模型无效。本轮按协议结束，不自动开展后续研究。
