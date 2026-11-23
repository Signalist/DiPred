# Corrector 固定实验协议

本轮只做阶段 A 的离线单步机制诊断。残差定义为 `R = Yhat - Y`，目标是同一 DiPred 成员的下一帧 `R[t+1]`。训练和评分使用原始 train/development/test 轨迹划分、完整 32 个扩散成员、窗口历史起点 400 和 420；相邻帧只在窗口内配对。

`R-only` 与 `State-conditioned` 使用完全相同的 9 通道、11,938 参数局部卷积网络、Adam(0.001)、随机种子 42、批次种子 20261119、批量 16 和 1000 次更新。R-only 将 Y、Yhat 两个槽位置零；State-conditioned 输入 `Y[t]`, `Yhat[t]=Y[t]+R[t]`, `R[t]`。两者都保留 `R[t] + delta` 跳连。Persistence 不训练，直接输出 `R[t]`。

真值 `Y[t]` 只用于离线 oracle 条件和构造目标；它不包含目标帧 `Y[t+1]`。因此本结果不能被解释为一次发布 20 帧预测中的可部署收益，也没有进行自由递推或跨发布窗口冷启动。

原始残差缓存位于 `Forecast/Runtime/runs/residual_dynamics_20261119`，其协议和哈希记录在 `results/residual_dynamics/`；本目录只保存轻量代码、配置、日志和模型权重。
