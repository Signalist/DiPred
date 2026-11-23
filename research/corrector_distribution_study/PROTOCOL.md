# Corrector 分布机制研究

固定前序 46/12/12 条训练、开发、旧测试轨迹及起点 400、420、440、460。沿用旧合法窗口、物理单位原生缓存、种子 500000–500031、训练归一化、A/C 各两个已选权重及 bfloat16 卷积。原生路径不重新生成。以 W 为目标、P 为原生预测、Q 为修正预测，定义 R=W−P、修正为 Q−P、E=W−Q。上一窗口记忆为已揭示历史减上一原生集合均值，不跨发布配对成员。

S0/S1/S2 只在训练合法窗上以原标准化经验 CRPS 拟合，从 alpha=1 开始。S2 每通道插值节点为物理 lead 1、10、20。边界 [0.25,4]，L-BFGS-B 最多 300 更新和 300 目标调用，无自动扩展。两通道同权，训练案例等权。开发按原轨迹等权在 identity/S0/S1/S2 中选 S*，平分优先 identity、S0、S1、S2。所有最终得分保持 32 成员，保留均值、分解重构及排序校验。rank 重排使用稳定排序，数值相等按原成员索引打破并列。每窗记录相邻有序样本的并列率；重排具有逐点精确相同的样本多重集。

A/C 同一窗口每种子生成一次，立即计算联合、均值-only、中心化-only、两种反事实排序。逐窗分数在原轨迹内平均，再平均种子得分。分时期报告前两个及后两个起点、分 Re 汇总，同时保留逐种子反例。增量为每个成员物理相隔 5 帧之差，沿用旧通道尺度。路径 ES 沿用前序全路径 RMS 距离。所有主要比较冻结后才打开测试缓存；主要统计比较及其单一指标预定为 S*−native 的经验 CRPS、A−S* 和 C−S* 的经验 CRPS、C−C_native_rank 的路径 ES。配对抽样原轨迹，10,000 次，seed42。四项主要区间用 Bonferroni 98.75% 百分位区间，其他为 95% 描述区间。

先运行训练拟合及开发机制，之后仅依据开发证据选择有限探索。任何新增假设在相应运行前写进探索日志。测试前写 FROZEN.json，固定参数/权重、对照和协议校验。测试仅一次，完成后不回到选择。没有可核验的新独立同任务轨迹时，旧测试及其时段均为探索性。最多 2 GPU 小时、16 GiB 新缓存。逐窗结果保存为轻量 CSV，无新增路径缓存。

从 Corrector 工作树运行（GPU、原本地权重和缓存已在机器上）：

~~~bash
source env.sh
export PYTHONPATH="$PWD/../Forecast/Runtime/.deps${PYTHONPATH:+:$PYTHONPATH}"
python -u -m research.corrector_distribution_study.run --phase development
python -u -m research.corrector_distribution_study.run --phase final
~~~

完整评分复现入口为 `bash research/corrector_distribution_study/reproduce.sh /path/to/fresh-output`。在此次开发证据与预登记探索固定后，该入口复现开发、探索、冻结和一次旧测试。分阶段入口在 development 与 final 之间运行 `--phase exploration`，先保留预登记的 EXPLORATION_LOG.md。探索的插值权重在 float64 构造，alpha 和物理评分在原 float32 路径应用。
