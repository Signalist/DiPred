# 冻结研究源码快照

这些文件原样来自实验，SHA256见`source_hashes.json`，包含记忆训练、相同批次的时间均值控制、训练内pilot、独立校准、完整评估和审计。保留原本地研究模块与`runs/`路径，依赖完整研究工作区，**不是可独立运行的仓库入口**。

可运行推理在[`../../reliability/memory_predict.py`](../../reliability/memory_predict.py)，结构及训练损失在[`../../reliability/memory_heads.py`](../../reliability/memory_heads.py)。训练协议和身份审计在[`../../metadata/reliability/fullblock_memory/`](../../metadata/reliability/fullblock_memory/)。不包含原始数据、预测缓存、认证配置或优化器状态。

根目录`train.py`是历史点预测对照，不能当作本次风险头的完整重训命令。本次发布冻结推理权重、可读实现、完整结果及原训练流程溯源。
