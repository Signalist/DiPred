# XPF16后续实验结果

阅读 [中文报告](REPORT.md)。`python restore_artifacts.py` 拼合并校验分片，恢复两份实际预测及完整逐查询CSV。5个查询头权重、标准化参数、逐组审计、曲线、报告和独立校验已包含。

训练/校准/开发的派生查询特征缓存未上传：它们含RealPDE数据，且可依赖原始工作区重新生成；文件长度与SHA-256见 `local_only_large_files.json`。完整训练重放需原RealPDE、历史权重和工作区依赖，详见 `research/xpf49_followup_20261116/README.md`；原训练脚本的工作区绝对/相对布局不能仅凭本仓库运行。实验为单种子、历史开发集探索。
