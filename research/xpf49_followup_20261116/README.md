# XPF16独立双分位头后续研究

[中文报告](../runs/xpf49_followup_20261116/REPORT.md)；`PROTOCOL.md` 为实验协议。此项目直接使用原 `xpf16_fullpercent_20261111` 49独立头、`xpf16_02grid_2048_20261112` 249独立双分位头＋1独立中位数头，以及 `legacy_xpf16_feature_ablation_20261116` 的 F/XP 同结构模型。

`prepare.py` 从冻结DiPred和真实Q05/Q25/Q49头提取查询所需特征；`audit_xpf.py` 和 `audit_grid250.py` 重放原始/在线分布；`query_xpf.py` 训练、评估匹配的查询头；`ecc.py` 使用真实路径和原XPF49边际生成秩耦合样本；`local.py` 做预定义组尺度校准；`benchmark.py` 测量实际耗时；`analysis.py`、`verify.py`、`report.py` 和 `completion_audit.py` 生成逐项统计、独立校验和报告。

依赖原 RealPDE、历史实验模型及 `f_risk_query_20261116` 的训练/校准/开发缓存，全部新结果写入 `runs/xpf49_followup_20261116`，没有新测试数据读入。旧文件只读。运行入口 `bash xpf49_followup_20261116/run.sh`；部分评估会覆盖本实验目录内同名结果，不会覆盖历史结果。GPU显存和数据缓存需要数GB。历史开发集已被重复使用，结果探索性。
