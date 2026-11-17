# RealPDE F 风险读出与查询先导

结果入口：[中文报告](../runs/f_risk_query_20261116/REPORT.md)。本轮只用原RealPDE，新增训练统一seed42；以开发数据进行机制、局部可靠性与查询筛选。原预测器冻结，不训练五场景。

- `PROTOCOL.md`：初始范围与隔离规则。
- `prepare.py`：从实际冻结四步预测器提取P/F，确定性与随机特征对照。
- `risk.py` / `evaluate_risk.py`：共同中位数风险头、旧头中心分解、局部分组审计。
- `curves.py` / `symmetric_control.py`：容量/标注量和均值中位数同时固定的对照。
- `local_calibration.py`：全局/重叠组尺度的风险—宽度曲线。
- `query.py`：区域时段均值与最大值共享分位查询模型。
- `query_ensemble.py`：真实8/32路径与F边际ECC控制，场样本在所有查询间复用。
- `query_decisions.py`：仅用校准集已见查询确定报警阈值，报告开发误报/漏报。
- `benchmark.py`：实际输入到查询输出的耗时，排除加载与CPU传输。
- `verify.py` / `completion_audit.py`：独立数值、数据/种子/完整性核验。
- `report.py`：结果表、PNG/PDF和中文报告。

从 `Workspace` 运行：

```bash
source env.sh
bash f_risk_query_20261116/run.sh
```

脚本在已有完整模型/特征上跳过训练，重新生成评测表与报告。输出固定在 `Workspace/runs/f_risk_query_20261116`；如需新实验，应先修改 `common.py` 中的 OUT，避免混用权重。数据默认位置为同工作区的 `OLD/REALPDE/realpdebench-competition/data/train_real`，只读。依赖已有PyTorch/functorch、NumPy、h5py、pandas、Matplotlib环境；运行前需有原发布权重、manifest和旧头缓存。

所有结果是旧开发集上的探索，不是新独立测试。Fdet是一调用、同结构的MSE预测器；不能将它与四步扩散的差异完全归因于训练目标。查询模型的负结果仅针对本轮汇聚读出器。逐点99分位CRPS与有限ensemble经验CRPS的估计口径分别保留。
