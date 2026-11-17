# RealPDE时空联合分布探索

主入口：[中文结果报告](../../results/reliability/joint_structure_20261117/REPORT.md)。本轮使用原RealPDE、冻结四步DiPred/49独立XPF头、seed42，不使用5pde。完整输入假设和选择规则见各`*_PROTOCOL.md`，初始规则见`PROTOCOL.md`。

代码依赖本工作区已有模型、缓存与PyTorch环境，不是独立模型分发包。所有输出写入`runs/joint_structure_20261117`。逐窗口结果存在时跳过，不要把已完成结果与修改后的代码混用；新机制应使用新输出目录。旧结果仅只读。

从`Workspace`执行：

```bash
source env.sh
python joint_structure_20261117/experiment.py calibration
python joint_structure_20261117/experiment.py development
python joint_structure_20261117/audit.py
python joint_structure_20261117/sensitivity.py calibration
python joint_structure_20261117/sensitivity.py development
python joint_structure_20261117/localized.py calibration
python joint_structure_20261117/localized.py development
python joint_structure_20261117/physics.py calibration
python joint_structure_20261117/physics.py development
python joint_structure_20261117/advection.py calibration
python joint_structure_20261117/advection.py development
python joint_structure_20261117/prepare_late.py
python joint_structure_20261117/physics.py late
python joint_structure_20261117/advection.py late
python joint_structure_20261117/helmholtz_filter.py calibration
python joint_structure_20261117/helmholtz_filter.py development
python joint_structure_20261117/helmholtz_filter.py late
python joint_structure_20261117/prepare_future.py
python joint_structure_20261117/physics.py future
python joint_structure_20261117/helmholtz_filter.py future
for stage in sensitivity localized physics advection helmholtz_filter; do
  python joint_structure_20261117/analyze_variants.py "$stage"
done
python joint_structure_20261117/inference.py
python joint_structure_20261117/report.py
python joint_structure_20261117/completion_audit.py
```

`inference.JointPredictor.predict(observed, method='smooth')`可直接从物理单位历史生成32个未来场。`ecc32`是未修正对照，`sol`为有取舍的旋转分量实验。三者都需要32条扩散路径；不宣称采样加速。

`per_window.csv`、`per_trajectory.csv`、`per_re.csv`、`paired.csv`保留平均收益、分组退步与轨迹重采样区间。`audit_*.pt`保存首末整场样本，`parts/*.pt`保存所有窗口的泛函分布和评分。`selection.json`为只用校准集的选择。所有实际采样与选择均为单种子；bootstrap是对现有轨迹重采样，不是重新多种子训练。

## 发布包说明

GitHub 包含实验源码、协议、报告、图表、逐窗口统计和审计摘要。原始数据、预训练权重、`*.pt` 张量缓存、完整场样本及 `parts/` 中间输出未上传；完整重跑或独立场样本重算需按原工作区路径提供这些资源。报告中的缓存相关描述指原工作区。

较大的逐窗口CSV以 `.csv.gz` 保存，可用 pandas 直接读取。
