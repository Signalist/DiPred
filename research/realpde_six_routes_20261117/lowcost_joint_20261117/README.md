# 低成本联合分布探索

主报告：[REPORT.md](../runs/lowcost_joint_20261117/REPORT.md)。完整用户目标仍active。此阶段固定逐点XPF分布和32个样本，评估真实残差生成的联合结构；实际加速和任务退步都保留。

所有训练seed42。四个阶段的假设/选择时序见PROTOCOL.md、CONDITIONAL_PROTOCOL.md、AR_PROTOCOL.md及scorefit/protocol.json。前三阶段失败后才转向直接训练联合评分，四阶段均在本目录development评估前冻结。

运行自Workspace，依赖当前主干、49头与原RealPDE缓存。旧文件存在会跳过对应窗口；改机制须使用新OUT目录，不能混用旧结果。

```bash
source env.sh
set -e
python lowcost_joint_20261117/study.py fit
python lowcost_joint_20261117/audit.py mechanics
python lowcost_joint_20261117/study.py calibration
python lowcost_joint_20261117/study.py fit --conditional-scores
python lowcost_joint_20261117/study.py calibration --conditional-scores
python lowcost_joint_20261117/study.py fit --ar
python lowcost_joint_20261117/study.py calibration --ar
python lowcost_joint_20261117/scorefit.py
python lowcost_joint_20261117/study.py calibration --scorefit
python lowcost_joint_20261117/audit.py mechanics --conditional-scores
python lowcost_joint_20261117/audit.py mechanics --ar
python lowcost_joint_20261117/audit.py mechanics --scorefit
for split in development late future; do
  python lowcost_joint_20261117/study.py "$split"
  python lowcost_joint_20261117/study.py "$split" --conditional-scores
  python lowcost_joint_20261117/study.py "$split" --ar
  python lowcost_joint_20261117/study.py "$split" --scorefit
done
python lowcost_joint_20261117/predict.py
python lowcost_joint_20261117/benchmark.py
python lowcost_joint_20261117/benchmark.py --scorefit
python lowcost_joint_20261117/calibrated.py fit
for split in development late future; do
  python lowcost_joint_20261117/calibrated.py "$split"
done
for split in calibration development late future; do
  python lowcost_joint_20261117/audit.py "$split" --scorefit
done
python lowcost_joint_20261117/report.py
```

计时必须单独运行，不能与其他实验同时占用GPU。compact_generator.pt只保留实际推理需要的PCA32与评分统计，预测不再加载完整残差库；统计不参与预测。

直接从物理单位的已观测历史生成32个未来场：

```python
import sys
sys.path.insert(0, 'lowcost_joint_20261117')
from predict import LowCostJointPredictor
predictor = LowCostJointPredictor()
samples = predictor.predict(observed_cuda)  # [32,2,20,32,64]
```

此入口不读取目标或数据缓存。默认输出原始边际版本；固定/在线校准的共同对照在calibrated.py中，尚未做成长期部署控制器。不能将当前候选视为极值任务上的等质量替代，也不能据加速宣称所有物理量更好。
