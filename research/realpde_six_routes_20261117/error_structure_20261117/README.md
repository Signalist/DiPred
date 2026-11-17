# RealPDE 结构误差阶段

结果：[REPORT.md](../runs/error_structure_20261117/REPORT.md)。完整目标仍active；未完成项目见报告及[PROTOCOL.md](PROTOCOL.md)。全部实验为seed42，无多种子训练。

从 Workspace 执行。依赖现有主干、manifest与原RealPDE数据，不是独立权重分发包。输出固定到 `runs/error_structure_20261117`；旧产物不重训。重新实验应复制代码并修改OUT，不应混入历史结果。

```bash
source env.sh
set -e
python error_structure_20261117/structure.py prepare --split train
python error_structure_20261117/structure.py prepare --split calibration
python error_structure_20261117/structure.py diagnose --split train
python error_structure_20261117/structure.py diagnose --split calibration
python error_structure_20261117/audit.py --segments calibration
python error_structure_20261117/local_oracle.py --split calibration
python error_structure_20261117/deterministic.py train
for split in train calibration; do
  python error_structure_20261117/deterministic.py prepare --split "$split"
done
python error_structure_20261117/structure.py learn
python error_structure_20261117/local_learn.py train --initial
python error_structure_20261117/structure.py prepare --split development
python error_structure_20261117/deterministic.py prepare --split development
python error_structure_20261117/structure.py evaluate --split development
python error_structure_20261117/local_learn.py evaluate --split development
python error_structure_20261117/inference_rollout.py audit
python error_structure_20261117/inference_rollout.py rollout
python error_structure_20261117/local_learn.py evaluate --split train
# 以下是首次开发结果后补充的控制；不改变原主候选选择。
python error_structure_20261117/local_learn.py train
python error_structure_20261117/local_learn.py train --fraction 0.25
python error_structure_20261117/local_learn.py train --fraction 0.5
python error_structure_20261117/local_learn.py evaluate --split development
for split in late future; do
  python error_structure_20261117/structure.py prepare --split "$split"
  python error_structure_20261117/deterministic.py prepare --split "$split"
  python error_structure_20261117/local_learn.py evaluate --split "$split"
done
python error_structure_20261117/ablation.py fit
for split in development late future; do
  python error_structure_20261117/ablation.py evaluate --split "$split"
done
python error_structure_20261117/audit.py
python error_structure_20261117/report.py
```

计时应单独运行，不和GPU训练并行。`inference_rollout.Predictor`的输入/输出均为当前checkpoint归一化的CUDA张量 `[B,2,20,32,64]`，不接受标签。主候选为 `Predictor('warp_F_last')`。

报告内的所有表来自原始CSV；`paired.csv`的区间是轨迹重采样，不是重复训练。`local/*_trajectories.csv`保留所有方法、所有轨迹；`rollout_windows.csv`保留自由滚动反例。单窗口后处理收益不等于长滚动收益。

确定性主干1500步为6000步余弦计划中的中间检查点，不能当作单独调优的1500步终点。其训练使用独立seed42批次生成器和显式FP32；原主干为历史训练产物。这里补足的是同输入同四步结构控制，不声称除目标函数外所有训练随机路径完全一致。
