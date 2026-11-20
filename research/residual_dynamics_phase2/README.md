# DiPred 残差机制验证

从仓库根目录运行完整实验。脚本为每次复跑创建独立结果目录和模型缓存目录。

```bash
bash research/residual_dynamics_phase2/reproduce.sh
```

仅复核本轮已生成的文件，可从仓库根目录执行以下命令。

```bash
source ../../env.sh
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
python research/residual_dynamics_phase2/tests.py
python research/residual_dynamics_phase2/verify.py
```

| 文件 | 分工 |
|---|---|
| common.py、audit.py | SHA-256、轨迹和时段划分、冻结模型重放、原尺度标签对齐、原生路径单独保存 |
| mechanisms.py fit / score | 原 A5/B、A6/C、A4 与零残差、复制、缩放、线性、翻核、镜像、位移的配对比较 |
| decomposition.py | 分离 bias 和 delta，测量时空相关、相关长度、简单预测和成员重排效应 |
| causal.py fit / score、metrics.py | 条件偏差头、完整块发布与揭示、oracle 和零初值开环、边际与全路径概率评分 |
| coupling.py | 保持所有边际不变的成员时间重排，补充对称线性基线 |
| consistency.py | 用真实物理场拟合、验证短时代理演化，检验路径与切向一致性 |
| tests.py、verify.py | 数值和时序守卫测试、真实数据和原结果复核 |

单独运行阶段时，环境变量 `DIPRED_PHASE2_OUT` 和 `DIPRED_PHASE2_CACHE` 应指向同一次运行的目录。原始 HDF5、全部冻结权重、已发布残差缓存在原工作区，路径见 common.py。复跑使用本机 A100 和原软件环境。不要对发布目录重新执行旧阶段一的训练命令。主报告包含方法、数值与反例。[研究报告](../../results/residual_dynamics_phase2/RESEARCH_REPORT.md)
