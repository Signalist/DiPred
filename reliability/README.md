# 当前可靠性主线

输入连续20帧，使用原4次去噪99→66→33→0联合生成完整后20帧。零初始噪声输出作为固定点预测；实际推理最后一步的48通道特征进入结构化分位数头，直接估计误差上下界，无需32次采样。物理场和去噪网络均冻结，不重新生成已揭示窗口的预测。

本目录只依赖PyTorch；没有工作区`runs/`、原始数据或实验脚本导入。`checkpoints/current.pt`约7.1 MiB，包含当前预测器、风险头、两个适配器、训练归一化统计与各自独立校准参数；SHA256见`metadata/reliability/provenance.json`。

## 两个预设

- `low_regression`（默认）：独立上下尾尺度适配＋历史矩统计＋整体尺度平滑＋覆盖反馈。对应研究版本`separate_tails/scale_fastcal`。测试Score0.631915，覆盖89.977%；相对早期2D参考4/96项退步，匹配对称尾参考仍46项退步。
- `best_mean_score`：历史末两帧变化幅度提示＋独立上下尾＋最近3个完整已揭示窗口的条件分位数拟合。对应`asym_history/memory_dynamic`。测试平均Score0.628699、覆盖89.905%；均值评分更低，但不代表覆盖偏差、最差帧或各工况全部更好。

这两个预设使用同一个点预测器，不按工况或测试真值自动切换。不采用之后尚无整体收益的检索距离、敏感度反馈或冻结风险表征变体。以上为反复使用过的历史数据上的探索结果，预设选择也受研究历史影响，不是未见数据上的独立选择结论。

## 推理

从仓库根目录运行：

```bash
TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 python -m reliability.predict \
  --input history.pt --output forecast.pt \
  --geometry 0.0034 -0.0034 0.02 --device cuda
```

几何数值只是调用示例，必须换成数据真实的有符号dx、dy和正dt。输入为原数据数值尺度float32张量`[B,2,20,32,64]`，通道依次u、v。若原始网格为64×128，按既定协议`[..., ::2, ::2]`下采样；不要额外重复归一化。

结果含`prediction`、`lower`、`upper`、`risk_half_width`，形状均为`[B,2,20,32,64]`，数值尺度与输入相同。区间中点允许偏离固定点预测，风险图定义仍为实际发出区间的半宽。偏平滑和高频不足尚未解决。

连续反馈使用Python接口：

```python
from reliability import ReliabilityPredictor

model = ReliabilityPredictor([dx, dy, dt], preset="low_regression", device="cuda")
first = model.predict(history0)
# 待完整20帧目标全部揭示后；此前不得用当前目标调整first。
model.reveal(truth0)
second = model.predict(history1)
# 更换轨迹或批次成员/顺序时：
model.reset()
```

一个实例维护固定顺序的轨迹批次；相邻观测窗口应连续。接口拒绝重复predict、未发预测时reveal、部分帧或非有限反馈。`reset()`必须由调用方在成员或顺序改变时执行，接口无法从张量数值判断轨迹身份。

## 代码与核验

- `networks.py`、`backbone.py`：当前动力学条件预测器、完整20帧U-Net及原4步采样。
- `heads.py`：45060参数时空分位数头、13124参数上下尾适配器。
- `features.py`：仅由已知历史、预测和实际去噪特征计算的输入。
- `calibration.py`：两种完整窗口反馈控制器；更新校准状态，不训练大型预测器。
- `predict.py`：独立推理CLI与predict/reveal/reset接口。
- `verify.py`：不依赖原始数据的CPU/GPU执行与反馈守卫核验。

```bash
TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 python -m reliability.verify --device cpu
python analyze_reliability.py
```

另已在原环境对两个预设各3个真实窗口、12条轨迹，逐位核对点预测、区间、风险半宽及校准状态，见`results/reliability/portable_inference_audit.json`。CPU或不同内核不承诺逐位一致。`TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0`应在启动Python前设置。

这里发布冻结推理实现和现有训练损失、训练协议/权重溯源；仓库根目录的`train.py`仍是历史六arm点预测对照，不是新可靠性预设的端到端复现实验。若改变预测器或风险头权重，必须用保留的独立校准轨迹重算对应参数，不能沿用本包系数。

## 数据与解释边界

按Re组隔离训练47条、校准10条、开发12条、测试12条轨迹。风险头及适配器只拟合其中46条长度足够的训练轨迹200:400时间段，预测器拟合前200帧；归一化不使用开发/测试。独立校准10条。没有交叉折、蒸馏或常规多训练种子。

90%是逐点边际覆盖目标。最差帧/区域、窗口覆盖比例和完整20帧联合覆盖分别报告，不能互相替代。当前完整20帧联合覆盖率仍为0。系统学习的是预测误差不确定性，不能直接解释为物理固有随机性或贝叶斯参数不确定性；经验在线校准没有形式CP保证。
