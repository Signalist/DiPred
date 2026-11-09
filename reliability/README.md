# 当前可靠性主线

输入连续20帧，使用原4次去噪99→66→33→0联合生成完整后20帧。零初始噪声输出作为固定点预测；实际推理最后一步的48通道特征进入结构化分位数头，直接估计误差上下界，无需32次采样。物理场和去噪网络均冻结，不重新生成已揭示窗口的预测。

本目录只依赖PyTorch；没有工作区`runs/`、原始数据或实验脚本导入。`checkpoints/current.pt`约7.1 MiB，保留原预测器、原风险头及两个旧预设；`checkpoints/context.pt`约0.91 MiB，补充扩展训练风险头、新适配器、校准参数及冻结的上下文系数。SHA256见`metadata/reliability/`两个provenance文件。

## 四个预设

- `low_regression`（默认）：独立上下尾尺度适配＋历史矩统计＋整体尺度平滑＋覆盖反馈。对应研究版本`separate_tails/scale_fastcal`。测试Score0.631915，覆盖89.977%；相对早期2D参考4/96项退步，匹配对称尾参考仍46项退步。
- `best_mean_score`（保留旧名称）：上一版历史末两帧变化幅度提示＋独立上下尾＋最近3个完整窗口条件分位数拟合，对应`asym_history/memory_dynamic`。Score0.628699、覆盖89.905%。
- `context_mean_score`：当前平均区间评分最优候选。扩展风险头训练时间覆盖，保留旧414样本拟合的历史提示适配器，再由8维已知上下文变化预测下窗误差尺度。Score0.626549、宽度0.476437、覆盖89.911%。相对其自身base为35/36/25项改善/持平/退步，相对早期2D仍7项退步。
- `context_balanced`：扩展风险头与819样本适配器的另一固定对照，加同类上下文尺度预测。Score0.631436、覆盖89.972%，相对其自身base仅19/36/41项改善/持平/退步；不替换低反例默认。

所有预设使用同一个点预测器，不按工况或测试真值自动切换。以上为反复使用过的历史数据上的探索结果，预设选择也受研究历史影响，不是未见数据上的独立选择结论。

上下文包含绝对半宽、历史变化、预测相对最新观测的变化、预测时间/空间梯度及均值等8个摘要。每通道冻结9个回归系数，前两个完整反馈窗口保持原策略；之后预测原始标准化误差的log尾部尺度创新，半宽修正倍数限于0.5…2。只使用已经揭示的残差和当前已知上下文；不调整点预测，也不新增空间排序能力。父控制器的条件形状/幅值更新保持一致，覆盖反馈按实际发出的区间计算。

## 推理

从仓库根目录运行：

```bash
TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 python -m reliability.predict \
  --input history.pt --output forecast.pt \
  --geometry 0.0034 -0.0034 0.02 --preset context_mean_score --device cuda
```

几何数值只是调用示例，必须换成数据真实的有符号dx、dy和正dt。输入为原数据数值尺度float32张量`[B,2,20,32,64]`，通道依次u、v。若原始网格为64×128，按既定协议`[..., ::2, ::2]`下采样；不要额外重复归一化。

结果含`prediction`、`lower`、`upper`、`risk_half_width`，形状均为`[B,2,20,32,64]`，数值尺度与输入相同。区间中点允许偏离固定点预测，风险图定义仍为实际发出区间的半宽。偏平滑和高频不足尚未解决。

连续反馈使用Python接口：

```python
from reliability import ReliabilityPredictor

model = ReliabilityPredictor([dx, dy, dt], preset="context_mean_score", device="cuda")
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
- `context.py`：绝对已知上下文、冻结尺度创新预测与完整窗口后状态更新。上下文预设可传`context_policy="base"/"hold"/"drift"/"context"`重现四种固定策略，默认context；CLI为`--context-policy`。
- `predict.py`：独立推理CLI与predict/reveal/reset接口。
- `verify.py`：不依赖原始数据的CPU/GPU执行与反馈守卫核验。

```bash
TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 python -m reliability.verify --device cpu
python analyze_reliability.py
```

原两个预设已核对72次真实窗口预测，见`results/reliability/portable_inference_audit.json`。两个新增预设×4策略×4窗口×12轨迹共384次预测再次逐位核对，包含修正启用后的40/60窗口，见`results/reliability/context_scale/portable_inference_audit.json`。CPU或不同内核不承诺逐位一致。`TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0`应在启动Python前设置。

这里发布冻结推理实现和现有训练损失、训练协议/权重溯源；仓库根目录的`train.py`仍是历史六arm点预测对照，不是新可靠性预设的端到端复现实验。若改变预测器或风险头权重，必须用保留的独立校准轨迹重算对应参数，不能沿用本包系数。

## 数据与解释边界

按Re组隔离训练47条、校准10条、开发12条、测试12条轨迹。预测器拟合前200帧；原风险头/适配器用46条训练轨迹200:400的414对样本。新增风险头补充45条训练轨迹400:600的405对，共819对，架构/2000次更新/seed42均不变；历史提示适配器仍414对，balanced适配器819对。上下文回归仅用44条训练身份查询620/640/660/680的标签，目标到720帧，后续720:800仅作同身份时间诊断。归一化不使用开发/测试，10条校准轨迹与所有风险拟合身份隔离。没有交叉折、蒸馏或常规多训练种子。

90%是逐点边际覆盖目标。最差帧/区域、窗口覆盖比例和完整20帧联合覆盖分别报告，不能互相替代。当前完整20帧联合覆盖率仍为0。系统学习的是预测误差不确定性，不能直接解释为物理固有随机性或贝叶斯参数不确定性；经验在线校准没有形式CP保证。
