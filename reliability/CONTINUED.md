# 当前平均区间评分最优版本

`ContinuedReliabilityPredictor`使用冻结的原四步DiPred和续训的48,340参数分位数风险头。输入连续20帧，一次联合预测紧接的完整20帧；四次原生调用固定为99→66→33→0。风险头读取物理历史、最后一次去噪的48通道特征，以及已完整揭示的上一窗口误差，直接输出非对称误差区间。

本次只续训风险头，没有改变点预测，没有点预测残差修正网络，也不需要32路径采样。新增收益来自纯分位数续训；Gaussian和Student-t完整边际似然对照没有胜过它，不能将本次改善归功于分布假设或新增传播机制。

## 结果与取舍

历史测试轨迹窗口2–39，Re及u/v分组等权：

|版本|逐点覆盖|标准化区间评分↓|标准化宽度↓|平均绝对组覆盖偏差↓|
|---|---:|---:|---:|---:|
|上次GitHub发布，`full/memory_dynamic`|89.9406%|0.625749768|0.476878906|0.00098112|
|上一轮本地分位数头，`density_reference/memory_dynamic`|89.9430%|0.622351110|0.475138573|0.00079881|
|本次分位数头，`memory_dynamic`|89.9358%|**0.620441834**|**0.474026420**|0.00070004|
|同一头，`scale_fastcal`|89.9994%|0.622562504|0.478304814|**0.00017827**|

推荐`memory_dynamic`作为平均评分方案，`scale_fastcal`作为覆盖接近90%的固定对照。每条流使用预先指定的策略，不根据当前未知真值挑选策略。

相对上次GitHub发布，平均评分降低约0.8482%；原八指标×开发/测试Re及通道96项中76改善、20退步。相对上一轮本地分位数头则为73改善、23退步，其中测试AURC由0.267783358变为0.267814919，最差区域覆盖由75.6384%变为75.5819%。没有全指标最优结论。`scale_fastcal`对早期2D参考仍有2/96项退步。所有强参考和反例都保留。

Score和Width按训练通道标准差标准化，为无量纲。90%区间评分等于宽度加20倍上下界越界距离。组覆盖偏差先逐Re/通道取绝对值，再平均。点预测MSE保持约1.28793e-4，属于原数据尺度，SI单位未确认。

## 推理

从仓库根目录运行：

```bash
pip install -r requirements.txt
TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 python -m reliability.continued_predict \
  --input history.pt --output forecast.pt \
  --geometry 0.0034 -0.0034 0.02 --strategy memory_dynamic
```

几何参数只是示例，必须替换为真实的有符号dx、dy和正dt。输入是原尺度float32张量`[B,2,20,32,64]`；按既定协议从原64×128网格用`[..., ::2, ::2]`下采样，不重复归一化。`prediction/lower/upper/risk_half_width`均为同形状、原尺度输出。CPU添加`--device cpu`。

CLI每次都是冷启动，连续在线反馈须复用实例：

```python
from reliability import ContinuedReliabilityPredictor

model = ContinuedReliabilityPredictor([dx, dy, dt], device="cuda")
first = model.predict(history0)
# 完整未来20帧到达后，才提供反馈：
model.reveal(truth0)
second = model.predict(truth0)
model.reveal(truth1)
model.reset()  # 更换轨迹、批次成员或顺序时重置
```

网络参数在线保持冻结。接口保存实际发出的点预测，只在`reveal()`后更新误差记忆及校准状态；拒绝重复发出、部分/非有限反馈，以及不等于上一已揭示块的历史。调用者仍须维护轨迹身份和批次顺序。

## 训练与复核

本次从上一冻结分位数头出发，在相同819对原训练身份200:600的真实预测误差上额外训练2000次，batch4、seed42、AdamW、20%记忆可用性dropout。三个目标采用同初始化、批次、可用性掩码和优化设置。权重冻结后，另用10条独立校准身份重新计算初始参数。开发/测试身份未用于风险头权重或初始校准参数拟合。

```bash
# 纯标准库，重算440组96项比较和全部反例：
python analyze_continued_reliability.py
# 无数据的权重、原四步执行和反馈守卫核验：
TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 python -m reliability.verify_continued --device cpu
```

- [`checkpoints/continued.pt`](checkpoints/continued.pt)：新风险头、误差归一化和初始校准参数；复用未改变的`current.pt`预测器。
- [`../results/reliability/continued_quantile/`](../results/reliability/continued_quantile/README.md)：14,976条完整窗口/通道记录、四种头和两种策略、55个参考组合、440组96项比较、全部反例及逐轨迹评分。
- [`../metadata/reliability/continued_quantile/`](../metadata/reliability/continued_quantile/)：训练协议、失败的密度模型门槛、冻结哈希和拟合审计。
- [`../research/continued_quantile/`](../research/continued_quantile/README.md)：原训练/评估源码溯源，说明完整本地依赖边界。

仓库实现须在96次真实完整窗口预测上与冻结研究实现逐位核对点、区间、历史误差和校准状态；审计见结果目录`portable_inference_audit.json`。跨设备或不同PyTorch内核不承诺逐位一致。

开发/测试轨迹历史已反复分析，当前“最好”是探索性选择，不是新鲜独立确认。90%为逐点边际目标，不是完整20帧联合覆盖或严格条件覆盖，完整窗口联合覆盖仍为0。这里估计的是预测误差风险，不代表已证明的物理随机性或贝叶斯参数不确定性。
