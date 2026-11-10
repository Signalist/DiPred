# 上次发布：历史误差风险头

更新的分位数续训版本及权重见[CONTINUED.md](CONTINUED.md)。本页保留匹配的记忆信息消融，不再是当前平均评分记录。

入口是`MemoryReliabilityPredictor(memory_mode="full", strategy="memory_dynamic")`。冻结原四步DiPred（99→66→33→0），输入20帧、联合预测完整后20帧。48,340参数风险头读取物理历史、实际去噪特征和已揭示的上一窗口误差，直接输出非对称误差区间。点预测不变，不接点预测Corrector，也不需要32次采样。

## 对照结果

历史测试轨迹窗口2–39，Re及u/v分组等权；在线控制器均为`memory_dynamic`：

|风险头输入|90%逐点覆盖|区间评分↓|宽度↓|平均绝对组覆盖偏差↓|
|---|---:|---:|---:|---:|
|冻结扩展核心（参考）|89.923%|0.634079|0.481095|0.00112408|
|`none`：不读历史误差|89.924%|0.626317|0.476686|0.00107988|
|`last`：重复最后误差帧|89.937%|0.625762|0.476573|0.00100744|
|`mean`：重复编码的时间均值|89.940%|0.625802|0.476870|0.00098547|
|`full`：完整20帧误差|89.941%|**0.625750**|0.476879|**0.00098112**|

四个信息对照使用同样48,340分配参数、819对训练样本、初始核心、2000次续训、seed42、优化器、批次和可用性dropout；输入信息秩有意不同。`none`的在线校准器依然接受完整反馈，只消融风险头的直接误差输入。

`full`比匹配`none`评分低约0.0906%，原8指标×开发/测试Re及通道的96项中77改善、1持平、18退步。冻结核心到`none`的大部分收益来自续训/架构，不能归功于记忆。

相对上一局部平均评分记录0.626485，新评分约低0.1174%，仍有36/96项退步，AURC、宽度和部分最差区域覆盖尚有代价。时间均值和末帧已接近完整序列，未证明完整时序或误差迁移的独有优势。原`ReliabilityPredictor(preset="low_regression")`仍保留。

Score和Width按训练通道标准差标准化，为无量纲。90%区间评分为宽度加20倍的上下界越界距离。原尺度点MSE仍约1.28793e-4，原数据SI单位未确认。组覆盖偏差先逐Re/通道取绝对值再平均，避免正负抵消。

## 使用

从仓库根目录运行：

```bash
TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 python -m reliability.memory_predict \
  --input history.pt --output forecast.pt \
  --geometry 0.0034 -0.0034 0.02 --memory full --strategy memory_dynamic
```

几何参数只是示例，需换成数据真实的有符号dx、dy和正dt。输入为原尺度float32张量`[B,2,20,32,64]`，u/v两通道；原64×128网格按既定协议`[..., ::2, ::2]`下采样，不要额外归一化。输出`prediction/lower/upper/risk_half_width`同形状、同尺度。

单次CLI是冷启动；连续反馈必须复用Python实例：

```python
from reliability import MemoryReliabilityPredictor

model = MemoryReliabilityPredictor([dx, dy, dt], memory_mode="full", device="cuda")
first = model.predict(history0)
# 待紧接的完整20帧真实到达后：
model.reveal(truth0)
second = model.predict(truth0)
model.reveal(truth1)
# 更换轨迹、批次成员或顺序时：
model.reset()
```

下一历史须等于上一完整已揭示块。接口拒绝重复发出、部分反馈、非有限反馈及非连续窗口。轨迹身份和批次顺序仍由调用者负责，数值相同的不同轨迹无法自动识别。`memory_mode`支持`none/last/mean/full`，`strategy`支持`memory_dynamic/scale_fastcal`；两者均只在完整反馈后更新状态，网络冻结。

## 代码与复核

- `memory_predict.py`：独立CLI和predict/reveal/reset，无工作区实验模块或缓存导入。
- `memory_heads.py`：风险结构、误差编码、40→80投影及训练分位数损失。
- `checkpoints/memory.pt`：四个风险头与校准参数，约2.01 MiB；共用`current.pt`的冻结预测器、归一化及特征统计。
- [`../metadata/reliability/fullblock_memory/`](../metadata/reliability/fullblock_memory/)：训练协议、身份审计、冻结哈希及训练内pilot门槛。
- [`../results/reliability/fullblock_memory/`](../results/reliability/fullblock_memory/README.md)：18,720条窗口记录、39个参考、390组96项比较、全部15,619条反例及图像。
- [`../research/fullblock_memory/`](../research/fullblock_memory/README.md)：原训练/校准/评估源码溯源快照，不包含完整本地研究依赖。

```bash
# 仅Python标准库，不需要原始数据、GPU或PyTorch：
python analyze_memory_reliability.py
# 无数据的执行、权重与反馈守卫核验：
TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 python -m reliability.verify_memory --device cpu
```

仓库版本已在4种记忆×2策略×4窗口×12轨迹，共384次原生四步预测上逐位核对点预测、区间、误差记忆及完整校准状态，见`portable_inference_audit.json`。CPU或不同内核不承诺逐位一致。

819对样本来自46条训练轨迹200:400及45条训练轨迹400:600，44条训练身份后续窗口作pilot；最终10条校准身份与训练身份分离，没有开发/测试身份拟合、交叉折、蒸馏或多种子扫描。改变预测器、风险头或归一化后必须重新校准。根目录`train.py`仍是历史点预测对照，不是本次风险头一键重训入口。

开发/测试轨迹长期反复分析，版本选择是探索性研究，不是独立泛化确认。90%是逐点边际目标，不能解释成整段20帧同时覆盖或严格条件覆盖，当前完整窗口联合覆盖仍为0。学习对象是预测误差风险，不是已证明的物理随机性或贝叶斯参数不确定性。
