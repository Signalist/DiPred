## 研究评估报告：当前瓶颈与 CRPS 改进路线

[阅读报告](docs/研究报告_当前瓶颈与CRPS改进路线_20261111.md) · [结构化证据与来源](results/reliability/research_review_20261111/evidence.json)

报告分析两分位输出、局部可靠性和独立验证的限制，建议先检验 **XPF16 单调分位函数＋CRPS 训练**，再研究历史误差驱动的分布校准。相关方案目前是研究计划；现有结论仍是探索性的区间与风险排序优势。新增 CRPS 诊断和去噪阶段对照报告，保留分布假设导致排序反转及各项负结果。

---

# XPF16、8-noise、32-noise：统一在线时空校准对照

三组使用同一冻结四步DiPred、同一零噪声点预测P和同一在线时空校准算法；每组独立拟合并更新校准状态。8-noise由实际8条路径估计std。之前XPF16与32-noise已使用相同控制器，本次明确核验并加入8-noise。

|测试窗口2–39，Re/u-v等权|90%覆盖率|宽度↓|区间评分↓|高斯补全CRPS↓|Laplace补全CRPS↓|AURC↓|
|---|---:|---:|---:|---:|---:|---:|
|XPF16|89.932%|**0.477147**|**0.626680**|**0.082509**|0.083209|**0.267562**|
|8-noise|89.979%|0.486107|0.677056|0.083054|0.083225|0.282193|
|32-noise|89.985%|0.479577|0.659619|0.082544|**0.082885**|0.275746|

XPF16区间评分分别比8-noise、32-noise低7.44%、4.99%；CRPS排序依赖分布补全假设，不声称稳定领先。两分位头不唯一决定完整CDF。宽度/区间评分/CRPS按训练通道std标准化。覆盖为逐点边际覆盖；历史数据反复使用，结果探索性，不是独立确认或联合覆盖保证。

```bash
python analyze_xpf16_noise8_noise32.py
```

[完整5616条记录、图表、校准状态和全部反例](results/reliability/xpf16_noise8_noise32/README.md) · [详细报告](docs/研究结果_XPF16与8noise32noise统一在线校准_20261111.md) · [研究源代码说明](research/xpf16_noise8_noise32/README.md)。原XPF16推理权重与接口保持不变。

---

# DiPred — 16通道精简版与三组正式对照

<!-- selected16_three_method_release -->
本次仅发布**16通道、原28通道、32-noise**三组正式比较。入口 [`reliability/selected16_predict.py`](reliability/selected16_predict.py)，完整用法见 [`reliability/SELECTED16.md`](reliability/SELECTED16.md)，权重及初始校准已包含。原四步99→66→33→0、20帧输入→联合预测完整后20帧保持冻结。

|测试窗口2–39，Re/u/v等权|风险头参数|Interval Score↓|Width↓|90%逐点覆盖|
|---|---:|---:|---:|---:|
|16通道|**34004**|0.626680|0.477147|89.932%|
|原28通道（同预算续训）|58770|**0.618357**|**0.473093**|89.933%|
|32-noise标准差|无额外神经头|0.659619|0.479577|89.985%|

16通道为梯度2＋坐标2＋历史时间波动2＋去噪特征8＋预测时间变化2；真实16输入卷积，移除历史/误差投影和位移模块。风险参数少42.14%，Score比原28差1.346%、比32-noise低4.994%。原96项分别23改善/73退步、70改善/26退步，仍有局部覆盖代价；不是无损裁剪或全线胜出。原28复用上一轮完全匹配的819训练对/2000步/seed42权重，并非本分支旧warp的0.620080版本。

```bash
pip install -r requirements.txt
TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 python -m reliability.selected16_predict \
  --input history.pt --output forecast.pt \
  --geometry 0.0034 -0.0034 0.02 --method selected16
python analyze_selected16_comparison.py
TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 python -m reliability.verify_selected16 --device cpu
```

几何值须换成真实dx/dy/dt。另外两组用 `--method original28` 或 `--method noise32`。CLI为冷启动；连续在线反馈使用`RequestedThreePredictor.predict/reveal/reset`，仅完整20帧真值揭示后反馈。三组共用零噪声点预测中心；32-noise实际生成32条路径取std并独立校准，不是32路径均值中心，也非经验5/95分位数。16/28各4次主干调用，32加共同中心为132次；未声称端到端加速。

[完整证据](results/reliability/selected16_comparison/README.md)保留全部5616条窗口/通道记录、两组96项对照、全部99反例及逐轨迹分析。Score/Width无量纲，点预测MSE不变。在线校准器仍有部分重叠物理特征；16仅指神经风险头。历史数据反复使用，结果探索性，无独立统计或严格/联合覆盖保证。旧版本作为历史归档保留，main分支不变。

---

# DiPred — 动态风险记忆支线

本分支 `research/dynamic-risk-memory` 发布当前**平均区间评分最佳的完整审计版本**：冻结原四步DiPred，20帧输入→联合预测完整后20帧，新增动态历史误差风险记忆。入口 [`reliability/dynamic_predict.py`](reliability/dynamic_predict.py)，使用说明与在线接口见 [`reliability/DYNAMIC_MEMORY.md`](reliability/DYNAMIC_MEMORY.md)。推理权重和初始校准已包含，不需要32路采样。

|测试窗口2–39，Re/u/v等权|上次发布 `39d7085`|本分支 `warp/memory_dynamic`|
|---|---:|---:|
|Interval Score↓|0.620441834|**0.620079902**|
|Width↓|0.474026420|**0.473896527**|
|AURC↓|0.267814919|**0.267708547**|
|最差空间块覆盖率↑|75.5819%|**75.6153%**|
|90%逐点覆盖率|89.9358%|89.9374%|

Score仅降低约0.05833%，点预测MSE不变。原8指标×12分组比较为83改善/1持平/12退步；对同容量纯调幅对照仍有44项退步，未证明空间迁移独有优势，原机制门槛保持失败。开发/测试历史反复使用，结果是探索性的；不声称全面胜出、独立泛化或严格/联合覆盖。Score/Width为训练通道标准化无量纲指标。

```bash
pip install -r requirements.txt
TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 python -m reliability.dynamic_predict \
  --input history.pt --output forecast.pt \
  --geometry 0.0034 -0.0034 0.02 --objective warp --strategy memory_dynamic
python analyze_dynamic_memory.py
TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 python -m reliability.verify_dynamic --device cpu
```

几何值必须换成真实dx/dy/dt。CLI是无历史误差的冷启动；连续20帧反馈使用 `DynamicRiskMemoryPredictor.predict/reveal/reset`。另保留 `--objective modulate` 同容量对照与 `--strategy scale_fastcal` 固定策略。全部11,232条窗口/通道记录、61个参考组合、366组96项比较及9,536反例见[完整证据](results/reliability/dynamic_memory/README.md)。旧版本全部保留，main分支不变。

---

# DiPred

当前推荐的平均区间评分方案是**冻结四步DiPred＋续训分位数风险头**。输入20帧，一次联合预测完整后20帧；保留99→66→33→0四次调用，直接输出误差区间，无需32路采样。入口为[`reliability/continued_predict.py`](reliability/continued_predict.py)，完整用法见[`reliability/CONTINUED.md`](reliability/CONTINUED.md)。推理权重和初始校准参数已包含。

|测试窗口2–39，Re及u/v等权|90%逐点覆盖|标准化区间评分↓|标准化宽度↓|
|---|---:|---:|---:|
|上次GitHub发布 `full/memory_dynamic`|89.9406%|0.625750|0.476879|
|上一轮本地分位数头|89.9430%|0.622351|0.475139|
|本次 `memory_dynamic`|89.9358%|**0.620442**|**0.474026**|
|同一头 `scale_fastcal`|89.9994%|0.622563|0.478305|

相对上次GitHub发布，平均区间评分降低约0.8482%，原八指标/96项比较为76改善、20退步。相对上一轮本地头仍有23项退步，AURC和最差区域覆盖略差，尚未全线超过强参考。点预测MSE保持原值。Gaussian和Student-t对照未胜过纯分位数，改进属于风险头续训收益，不作为新分布机制的证明。

```bash
pip install -r requirements.txt
TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 python -m reliability.continued_predict \
  --input history.pt --output forecast.pt \
  --geometry 0.0034 -0.0034 0.02 --strategy memory_dynamic
python analyze_continued_reliability.py
TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 python -m reliability.verify_continued --device cpu
```

几何参数需换为真实dx、dy、dt。CLI是冷启动；在线连续反馈使用`ContinuedReliabilityPredictor.predict/reveal/reset`，完整20帧真值到达后才反馈。[结果目录](results/reliability/continued_quantile/README.md)包含14,976条窗口/通道记录、全部47个既有参考和本次控制、440组96项比较、所有反例及逐轨迹分析。开发/测试历史已反复使用，结果为探索性；90%逐点覆盖不代表严格条件覆盖或完整20帧联合覆盖。Score和Width为训练通道标准化的无量纲值。

---

## 上次发布：完整窗口误差记忆对照

上次发布的平均区间评分方案是**冻结四步DiPred＋历史误差风险头**。输入连续20帧，一次联合预测完整后20帧；保持99→66→33→0四次调用，直接输出预测误差区间，无需32路采样。

入口：[`reliability/memory_predict.py`](reliability/memory_predict.py)，使用说明：[`reliability/MEMORY.md`](reliability/MEMORY.md)。权重、归一化、校准参数和完整对照均已包含。

|测试窗口2–39，Re及u/v等权|90%逐点覆盖|标准化区间评分↓|标准化宽度↓|
|---|---:|---:|---:|
|匹配无记忆头 `none`|89.924%|0.626317|0.476686|
|末帧误差 `last`|89.937%|0.625762|0.476573|
|时间均值 `mean`|89.940%|0.625802|0.476870|
|完整窗口误差 `full`|89.941%|**0.625750**|0.476879|

推荐`full/memory_dynamic`，48,340参数风险头；点预测保持原样。评分从上一局部最好0.626485降至0.625750，但对该强参考仍有**36/96项退步**。相对匹配无记忆是77改善/1持平/18退步。末帧与时间均值已接近完整序列，未证明完整误差时序或迁移机制的独有优势。旧低反例默认`low_regression`继续保留，没有单个全指标最优方案。

```bash
pip install -r requirements.txt
TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 python -m reliability.memory_predict \
  --input history.pt --output forecast.pt \
  --geometry 0.0034 -0.0034 0.02 --memory full --strategy memory_dynamic
python analyze_memory_reliability.py
TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 python -m reliability.verify_memory --device cpu
```

几何参数需替换成真实dx、dy、dt。单次CLI为冷启动；连续在线实验使用`MemoryReliabilityPredictor.predict/reveal/reset`，只在完整20帧真值揭示后反馈。独立入口不依赖本地研究缓存。

新增[`results/reliability/fullblock_memory/`](results/reliability/fullblock_memory/README.md)含18,720条窗口记录、39个参考的390组96项比较、全部反例、指标图与逐轨迹评分。384次真实窗口预测的仓库输出和完整反馈状态与冻结实验逐位一致。Score/Width为训练通道标准化的无量纲指标，点MSE仍约1.28793e-4。开发/测试数据历史已反复使用，不能视作独立确认或严格/联合覆盖保证。

---

## 此前发布的可靠性版本

当前主线：**Reliability-aware PDE Forecasting**。冻结原4步DiPred（99→66→33→0），输入20帧、联合预测后20帧，再直接输出预测误差区间，不需要32路采样。没有点预测残差修正网络。

本次更新的入口是 [`reliability/`](reliability/README.md)。已包含共享预测权重、归一化/校准参数，以及最新“扩展训练风险头＋已知上下文尺度预测”方案。新增权重约0.91 MiB，继续使用原4次网络调用：

|预设|定位|测试90%逐点覆盖|标准化区间评分↓|标准化宽度↓|
|---|---|---:|---:|---:|
|`low_regression`（默认）|当前低反例主线|89.977%|0.631915|0.479664|
|`context_mean_score`|当前平均区间评分最优候选|89.911%|**0.626549**|**0.476437**|
|`context_balanced`|上下文尺度的另一固定对照|89.972%|0.631436|0.479199|
|`best_mean_score`|保留的上一版平均评分候选|89.905%|0.628699|0.478440|

统计为历史测试轨迹窗口2–39，Re及u/v等权；全部使用完全相同的点预测，MSE约1.28793e-4。新方案相对匹配的无上下文控制，Score降低0.289%、宽度减少0.404%，96项中35改善、36持平、25退步。全局尺度预测没有增加空间排序信息，最差帧/区域覆盖仍有小幅代价。默认低反例版本相对早期2D参考仍有4/96项退步，最新平均评分候选仍有7/96项退步，因此保留两种取舍；没有单个全指标最优版本。未证明独立泛化或严格/联合时空覆盖。

```bash
pip install -r requirements.txt
TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 python -m reliability.verify --device cpu
python analyze_reliability.py
python -m reliability.predict --input history.pt --output forecast.pt \
  --geometry 0.0034 -0.0034 0.02 --preset context_mean_score
```

几何参数须替换成数据真实的dx、dy、dt。[`results/reliability/`](results/reliability/README.md)包含33,696条完整窗口/通道记录、强参考、全部反例及独立推理核验。新增方案的base/hold/drift/context四种固定策略均保留；384次真实窗口预测的导出实现与冻结研究版本逐位一致。可读说明见[上下文尺度实验](docs/研究结果_上下文预测误差尺度创新_20261109.md)。以下保留的是较早的点预测实验，指标时段与新主线不同，不直接横向计算提升。

---

## 历史点预测版本

Physical-field diffusion forecasting with a matched deterministic four-step control.

给定连续20帧，联合预测紧接的20帧。主模型直接在物理场上生成，不依赖AE/VAE/RAE或额外残差修正网络。每一次网络调用都处理全部未来20帧，未进行10+10或5帧时间分块。

## 历史点预测证据

测试目标帧索引[600,800)，按Re组及u/v等权，原数据MSE：

| 方法 | 推理网络调用 | MSE |
| --- | ---: | ---: |
| 单次状态CNN | 1 | 1.4519220e-4 |
| 确定性终点MSE状态模型 | 4 | 1.4722425e-4 |
| 确定性终点MSE增量模型 | 4 | 1.4772300e-4 |
| 原生去噪训练、零初始噪声4步 | 4 | **1.3455370e-4** |
| 原生去噪训练、32路径均值 | 128 | 1.3474197e-4 |

同4步推理图下，原生去噪训练版本相对确定性状态模型原MSE低8.61%、标准化MSE低8.69%。确定性模型在固定训练窗口拟合更好，在未参与拟合的Re工况上更差。这支持继续研究去噪训练的泛化价值，尚不能证明随机推理必需或具体因果机制。

这些是历史已用数据、固定预算和单训练种子下的探索结果，不是新鲜外部验证或普适优势证明。两种4步模型训练更新数相同，但确定性终点训练每次反传4步，训练FLOP更高。偏平滑/高频细节问题仍存在；区间覆盖是逐点边际量，不是整段同时覆盖。

## 主要代码

- `model.py`：完整20帧U-Net、原生去噪损失和4步采样，参数1,597,672。
- `train.py`：状态/增量Diffusion、单次CNN、确定性4步状态/增量模型，共6个arm。
- `predict.py`：统一推理入口，支持零噪声点预测和32路径样本矩。
- `online_interval.py`：整段20帧真值揭示后才更新下一窗口的区间尺度。
- `runtime.py`：读取只由训练集拟合的文本归一化统计。
- `metadata/`：完整划分、训练归一化和已锁定校准。
- `results/`、`docs/`：分组结果、对照表、原始审计记录和两份研究说明。

旧点预测六个arm的权重、初始化张量、原始数据和预测缓存保存在本地研究压缩包中，未上传到GitHub。新的可靠性主线推理权重已包含于`reliability/checkpoints/current.pt`，与旧六个arm权重不是同一包。历史文档中的原工作区路径/图片引用保留为溯源信息。

用于后续分析的[完整窗口指标、配对提升和同色标图](results/analysis/README.md)位于`results/analysis/`：16,848条窗口/通道记录、11,232条配对比较，保留全部退步项。运行`python analyze_results.py`即可用Python标准库重算分组误差与提升，不需要原始数据或权重。

## 使用

```bash
pip install -r requirements.txt
source env.sh
python train.py --data /path/to/train_real --arm state_diffusion --output retrain_state_diffusion
python train.py --data /path/to/train_real --arm state_fourstep --output retrain_state_fourstep
```

训练仅打开`metadata/manifest.json`中47条train轨迹，核对前200帧哈希；固定seed42、batch8、6000步、1551对窗口，不读取开发、测试或校准文件。训练需CUDA。默认使用独立代码构造器的新鲜seed42权重；它的构造顺序与历史实现不同，因此不承诺直接逐位重现表格。要匹配历史初始化，请从本地研究zip中提供未训练的初始化文件：

```bash
python train.py --data /path/to/train_real --arm state_diffusion --output retrain_state_diffusion \
  --initialization /path/to/archive/metadata/initialization_seed42.pt
```

该初始化不是教师或已拟合预测器。打包时，对6个arm使用归档初始化的首批次损失、梯度和一次更新均与原实现完全一致；完整重训仍可能受GPU/PyTorch内核差异影响。确切原环境在`metadata/environment.json`。

从本地zip复制6份checkpoint到`weights/`，或使用`--weights-dir`指定路径：

```bash
python predict.py --input history.pt --weights-dir /path/to/archive/weights --output prediction.pt
python predict.py --input history.pt --weights-dir /path/to/archive/weights --samples 32 --output ensemble.pt
```

输入为原速度数值尺度float32张量[B,2,20,32,64]。原64×128网格按既定协议`[..., ::2, ::2]`下采样；不要预先重复归一化。CPU推理加`--device cpu`。

默认零噪声4步用于点预测；32路径模式输出均值、标准差及以样本均值为中心的固定90%边际区间。既有校准不能直接换用零噪声中心；其他样本数只输出样本矩，不套用32路径校准。在线尺度实例按轨迹重置，必须先issue预测，再reveal完整20帧真值。

## 数据与复现边界

训练47条、校准10条、开发12条、测试12条轨迹，按Re前缀整体隔离；旧开发/测试工况未进入本轮训练或校准。开发、测试已有研究使用史，本轮无开发集选checkpoint、交叉折、蒸馏或多训练种子。评测是每20帧更新真实观测的窗口预测，不是无反馈800帧自由演化。

`results/*/independent_audit.json`是原工作区的历史审计记录，部分被哈希的大型文件没有上传。完整文件哈希与固定示例核验保存在本地研究zip。AE/VAE额外对照在结果表中保留，相关权重不包含在此代码仓库。
