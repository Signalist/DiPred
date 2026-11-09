# DiPred

最新发布的平均区间评分最优方案是**冻结四步DiPred＋历史误差风险头**。输入连续20帧，一次联合预测完整后20帧；保持99→66→33→0四次调用，直接输出预测误差区间，无需32路采样。

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
