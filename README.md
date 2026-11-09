# DiPred

Physical-field diffusion forecasting with a matched deterministic four-step control.

给定连续20帧，联合预测紧接的20帧。主模型直接在物理场上生成，不依赖AE/VAE/RAE或额外残差修正网络。每一次网络调用都处理全部未来20帧，未进行10+10或5帧时间分块。

## 当前证据

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

本仓库包含主要代码与轻量结果；权重、初始化张量、原始数据和预测缓存保存在本地研究压缩包中，未上传到GitHub。文档中的原工作区路径/图片引用保留为溯源信息，相关大文件不在仓库内。

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
