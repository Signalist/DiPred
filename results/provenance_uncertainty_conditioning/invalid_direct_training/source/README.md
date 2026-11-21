# DiPred 来源与父集合不确定性条件化

本目录实现三臂等容量的历史条件适配器。原生20→20四步Diffusion、父生成器与归一化冻结；仅训练1760参数适配器。任务、不可变协议、配置及数据划分位于 `results/provenance_uncertainty_conditioning/`。

## 入口

- `core.py`：三臂、保留完整图的四步采样、发布条件、完整32成员CRPS、checkpoint及哈希工具。
- `tests.py`：训练前数值/梯度/成员/物理时间/未来标签污染及零初始化复现。
- `train.py`：同步九次训练、每250步完整开发评价、科学和保护候选的独立选择。
- `metrics.py`：完整概率评分、接缝、时间ACF、跨成员中心化时间协方差和独立B/C采样基线。
- `test_diagnostics.py`：仅训练数据上的评价实现检查。
- `evaluate.py`：候选冻结后才执行历史测试和固定20个后续窗口，先发布后读取评分标签。
- `analyze.py`：原轨迹内平均窗口/重复及种子，轨迹配对bootstrap；保留每seed、Re、通道、lead、k及窗口。
- `development_analysis.py`：披露所有36个非零开发候选及同一步数臂间配对；不改冻结选择。
- `check_plan.py`：只读元数据核对528组覆盖、Re隔离和全部噪声种子唯一。
- `verify.py`：九训练/45checkpoint/18选择、全部528发布组身份及大明细、固定保存子集的评分和模型复现。

## 复跑命令

从DiPred仓库根目录运行。使用一块A100，已有依赖通过 `../../env.sh` 配置。需要原始HDF5、原checkpoint和 `native_restart_consistency` 的旧缓存；准确路径与SHA256在config/data_sources/native旧manifest中。不得把500000系列缓存替代本协议身份。

默认输出已经归档，程序拒绝覆盖已有开始/完成记录。完整重现请使用**全新**结果及缓存路径：

```bash
source ../../env.sh
export DIPRED_CONDITION_OUT="$PWD/results/provenance_uncertainty_conditioning_reproduction"
export DIPRED_CONDITION_CACHE="$(realpath ../../runs)/provenance_uncertainty_conditioning_reproduction"
mkdir -p "$DIPRED_CONDITION_OUT"
python research/provenance_uncertainty_conditioning/tests.py > "$DIPRED_CONDITION_OUT/tests.log" 2>&1
python research/provenance_uncertainty_conditioning/test_diagnostics.py > "$DIPRED_CONDITION_OUT/evaluation_tests.log" 2>&1
python research/provenance_uncertainty_conditioning/check_plan.py > "$DIPRED_CONDITION_OUT/plan_checks.log" 2>&1
python research/provenance_uncertainty_conditioning/train.py > "$DIPRED_CONDITION_OUT/train.log" 2>&1
python research/provenance_uncertainty_conditioning/development_analysis.py > "$DIPRED_CONDITION_OUT/development_analysis.log" 2>&1
python research/provenance_uncertainty_conditioning/evaluate.py > "$DIPRED_CONDITION_OUT/evaluate.log" 2>&1
python research/provenance_uncertainty_conditioning/analyze.py > "$DIPRED_CONDITION_OUT/analyze.log" 2>&1
python research/provenance_uncertainty_conditioning/verify.py > "$DIPRED_CONDITION_OUT/verify.log" 2>&1
```

`tests.py`记录其执行时源码哈希；之后应保持源码不变。此归档的训练代码在TRAINING_FROZEN.json中冻结；评价/分析/审计代码另在正式测试前的EVALUATION_FROZEN.json中冻结。开发选择的权重在FROZEN_SELECTION.json中冻结，早于测试发布事件。新复跑时所有源文件均已存在，哈希应自然包含全套文件。

训练全部使用32成员经验CRPS，不存在小成员近似或省略成员距离项。每步训练一个轨迹/窗口/重复/条件块，九模型抽样与噪声完全一致；训练种子仅控制共同对齐的适配器初始化。父生成器不更新，成员不平均后再重启。

## 表格约定

`policy=scientific`按开发重启尾段CRPS选择，包括step0；`guarded`额外要求每k尾段和全观测直接预测的原/标准化MSE均不超过相应原生开发均值1%。这是预设描述性保护界，不是统计非劣证明。step0重启=原S，直接=原B。

`view=tail`为子路径前20−k、对齐原B后20−k；`joint`为父前k＋同成员子前20−k，完整20帧；`direct`为原真实20帧历史的完整直接输出，k=0。时间来源mask只由k决定；std槽是父集合逐点总体离差除冻结通道std，观测位置为0。

crps、energy、smse、standardized_variance、standardized_width90、time_mean_crps用冻结通道标准化；mse、variance、width90保留物理尺度。Energy Score覆盖对应view完整路径，lead表中的energy仅该lead。每条路径先沿时间平均后计算time_mean_crps，不能用逐lead平均替代。

`seam_rms`与`seam_increment_mse`在joint视图的k−1→k定义，后者是标准化的逐成员增量对真值增量MSE。ACF去通道总体均值；时间协方差先按成员中心化再在成员与空间上平均。全部矩阵、ACF在缓存的temporal文件中，由evaluation_manifest.jsonl索引。

restart_excess_ed_u=ED_U(B_tail,预测尾段)−ED_U(B_tail,C_tail)，B/C独立于父/子及彼此。B对自身的U留空，不冒充独立采样；共享子噪声的适配器与S之间只做真实评分及配对统计，不使用独立U公式。

大明细以CSV.gz保存，可直接 `gzip -dc` 或Python `gzip.open`读取。配对差负值表示前者指标数值较小；方差、宽度、协方差或接缝幅度的下降本身不等于改善。`seed=mean3`先在同一原轨迹内平均三种子，再bootstrap，独立样本数仍为12。

新extension原生缓存与固定审计适配输出在Workspace/runs，未进Git。其余适配输出保留完整哈希、条件/噪声身份、评分及矩阵，可以从小型冻结checkpoint重现。大缓存无需下载到Git；原生父+子两次四步的成本必须同时披露。

首次启动曾因Python3.8不支持Path.is_relative_to在第一个零步checkpoint元数据写入时终止，优化更新为0。所有失败产物在preupdate_failure/保留；仅修复路径兼容性后重新预检，没有修改协议或重复正式优化。
