# 冻结 DiPred 的重启输入归因与 JVP

协议与固定配置位于 `results/restart_input_attribution/`。本目录只做离线输入干预和自动微分诊断，无优化器、权重更新或训练。原任务及解释边界见 `TASK.md`、`PROTOCOL.md`。正式评分以原轨迹为统计单位。

## 文件和执行顺序

- `core.py`：四步可微采样器、只读 k 帧的 oracle 接口、固定子噪声恢复、响应与有符号 MSE 分解。
- `preflight.py`：固定两条训练轨迹，旧输出复现、单位 RMS 方向有限差分、真实增量通道梯度和未来标签污染检查。
- `run.py freeze`：在正式评分前记录协议、代码、配置和预检哈希。
- `run.py formal`：按旧清单顺序生成 R/M，先保存输出再读取评分尾段；S/B 仅来自旧缓存；固定成员 JVP 与半剂量诊断单独计费。
- `analyze.py`：逐轨迹汇总与 10,000 次配对 bootstrap，95% 与三 k Bonferroni 98.333% 区间。
- `verify.py`：从已有缓存核对哈希、帧/成员、噪声、真值读取事件、评分与分解恒等式，不调用采样器或训练。

## 复跑

从仓库根目录执行。需要原始 HDF5 数据、冻结 checkpoint 和原 `native_restart_consistency` 缓存；绝对位置、SHA256 和划分在配置、旧/新缓存 manifest 中。Python/CUDA 环境用已有 `../../env.sh`。一块 A100；不同时运行多个副本。

以下是复跑命令模板，**不会写入本次已归档结果**。请将两个路径指定为尚不存在的目录。程序拒绝覆盖已经开始/完成的运行；勿删除保护文件来重复正式实验。

```bash
source ../../env.sh
export DIPRED_ATTRIBUTION_OUT="$PWD/results/restart_input_attribution_reproduction"
export DIPRED_ATTRIBUTION_CACHE="$(realpath ../../runs)/restart_input_attribution_reproduction"
mkdir -p "$DIPRED_ATTRIBUTION_OUT"
python research/restart_input_attribution/preflight.py > "$DIPRED_ATTRIBUTION_OUT/preflight.log" 2>&1
python research/restart_input_attribution/run.py freeze > "$DIPRED_ATTRIBUTION_OUT/freeze.log" 2>&1
python research/restart_input_attribution/run.py formal > "$DIPRED_ATTRIBUTION_OUT/run.log" 2>&1
python research/restart_input_attribution/analyze.py > "$DIPRED_ATTRIBUTION_OUT/analyze.log" 2>&1
python research/restart_input_attribution/verify.py > "$DIPRED_ATTRIBUTION_OUT/verify.log" 2>&1
```

预检若出现数值失败，不可更改容差：旧路径复现失败会阻止正式执行；仅有限差分失败时主 R/M 控制仍可运行，JVP 被禁用。资源限制中止会写 `RESOURCE_STOP.json`，不会对不完整清单偷偷生成完整统计。

## 指标阅读

`quality.csv` 的 CRPS、ES 和 time_mean_crps 在训练冻结标准化坐标计算；ES 使用完整尾段的 RMS 欧氏距离。mse/variance/width90 在物理尺度，smse/standardized_variance/standardized_width90 是相应标准化量。`lead_channel.csv` 的 energy 是单 lead 空间/通道 ES，完整尾段 ES 只看 quality 表；单 lead 的 time_mean_crps 与该 lead CRPS 相同。

`response_members.csv` 保留所有 32 成员；`response_ensemble.csv` 单独评价集合均值响应。D = S−R，Dm = M−R，Dd = S−M；cross = 2〈Dm,Dd〉，平方量不可解释为正交贡献率。RMS 在各预设块内计算后逐轨迹平均。

`jvp.csv` 的输入/输出都是标准化坐标；每个固定成员包含 full/half、overall/u/v。half 的 input_rms 是 0.5e，误差对照 0.5J e；它只用于剂量诊断，不是概率评分。near_zero 的相对指标留空，独立统计单位仍是原轨迹。

R 读取新观测，仅是离线 oracle；M 是均值前缀干预，不能据此丢弃部署时的成员不确定性。共用噪声并不表示同一物理随机轨迹。本实验不计算耦合组的独立样本 ED_U，不证明 PDE 因果、Markov 充分性或部署收益。

大型缓存保存在 `Workspace/runs/restart_input_attribution/`，不提交 Git；小型结果、日志、协议、配置和 manifest 提交在结果目录。原缓存和权重保持只读。
