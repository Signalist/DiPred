# 复现与审查入口

从仓库根目录进入原Workspace环境。原始数据、冻结主干和原始权重文件须保留其工作区位置，详见config.json、data_hashes.json与旧模型来源。所有新缓存只含由历史生成的完整路径和身份，不用残差重建。

人工复跑须为结果和大缓存各建一个新目录，以保留交付：

```bash
source ../../env.sh
export DIPRED_RESTART_OUT="$(mktemp -d "$PWD/results/native_restart_consistency/replay.XXXXXX")"
export DIPRED_RESTART_CACHE="$(mktemp -d "$PWD/../../runs/native_restart_replay.XXXXXX")"
python research/native_restart_consistency/tests.py
python -u research/native_restart_consistency/run.py preflight
python -u research/native_restart_consistency/run.py freeze
python -u research/native_restart_consistency/run.py formal
python research/native_restart_consistency/analyze.py
python research/native_restart_consistency/verify.py
```

程序使用本交付的固定config.json和PROTOCOL.md，不自动追加重启点/种子。默认正式缓存目录为 `Workspace/runs/native_restart_consistency`。若已存在同一正式缓存而重新启动formal，程序会停止并要求审查，避免静默覆盖或重抽随机路径。若资源限制使清单未完成，RESOURCE_STOP.json说明精确停止位置；不能把部分清单当作全实验完成。

只读复核现有交付：在不设置上述两个环境变量时运行verify.py，或从仓库根目录执行 `sha256sum -c results/native_restart_consistency/SHA256SUMS`。核验会读取大缓存和原始HDF5，但不生成新路径、不更新网络。

本次实际命令使用默认输出位置：tests.py → run.py preflight → run.py freeze → run.py formal；后续分析与审查日志为analyze.log、verify.log。tests.log、preflight.log、freeze.log、formal.log保存实际执行记录。新噪声方案的正式路径无法与旧500000系列缓存认定为同一随机身份，因此本轮生成新A/B/C及重启路径，而没有重跑旧后处理头或旧extension评价。

数据源：`DiPred+ResCrct/OLD/REALPDE/realpdebench-competition/data/train_real`，u/v按空间`::2,::2`取32×64；冻结打包checkpoint为`reliability/checkpoints/current.pt`，与`Workspace/runs/physical20_augmented_dynamics_20261109/augmented_diffusion/model.pt`逐张量核对。大缓存与原始数据不随Git上传，代码、配置、逐文件及逐张量哈希、统计和日志可审查。
