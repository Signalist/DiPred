# 可纠正性研究：代码与复现

[总报告](../runs/recoverability_20261118/REPORT.md)、[20项要求状态](CHECKLIST.md)、[主协议](PROTOCOL.md)、[文献对照](LITERATURE.md)。输出在`runs/recoverability_20261118`。本轮结果是探索性结论，test和跨场景数据有历史使用，底座只有一个seed。

## 从原始RealPDE HDF5重放主表

需要Python、NumPy、pandas、h5py、PyTorch及CUDA GPU；运行环境实际为Python 3.8.10、PyTorch `1.14.0a0+44dac51`（CUDA 12.0）、NumPy 1.22.2、pandas 1.5.2、h5py 3.11.0、A100。此环境中的PyTorch为定制版本；不同CUDA/torch组合可能产生数值差异。scipy 1.6.3、pyarrow 9.0.0用于其他诊断。

1. 获取本仓库`others`分支，以及[官方RealPDE Competition公开训练数据](https://huggingface.co/datasets/AI4Science-WestlakeU/RealPDE-Competition-Data)的`train_real`目录。遵守其许可并排除数据卡标明的重复文件；本实验manifest中已排除`7575_0.h5`。
2. 获取发布的`primary_assets.tar.gz`并解压到新目录，不要覆盖已有实验：

   ```bash
   mkdir -p /path/to/primary_assets
   tar -xzf primary_assets.tar.gz -C /path/to/primary_assets
   ```

3. 在仓库根目录运行（`STUDY`指向`research/realpde_six_routes_20261117`）：

   ```bash
   STUDY=research/realpde_six_routes_20261117
   python "$STUDY/recoverability_20261118/standalone_reproduce.py" \
     --repository . --data /path/to/train_real --assets /path/to/primary_assets \
     --checkpoint reliability/checkpoints/current.pt --out /path/to/output \
     --reference "$STUDY/runs/recoverability_20261118/primary.csv"
   ```

脚本核验主干和冻结资产的SHA-256，从原始HDF5发报、计算三个训练种子的50%路由并生成`raw_windows.csv`/`raw_summary.csv`，可选对比参考表。原环境重放三个收益率的最大差为6.29e-7个百分点。脚本使用冻结权重而不是重训练；原始轨迹不同、设备或算子差异可能使严格误差阈值不适用。不要跳过哈希后声称完全相同的实验。

## 其余代码

`fit.py`训练三个warp/ResCrct头和两个收益/风险路由；`evaluation_recoverability.py`评估；`issue_audit.py`检查不读真值的路由；`cross_scene.py`读取本地RealPDEBench Arrow；`drift_pde.py`及`drift_free.py`运行受控模拟；其余脚本重算诊断和图表。这些脚本保留研究Workspace布局，依赖之前六路线源码、训练/校准缓存及相关权重；与独立重放主表不同，直接执行需要完整受控环境，不能仅凭公开本目录从头重训。前一轮六路线的所有二进制和中间缓存仍未全部打包，见[清单](CHECKLIST.md)。

原`selection.json`记录冻结时文件哈希；之后为了避免与旧`evaluate`模块同名，评估脚本在读取任何本轮评估结果之前改名。该变动不更改训练、选模或冻结权重，后续独立发报重放证实掩码一致。模拟漂移不是现场测量，极值样本只有6个存档窗口，详见报告。
