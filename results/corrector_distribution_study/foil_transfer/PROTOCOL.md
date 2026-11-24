# RealPDEBench 翼型 PIV 跨采集评估

## 样本与适配

使用 RealPDEBench 2.0.1 的 `foil/hf_dataset/real`，只评估官方 `foil/out_dist_test_params_real.json` 中的实测 OOD 轨迹。依据竞争版原始清单中的文件名 Re 和实测 Re 与攻角配对，排除同工况的 `17812_15.0.h5` 与 `3750_0.0.h5`，保留余下全部八条。按旧数据处理方式计算适配后首 200 帧的速度摘要，核对其在既有清单及本批中的精确重复。该筛查针对可查明的重合，采集来源和更早的使用历史仍有限。

源场为 float32 的 `u/v`。从 128×256 位置网格取每两点一个样本，按坐标反转行序，保持速度分量符号。进入原模型评估前继续沿用原来的两倍空间抽取，因此模型网格为 32×64。源帧间隔 5 毫秒，每四帧取一帧，得到与原任务匹配的 20 毫秒；原四个历史起点 400、420、440、460 按此适配时间轴计数。每个窗口 20 帧历史，随后 20 帧目标。模型网格的间距约相差 2.64%，两批数据的物理观测区域也不同。数值报告应解释为跨采集迁移检验。

## 冻结评价

在读取目标做评分前记录八条轨迹、适配物及配置摘要。沿用冻结原生 DiPred、归一化、32 个成员种子和原来的 S2 与近期变化模型参数。保留未修正 DiPred 对照。主对照为近期变化条件尺度减 S2 的标准化经验 CRPS，先在轨迹内平均四个窗口，再按轨迹配对 bootstrap 10,000 次。Energy Score、均值 MSE、覆盖和宽度、时间平均场及五帧增量 CRPS 同步计算。未利用本批数据校准或选模。

~~~bash
source env.sh
export PYTHONPATH="$PWD/../Forecast/Runtime/.deps"
python -m research.corrector_distribution_study.foil_transfer prepare .cache/corrector_foil_transfer
python -m research.corrector_distribution_study.foil_transfer score .cache/corrector_foil_transfer
~~~

原始 Arrow 与生成的 HDF5 保持本地，交付只包含代码、清单、日志与轻量评分。[RealPDEBench 数据与许可](https://huggingface.co/datasets/AI4Science-WestlakeU/RealPDEBench)
