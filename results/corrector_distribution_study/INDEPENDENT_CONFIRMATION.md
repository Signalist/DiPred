# 固定 S2 与近期变化模型的独立复验

比较未修正 DiPred、静态 S2、仅依赖近期流场变化的条件尺度。两种校准沿用已冻结参数，保持原生集合均值和逐点成员秩。主比较是条件尺度减静态 S2 的标准化经验 CRPS。每条轨迹先平均四个窗口，再等权平均轨迹，并按原轨迹做 10,000 次配对 bootstrap。Energy Score、覆盖与宽度、时间平均场及五帧增量 CRPS 同时报告。只有一条轨迹时仅给效应大小。

## 数据

输入 1–12 条同任务实测轨迹。HDF5 需包含原尺度的 `u`、`v` 数据，空间网格为 64×128，采样间隔、变量与量纲应与原研究一致。四个零基历史起点为 400、420、440、460，历史和目标各 20 帧，因此至少需要 500 帧。JSON 清单为对象数组，每条对象填写 `path`（绝对路径）、`trajectory_id`、`provenance`（可查证的采集与使用历史），以及确认来源后设为 `true` 的 `unseen_by_backbone_and_method_selection`。这里的未见涵盖主干和 Corrector 的训练、校准、选模及既有结果报告。

锁定步骤核对旧清单、文件 SHA-256 和按旧流程计算的前 200 帧摘要。摘要可排查精确重复。参与史仍需要由数据来源记录证明。锁定后评分入口会再次核对输入、程序和已冻结参数的摘要。

## 执行

在 Corrector 根目录执行。输出目录在锁定前须尚未创建。评估每个窗口时先生成当前 32 条路径并计算两个校准预测，再读取对应目标评分。本轮已有缓存继续只读复用。

~~~bash
source env.sh
export PYTHONPATH="$PWD/../Forecast/Runtime/.deps"
python -m research.corrector_distribution_study.confirm verify
python -m research.corrector_distribution_study.confirm lock /path/to/trajectories.json .cache/corrector_independent_confirm
python -m research.corrector_distribution_study.confirm score .cache/corrector_independent_confirm
~~~

当前仅在两份开发缓存案例上核验了与原实现的预测逐位一致、均值和秩保持，以及既有评分的复现。最大评分绝对差为 2.36×10⁻⁸。独立评分尚无合格输入。[本地数据来源核查](DATA_PROVENANCE.md)
