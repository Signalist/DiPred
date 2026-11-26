# DiPred 时空残差约束机制与泛化验证

本轮新增 **15 组核心及能量对照训练**、**39 组工况留出训练**。Rule-A 按开发集冻结的全局候选为 **historical_B_ACC_raw_1**。主要机制候选为 **E_TM2_0.3**。全部验证以冻结记录 `2b29fb57a72db474a9693fa5dc16fc30fe65d392e4e0cc8e735dec61e55a14f1` 为准。

## 数学含义

最终预测、真值之差除以公共尺度 (s_R) 为 (e=(widehat R-R)/s_R)。有效差分为线性算子，(D_1widehat r-D_1y=D_1e)，二阶同理。普通二阶平滑惩罚 (D_2r)，目标为预测曲率收缩。真实二阶匹配惩罚预测和真值之差的曲率。逐序列二次型为 (e^	op (I/20+lambda D_2^	op D_2/18)e)，其余维度等权平均。给定输入，设目标条件均值为 (mu)。对固定正定矩阵 (A)，条件风险等于((r-mu)^	op A(r-mu)+operatorname{tr}(Aoperatorname{Cov}(ymid x)))。无限表达能力下最优解仍为条件均值。有限模型中二次型改变误差权重、梯度和优化路径。

输出能量训练目标 (E[(r-y)^2+eta r^2]) 等于 ((1+eta)E[(r-y/(1+eta))^2]+eta E[y^2]/(1+eta))。它相当于对收缩真值重新训练并改变目标和梯度尺度。发布修正强度固定为 1。本报告中的 E 指模型分支，上一窗剩余残差为 (E_{n-1})，当前修正能量为 (U)。

## 主要比较

| 比较 | 6000 步开发 | 6000 步历史测试 | 历史测试描述性区间及改善轨迹 | 开发选中检查点的历史测试 | 判定 |
|---|---:|---:|---|---:|---|
| P1 · E_TM2_0.3 vs E_M0 | +2.515% | +0.690% | [-0.753%, +1.228%]，5/12 条（final） | +0.690% | 部分支持 |
| P2 · E_TM2_0.3 vs E_EN_0.3 | +1.631% | +0.778% | [-0.553%, +1.329%]，5/12 条（final） | +0.778% | 部分支持 |
| P2_U_selected · E_TM2_0.3 vs E_energy_match_1 | +1.010% | +0.914% | [-0.584%, +1.567%]，5/12 条（selected） | +0.914% | 部分支持 |
| P2_U_final · E_TM2_0.3 vs E_energy_match_1 | +1.010% | +0.914% | [-0.584%, +1.567%]，5/12 条（final） | +0.914% | 部分支持 |
| P3 · E_TM2_0.3 vs E_TM1_0.1 | +0.586% | -0.000% | [-1.102%, +0.887%]，4/12 条（final） | +0.269% | 部分支持 |

三项主要假设对应五个具名对照行，P2 的能量匹配按两个检查点口径分别保留。其余为次要比较。配对表共 308 行，17 种比较标签，含各队列及检查点口径。区间由 10,000 次轨迹配对重抽样得到。开发选择和多个 lead 比较使区间只能作描述性解释。

## 时间结构与幅度机制

- B 的 selected 口径采用 B_EN_0.1。训练集 U 相对 B_TM2_0.3 偏差 -3.263%，达到 5% 容差。TM2 相对该 EN 的开发与历史测试 MSE 改善分别为 +1.663%、+0.160%。
- B 的 final 口径采用 B_EN_0.1。训练集 U 相对 B_TM2_0.3 偏差 +3.392%，达到 5% 容差。TM2 相对该 EN 的开发与历史测试 MSE 改善分别为 +1.492%、+0.812%。
- E 的 selected 口径采用 E_energy_match_1。训练集 U 相对 E_TM2_0.3 偏差 -1.145%，达到 5% 容差。TM2 相对该 EN 的开发与历史测试 MSE 改善分别为 +1.010%、+0.914%。
- E 的 final 口径采用 E_energy_match_1。训练集 U 相对 E_TM2_0.3 偏差 -1.145%，达到 5% 容差。TM2 相对该 EN 的开发与历史测试 MSE 改善分别为 +1.010%、+0.914%。
- 跨时刻交叉项的作用。正常 TM2 相对 E_DIAG 的固定步开发和历史测试 MSE 改善为 +2.783%、+1.220%，支持（限所列队列）。三者 λ 相同。初始和训练中的实际梯度尺度见梯度明细。
- 正确时间邻接的作用。正常 TM2 相对 E_PERM 的固定步开发和历史测试 MSE 改善为 +3.331%、+1.506%，支持（限所列队列）。三者 λ 相同。初始和训练中的实际梯度尺度见梯度明细。
- 八批训练探针测得三种同 λ 目标在共同检查点的辅助/MSE 梯度范数比分别为 historical_E_ext_match2 29.74%、E_DIAG 2951.22%、E_PERM 1787.44%。

时间分解使用有效二阶差分矩阵的正交本征向量。前两模式为常数与线性趋势张成的零空间，三段模式在冻结前定义。能量比 (U/E[R^2]) 先聚合二阶矩；逐窗口 RMS 比的平均则先各自开方。全局内积 (C) 未中心化，Pearson 使用中心化协方差并按方差归一化。全局余弦由聚合后的 C、U 和真值能量计算，另单独报告逐窗口余弦均值。

- development · u 通道，E-TM2 相对等能量 EN 的低、中、高模式误差改善为 +0.837%、+1.539%、+2.828%。对应预测能量降幅为 -5.808%、+20.722%、+74.518%。
- development · v 通道，E-TM2 相对等能量 EN 的低、中、高模式误差改善为 +1.444%、+7.127%、+5.491%。对应预测能量降幅为 +27.803%、+51.118%、+89.361%。
- historical_test · u 通道，E-TM2 相对等能量 EN 的低、中、高模式误差改善为 +0.746%、+3.678%、+2.552%。对应预测能量降幅为 -6.452%、+27.383%、+67.041%。
- historical_test · v 通道，E-TM2 相对等能量 EN 的低、中、高模式误差改善为 +0.539%、+1.932%、+3.340%。对应预测能量降幅为 +15.698%、+42.416%、+86.251%。
- development · u 通道，E-TM2 相对 E-M0 的物理 MSE 改善 +2.551%，空间梯度匹配误差变化 -1.804e-06。高误差区覆盖 4.40%，贡献的 MSE 降幅为 +3.133e-06，全通道 MSE 降幅为 +6.799e-06。
- development · v 通道，E-TM2 相对 E-M0 的物理 MSE 改善 +2.155%，空间梯度匹配误差变化 -1.1e-07。高误差区覆盖 3.61%，贡献的 MSE 降幅为 +2.234e-07，全通道 MSE 降幅为 +5.857e-07。
- historical_test · u 通道，E-TM2 相对 E-M0 的物理 MSE 改善 +0.572%，空间梯度匹配误差变化 -6.728e-07。高误差区覆盖 3.68%，贡献的 MSE 降幅为 -8.955e-08，全通道 MSE 降幅为 +1.054e-06。
- historical_test · v 通道，E-TM2 相对 E-M0 的物理 MSE 改善 +1.487%，空间梯度匹配误差变化 -6.568e-08。高误差区覆盖 3.80%，贡献的 MSE 降幅为 +1.012e-07，全通道 MSE 降幅为 +4.07e-07。
- development · E-TM2 相对 E-M0 的 Δ2C=-3.44e-08，ΔU=-3.73e-06，ΔG=+3.69e-06，全局方向余弦变化 +0.0313。
- historical_test · E-TM2 相对 E-M0 的 Δ2C=-1.08e-06，ΔU=-1.81e-06，ΔG=+7.31e-07，全局方向余弦变化 +0.0108。

## 跨分支及联合归因

- S3 EN 固定 6000 步的实际物理 MSE 降幅（ρ=0.3），开发 +6.075e-09，历史测试 -1.642e-07。
- S3 TM1 固定 6000 步的实际物理 MSE 降幅（ρ=0.3），开发 +7.782e-07，历史测试 -3.308e-07。
- S3 TM2 固定 6000 步的实际物理 MSE 降幅（ρ=0.3），开发 +1.079e-06，历史测试 -2.052e-07。
- B EN 固定 6000 步的实际物理 MSE 降幅（ρ=0.3），开发 -8.515e-07，历史测试 -6.436e-07。
- B TM1 固定 6000 步的实际物理 MSE 降幅（ρ=0.3），开发 -2.504e-07，历史测试 +8.573e-07。
- B TM2 固定 6000 步的实际物理 MSE 降幅（ρ=0.3），开发 +1.756e-06，历史测试 +4.878e-07。
- E EN 固定 6000 步的实际物理 MSE 降幅（ρ=0.3），开发 +1.319e-06，历史测试 -9.335e-08。
- E TM1 固定 6000 步的实际物理 MSE 降幅（ρ=0.3），开发 +2.177e-06，历史测试 +6.076e-07。
- E TM2 固定 6000 步的实际物理 MSE 降幅（ρ=0.3），开发 +3.693e-06，历史测试 +7.307e-07。

分支内部配对控制了输入、布局、步数及初始化。跨分支差异同时包含输入和模型条件，因此 E 的新增历史预测残差是否单独造成效应差异仍为部分支持。
- B · development，保持四项组成调权后改善 +1.411%，在相同四权重上增加 MATCH 后再改变 +0.095%。
- B · historical_test，保持四项组成调权后改善 +1.557%，在相同四权重上增加 MATCH 后再改变 -0.276%。
- E · development，保持四项组成调权后改善 +4.050%，在相同四权重上增加 MATCH 后再改变 +1.578%。
- E · historical_test，保持四项组成调权后改善 +1.265%，在相同四权重上增加 MATCH 后再改变 +0.706%。
- E 的 MAG × MATCH 配对交互在 development 为 -3.061e-06 原尺度 MSE；正值表示联合的净改善超过两单项净改善之和。
- E 的 MAG × MATCH 配对交互在 historical_test 为 +4.066e-07 原尺度 MSE；正值表示联合的净改善超过两单项净改善之和。

## 开发集选模及历史失败

- 旧冻结 B-ACC · development · selected，相对自身纯 MSE 改善 +1.707%，9/12 条轨迹改善。
- 旧冻结 B-ACC · development · final，相对自身纯 MSE 改善 +1.620%，10/12 条轨迹改善。
- 旧冻结 B-ACC · historical_test · selected，相对自身纯 MSE 改善 -1.157%，3/12 条轨迹改善。
- 旧冻结 B-ACC · historical_test · final，相对自身纯 MSE 改善 -0.357%，5/12 条轨迹改善。
- 旧研究有 1 个开发选中点处于共享 MSE 前缀。旧结果与本轮重算的历史锚点身份均已保留。
- historical_B_ACC_raw_1 的开发正向降幅中，最大两条轨迹贡献 94.9%，26700_15.h5（Re 26761）、26700_10.h5（Re 26761）。
- E_TM2_0.3 的开发正向降幅中，最大两条轨迹贡献 87.5%，26700_10.h5（Re 26761）、26700_15.h5（Re 26761）。
- B · historical_grid 的 Rule-A 留一轨迹选择频率为 B_ACC_raw_1 10/12、B_JOINT_top2 1/12、B_VEL_rho_0.1 1/12。留出改善率的均值为 +0.312%。
- B · core 的 Rule-A 留一轨迹选择频率为 B_TS1_0.1 11/12、B_TM2_0.3 1/12。留出改善率的均值为 +0.182%。
- B · expanded 的 Rule-A 留一轨迹选择频率为 historical_B_ACC_raw_1 11/12、B_TS1_0.1 1/12。留出改善率的均值为 +0.379%。
- E · historical_grid 的 Rule-A 留一轨迹选择频率为 E_ext_match2 11/12、E_JOINT_tuned 1/12。留出改善率的均值为 +0.558%。
- E · core 的 Rule-A 留一轨迹选择频率为 E_TM2_0.3 11/12、E_J5 1/12。留出改善率的均值为 +0.558%。
- E · expanded 的 Rule-A 留一轨迹选择频率为 E_TM2_0.3 11/12、E_J5 1/12。留出改善率的均值为 +0.558%。
- S3 · core 的 Rule-A 留一轨迹选择频率为 S3_TM2_0.3 12/12。留出改善率的均值为 +0.329%。
- S3 · expanded 的 Rule-A 留一轨迹选择频率为 S3_TM2_0.3 12/12。留出改善率的均值为 +0.329%。

开发留一和影子规则用于敏感性诊断。Rule-A 仍为原定主规则，Rule-B、Rule-C 的模型已在验证前冻结。

## 工况迁移、时段扩展与连续反馈

- Re 17812 留出 4 条轨迹。B-ACC、B-MAG、B-TM2、E-EN、E-TM1、E-TM2 相对各自纯 MSE 的冻结步改善依次为 +0.160%、+0.379%、+0.674%、+1.014%、+0.807%、+1.297%。
- Re 25482 留出 4 条轨迹。B-ACC、B-MAG、B-TM2、E-EN、E-TM1、E-TM2 相对各自纯 MSE 的冻结步改善依次为 +0.596%、+0.655%、+0.499%、+1.879%、+1.172%、+1.692%。
- Re 8863 留出 5 条轨迹。B-ACC、B-MAG、B-TM2、E-EN、E-TM1、E-TM2 相对各自纯 MSE 的冻结步改善依次为 +0.053%、-0.231%、-0.365%、-0.003%、-0.146%、+0.020%。
冻结主干的训练清单覆盖三组全部留出轨迹。每组 4 个 OOF 参考和 1 个完整子训练 S3 的拟合身份均与外层隔离。沿用旧测试的连续 96 个窗口，其中 48 窗位于旧窗口研究的最晚起点之后。这些是同轨迹时段扩展。依据本地同任务清单，没有满足独立确认条件的合法轨迹。

- E-M0 自身反馈相对固定参考的物理 MSE 改善为 -0.565%，相对 Native 为 +3.407%。主干在每次发布时用新到达的观测窗口。
- E-EN 自身反馈相对固定参考的物理 MSE 改善为 +0.178%，相对 Native 为 +4.488%。主干在每次发布时用新到达的观测窗口。
- E-TM1 自身反馈相对固定参考的物理 MSE 改善为 -0.529%，相对 Native 为 +4.406%。主干在每次发布时用新到达的观测窗口。
- E-TM2 自身反馈相对固定参考的物理 MSE 改善为 -0.194%，相对 Native 为 +4.647%。主干在每次发布时用新到达的观测窗口。

## 保留建议与限制

| 数据 | 候选 | 物理 MSE | 相对 Native | 相对 S3 | 相对自身纯 MSE |
|---|---|---:|---:|---:|---:|
| development | historical_B_ACC_raw_1 | 0.00014221008 | +7.681% | +2.514% | +1.707% |
| development | B_TM2_0.3 | 0.000142675034 | +7.379% | +2.195% | +1.386% |
| development | E_TM2_0.3 | 0.000143147323 | +7.072% | +1.872% | +2.515% |
| historical_test | historical_B_ACC_raw_1 | 0.000107629563 | +3.568% | -1.083% | -1.157% |
| historical_test | B_TM2_0.3 | 0.000106607638 | +4.484% | -0.123% | -0.196% |
| historical_test | E_TM2_0.3 | 0.00010514158 | +5.798% | +1.253% | +0.690% |
| period_new | historical_B_ACC_raw_1 | 0.000114262349 | +2.836% | -1.123% | +0.295% |
| period_new | B_TM2_0.3 | 0.000114645097 | +2.511% | -1.461% | -0.039% |
| period_new | E_TM2_0.3 | 0.000113003546 | +3.907% | -0.009% | +1.471% |

E-TM2 的主队列结果为部分支持，工况留出为支持（限所列队列）。保留冻结主规则结果及 E-TM2 作为预先指定的机制候选。正式泛化结论仍需独立同任务轨迹。历史测试反复使用、旧开发参与选择、单训练种子、主干见过外层轨迹和逐帧多重比较，均限制了论文中的强主张。

## 产物

- [PROTOCOL.md](PROTOCOL.md)
- [hypotheses.json](hypotheses.json)
- [experiment_plan.json](experiment_plan.json)
- [data_roles.json](data_roles.json)
- [sources.json](sources.json)
- [reuse_audit.json](reuse_audit.json)
- [FROZEN.json](FROZEN.json)
- [selections.json](selections.json)
- [scores.csv](scores.csv)
- [leads.csv](leads.csv)
- [channels.csv](channels.csv)
- [trajectories.csv](trajectories.csv)
- [regimes.csv](regimes.csv)
- [contrasts.csv](contrasts.csv)
- [lead_primary_contrasts.csv](lead_primary_contrasts.csv)
- [verification.json](verification.json)
- [artifact_audit.json](artifact_audit.json)
- [reference_lineage.json](reference_lineage.json)
- [exploration_log.md](exploration_log.md)
- [图 01](<../../.User/corrector_spacetime_validation/01_mechanism_matrix.png>)
- [图 02](<../../.User/corrector_spacetime_validation/02_all_twenty_leads.png>)
- [图 03](<../../.User/corrector_spacetime_validation/03_train_energy_matching.png>)
- [图 04](<../../.User/corrector_spacetime_validation/04_diagonal_and_permuted.png>)
- [图 05](<../../.User/corrector_spacetime_validation/05_cross_branch_absolute_gain.png>)
- [图 06](<../../.User/corrector_spacetime_validation/06_joint_attribution.png>)
- [图 07](<../../.User/corrector_spacetime_validation/07_gradient_training_dynamics.png>)
- [图 08](<../../.User/corrector_spacetime_validation/08_finite_difference_modes.png>)
- [图 09](<../../.User/corrector_spacetime_validation/09_regime_trajectory_channel.png>)
- [图 10](<../../.User/corrector_spacetime_validation/10_selection_sensitivity.png>)
- [图 11](<../../.User/corrector_spacetime_validation/11_regime_holdout_and_new_period.png>)
- [图 12](<../../.User/corrector_spacetime_validation/12_consecutive_feedback.png>)
- [图 13](<../../.User/corrector_spacetime_validation/13_fixed_physical_case.png>)
