# 预期收益分配计算：固定预算下出现可复核的任务收益

主要动作是廉价AR+PCA与昂贵32路径smooth之间选择；主要指标预先固定为五个训练尺度标准化物理泛函的平均经验CRPS。**50%昂贵调用预算下，预计收益策略在三个时段的均值都优于同预算的随机期望、预测风险和区间宽度分配；相对随机期望，development和late的轨迹级95%区间仍跨零。** 这是一项同步多请求服务实验；不等价于单轨迹在线调度，也不是新的独立数据确认。

|说明|结果|
|---|---|
|训练|92窗口/46轨迹；原RealPDE、seed42；标签=廉价损失−昂贵损失|
|模型|X+P+F_last，训练PCA 16维，带截距17个监督读出权重，ridge alpha=0.01；PCA/标准化参数另存，不冒充总共17参数|
|选择|90校准窗口/10轨迹，在25/50/75%预算的平均策略损失上选择；随后冻结全部动作掩码才读取评估评分|
|评估|development108、late120、future72窗口，同12开发轨迹；逐轨迹平均和配对区间|
|信息|共同四步P/F与49边际头；路由只读已观测X和当前P/F/宽度；新增路径仅在选中请求执行|

## 同预算比较

正数表示预计收益路由降低主要评分。CI是基线−本策略的绝对评分差，按轨迹重采样10000次；它描述固定策略/实际请求批次，未包含拟合/选模不确定性，未将窗口或像素当独立样本。

| split | baseline | gain_percent | ci_low | ci_high | wins | trajectories |
| --- | --- | --- | --- | --- | --- | --- |
| development | random_expected | 1.60475 | -0.000112214 | 0.0025276 | 9 | 12 |
| development | risk | 4.33593 | 0.00111843 | 0.00565563 | 11 | 12 |
| development | width | 3.92501 | 0.000173009 | 0.00575619 | 8 | 12 |
| late | random_expected | 2.22584 | -6.78443e-05 | 0.00304413 | 8 | 12 |
| late | risk | 4.35405 | 0.000292118 | 0.00576596 | 9 | 12 |
| late | width | 4.43378 | 1.42126e-05 | 0.00602719 | 8 | 12 |
| future | random_expected | 2.77569 | 0.000269548 | 0.00384065 | 10 | 12 |
| future | risk | 5.26976 | 0.000738198 | 0.00753839 | 9 | 12 |
| future | width | 5.96283 | 0.00106252 | 0.00798891 | 9 | 12 |

完整25/50/75%结果见`paired.csv`，全部预算曲线如下。0/100%为共同端点。50%不是查看开发结果后选出的最佳预算；预先指定的全部预算均保留。随机期望由固定配额下每请求被选概率精确计算，不是多训练种子结果；另保留单seed42随机实现。

![预算—任务损失](figures/allocation.png)

## 代价与强对照

与始终使用昂贵动作比较：

| split | baseline | main_half_budget_gain_percent |
| --- | --- | --- |
| development | always_smooth | 0.891738 |
| development | always_secc3 | 0.679751 |
| late | always_smooth | 1.45712 |
| late | always_secc3 | -0.0017381 |
| future | always_smooth | 3.5613 |
| future | always_secc3 | 2.25552 |

最大速度仍有代价，不能把宏平均收益说成所有任务都改善：

| split | metric | gain_vs_full_percent | gain_vs_equal_random_percent |
| --- | --- | --- | --- |
| development | functional_crps | 0.891738 | 1.60475 |
| development | crps_speed_max | -8.28906 | 1.03216 |
| late | functional_crps | 1.45712 | 2.22584 |
| late | crps_speed_max | -7.65354 | 0.777539 |
| future | functional_crps | 3.5613 | 2.77569 |
| future | crps_speed_max | -3.71166 | 3.30553 |

把同一冻结掩码的昂贵动作换成S-ECC3的敏感性分析（未重训该收益模型）：

| split | gain_percent | ci_low | ci_high | wins |
| --- | --- | --- | --- | --- |
| development | 2.03143 | 2.5496e-05 | 0.00276491 | 9 |
| late | 1.44377 | -0.000392611 | 0.00239777 | 8 |
| future | 2.85635 | 0.000395893 | 0.00350765 | 11 |

逐个任务、Re组、全部表示和全部失败配置分别保留在`task_gains.csv`、`by_re.csv`、`representations.csv`、`selection_grid.csv`。S-ECC3完整端点也在图中，不能只与较弱的smooth比较。

相同预算下的收益并非每项任务共同改善。平均收益包含涡量和MVPE方面的改善；TKE、动能或VS在部分时段变差：

| split | metric | gain_vs_equal_random_percent |
| --- | --- | --- |
| development | functional_crps | 1.60475 |
| development | es | -0.000489671 |
| development | vs | -0.166988 |
| development | marginal_crps | -2.34809e-07 |
| development | crps_tke_peak | -4.007 |
| development | crps_kinetic_mean | 2.11308 |
| development | crps_enstrophy_grid_peak | 2.94829 |
| development | crps_speed_max | 1.03216 |
| development | crps_mvpe | 1.8153 |
| late | functional_crps | 2.22584 |
| late | es | 0.00852261 |
| late | vs | -0.408068 |
| late | marginal_crps | 2.16111e-07 |
| late | crps_tke_peak | -1.13303 |
| late | crps_kinetic_mean | -5.37738 |
| late | crps_enstrophy_grid_peak | 11.3204 |
| late | crps_speed_max | 0.777539 |
| late | crps_mvpe | 1.9272 |
| future | functional_crps | 2.77569 |
| future | es | 0.0133194 |
| future | vs | -0.21163 |
| future | marginal_crps | 1.59555e-07 |
| future | crps_tke_peak | 1.09971 |
| future | crps_kinetic_mean | 0.794242 |
| future | crps_enstrophy_grid_peak | 6.02387 |
| future | crps_speed_max | 3.30553 |
| future | crps_mvpe | 2.33917 |

逐Re均值（仍不是独立的新工况确认）：

| split | group | benefit | random_expected | gain_percent |
| --- | --- | --- | --- | --- |
| development | 6300 | 0.0257273 | 0.0262547 | 2.00849 |
| development | 7575 | 0.0425178 | 0.0427242 | 0.483152 |
| development | 26700 | 0.170412 | 0.173662 | 1.87093 |
| future | 6300 | 0.0291027 | 0.0301448 | 3.45711 |
| future | 7575 | 0.0337628 | 0.0343829 | 1.80351 |
| future | 26700 | 0.159312 | 0.163966 | 2.8387 |
| late | 6300 | 0.0272309 | 0.028136 | 3.21668 |
| late | 7575 | 0.037128 | 0.0372104 | 0.221482 |
| late | 26700 | 0.147779 | 0.151685 | 2.57511 |

预测风险与真实动作收益的Spearman相关在三个时段为负，而收益读出为正；这是本数据中风险排名无法替代收益排名的直接诊断，不是一般定理：

| split | policy | spearman_with_true_benefit | spearman_with_cheap_loss |
| --- | --- | --- | --- |
| development | benefit | 0.206589 | -0.19429 |
| development | risk | -0.445759 | 0.492822 |
| development | width | -0.300609 | 0.594777 |
| late | benefit | 0.302521 | -0.444698 |
| late | risk | -0.405348 | 0.491319 |
| late | width | -0.336954 | 0.507903 |
| future | benefit | 0.355875 | -0.347032 |
| future | risk | -0.363464 | 0.635346 |
| future | width | -0.48032 | 0.539359 |

## 校准与表示

同一动作掩码用于raw/fixed/online共同边际校准，避免把改变校准归功于路由。50%预算、相对同预算随机期望：

| split | mode | gain_percent | ci_low | ci_high |
| --- | --- | --- | --- | --- |
| development | raw | 1.60475 | -0.000112214 | 0.0025276 |
| development | fixed | 1.68506 | -7.15679e-05 | 0.00262259 |
| development | online | 1.78162 | -8.40124e-05 | 0.00287686 |
| late | raw | 2.22584 | -6.78443e-05 | 0.00304413 |
| late | fixed | 2.26391 | -8.79272e-05 | 0.00311588 |
| late | online | 2.47643 | -1.86932e-05 | 0.00339919 |
| future | raw | 2.77569 | 0.000269548 | 0.00384065 |
| future | fixed | 2.89283 | 0.000337832 | 0.0039247 |
| future | online | 2.99973 | 0.000216783 | 0.00443063 |

等监督维度网格的X/P/XP/F读出全部保留：

| split | representation | dimensions | alpha | functional_crps |
| --- | --- | --- | --- | --- |
| development | X | 4 | 0.01 | 0.0684616 |
| development | P | 16 | 0.01 | 0.0680481 |
| development | X+P | 16 | 0.01 | 0.067941 |
| development | F_last | 16 | 0.01 | 0.0679651 |
| development | F_mid0 | 16 | 0.01 | 0.0684362 |
| development | X+P+F_last | 16 | 0.01 | 0.0674954 |
| late | X | 4 | 0.01 | 0.0614688 |
| late | P | 16 | 0.01 | 0.0615257 |
| late | X+P | 16 | 0.01 | 0.0611283 |
| late | F_last | 16 | 0.01 | 0.0616063 |
| late | F_mid0 | 16 | 0.01 | 0.061492 |
| late | X+P+F_last | 16 | 0.01 | 0.0606671 |
| future | X | 4 | 0.01 | 0.0645341 |
| future | P | 16 | 0.01 | 0.0640958 |
| future | X+P | 16 | 0.01 | 0.063756 |
| future | F_last | 16 | 0.01 | 0.0643065 |
| future | F_mid0 | 16 | 0.01 | 0.0645636 |
| future | X+P+F_last | 16 | 0.01 | 0.0632083 |

本实验支持冻结预测器表示可用于简单收益读出，不证明扩散F增加了新观测信息或有独有优势。同四调用确定性表征与小标注对照已有`../error_structure_20261117/REPORT.md`中的限制；这里没有重新证明不同训练目标的因果优势。

## 实际分发成本

| policy | quota | actual_fraction | requests | median_batch_ms | median_ms_per_request | p10_batch_ms | p90_batch_ms | peak_mb | repeats |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| benefit | 0 | 0 | 12 | 453.312 | 37.776 | 452.19 | 454.684 | 369.766 | 7 |
| benefit | 0.25 | 0.25 | 12 | 1130.48 | 94.2071 | 1125.05 | 1137.13 | 369.766 | 7 |
| benefit | 0.5 | 0.5 | 12 | 1811.44 | 150.953 | 1797.94 | 1826.53 | 376.844 | 7 |
| benefit | 0.75 | 0.75 | 12 | 2475.51 | 206.293 | 2461.58 | 3033.73 | 377.715 | 7 |
| benefit | 1 | 1 | 12 | 3172.68 | 264.39 | 3129.67 | 3204.74 | 380.586 | 7 |
| risk | 0.5 | 0.5 | 12 | 1815.69 | 151.308 | 1803.84 | 2145.96 | 369.766 | 7 |
| width | 0.5 | 0.5 | 12 | 1808.62 | 150.718 | 1793.99 | 2766.89 | 369.766 | 7 |
| random | 0.5 | 0.5 | 12 | 1809.94 | 150.828 | 1803.44 | 1840.41 | 377.715 | 7 |

预载模型，A100，12个同时到达请求串行GPU执行，包含共同预测、特征提取、路由、所选联合生成和保存GPU输出；不含磁盘与评分。七轮按seed42随机顺序交错测量，完整原始耗时与测量次序见timing_raw.json；初始顺序计时保留作诊断。记录总批次延迟及其除以12的平均服务成本，后者不能称为单用户响应延迟。昂贵分支复用共同P/F/边际，不重新运行49头；不先生成随后丢弃的廉价样本。比较相同配额策略实测成本，非只计主干调用数。

## 审计与边界

`accounting_audit.json`独立用pandas重算top-k配额、逐指标组合评分，并枚举首批次所有半预算子集核对随机期望。`dispatch_audit.json`直接从观测历史分发预测，检查与冻结掩码及缓存评分一致。上游场评分、边际相同、物理坐标和合法延迟校准沿用已完成数值审计。

廉价生成器和收益监督都用了同92训练窗口，因此训练动作标签存在拟合内乐观风险；真正比较使用轨迹分离的cal/development，后者有历史复用。分配是在同相对发行时刻的多轨迹批次中进行，不声称这些实验在现实日历同时发生，不排名未来窗口；跨轨迹资源竞争也意味着bootstrap仅作描述性证据。单seed符合当前小数据探索范围。

本路线已有具体用途收益，仍需要新轨迹/新工况独立确认。整体六路线目标仍active；在线有符号状态、主动观测与经验物理任务误差估计继续推进。原主干、默认推理和外部仓库未替换。
