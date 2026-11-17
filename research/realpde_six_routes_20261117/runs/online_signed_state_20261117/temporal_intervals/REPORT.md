# 同一完整块反馈下的时序任务区间对照

12条已用开发轨迹，每条按0:20:800连续发报41块；所有方法先收到10块完整20帧反馈，再评价31块，共372个窗口。预热120块明确计入协议。它不同于此前dev/late/future逐split冷启动，不能把两者的数值直接混用。全部任务、区间算法、状态模型在评价前冻结。

SPCI行是固定共同点预测器上的条件残差QRF适配：10棵树、深度2、5个beta候选，lag3为预定主设置，lag1/5仅敏感性；使用标准叶权经验分位数代替作者包。AcMCP行是按官方R代码翻译的块h=1递推，含P/I与均值scorecaster；未验证逐帧h=20的MA/跨horizon模块。其KI只来自90个独立校准窗口，残差按冻结训练任务尺度标准化。没有在开发轨迹之间拼接伪历史。

这些方法只产生五个标量任务区间，不生成联合场，也不改变场点MSE。比较覆盖率、训练尺度归一化宽度和90% interval score；低分更好。覆盖率不足不能仅凭区间窄宣称优势。无限/反向区间如实保留，汇总显式传播无穷值。

## 主要发现

同一连续协议下，直接从field_online提取的五任务90%区间宏覆盖只有69.14%，其中最大速度45.97%、网格涡量峰值59.68%、MVPE59.68%。逐场分位校准没有自动保证非线性任务覆盖，这是一条明确失败边界。

AcMCP块h=1宏覆盖90.27%，标准化宽度约为field_online的2.49倍；interval score为.6661对.6737。去掉均值scorecaster的PID对照覆盖91.02%、score .6004；本短块序列没有显示scorecaster的额外优势。SPCI适配主设置覆盖71.08%、score .4087；简单rolling10覆盖74.03%、score .3534。它们的评分虽更低，仍未达到90%目标，不能据此宣称已校准；有限同轨迹历史和适配实现也限制了对原方法的推断。

在相同AcMCP块控制器下，geo_shift/lin_shift的score约.6624/.6626，接近base的.6661；局部静态修正为.6519。标量时序校准的收益不应归因于潜状态机制。相同算法下修正增益的轨迹区间另存same_algorithm_state_gains.csv。

## 原始P为共同中心时的结果

| 区间方法 | 覆盖率% | 标准化宽度 | 标准化interval score | 无限区间% |
|---|---:|---:|---:|---:|
| acmcp_block_h1 | 90.27 | 0.4955 | 0.6661 | 0.00 |
| acmcp_lr005 | 88.87 | 0.4637 | 0.6439 | 0.00 |
| acmcp_lr020 | 91.29 | 0.5691 | 0.7355 | 0.00 |
| field_fixed | 69.89 | 0.1977 | 0.6735 | 0.00 |
| field_online | 69.14 | 0.1992 | 0.6737 | 0.00 |
| field_raw | 69.19 | 0.1975 | 0.6850 | 0.00 |
| pid_no_scorecaster | 91.02 | 0.5032 | 0.6004 | 0.00 |
| rolling10 | 74.03 | 0.1778 | 0.3534 | 0.00 |
| spci_qrf_lag1 | 68.76 | 0.1691 | 0.4025 | 0.00 |
| spci_qrf_lag3 | 71.08 | 0.1690 | 0.4087 | 0.00 |
| spci_qrf_lag5 | 67.53 | 0.1632 | 0.4326 | 0.00 |
| task_fixed | 79.52 | 0.5704 | 1.8611 | 0.00 |

## 同一算法作用于不同点修正

| 点方法 | 区间算法 | 覆盖率% | interval score |
|---|---|---:|---:|
| base | field_online | 69.14 | 0.6737 |
| base | task_fixed | 79.52 | 1.8611 |
| base | rolling10 | 74.03 | 0.3534 |
| base | spci_qrf_lag3 | 71.08 | 0.4087 |
| base | acmcp_block_h1 | 90.27 | 0.6661 |
| geo_shift | field_online | 67.80 | 0.6641 |
| geo_shift | task_fixed | 78.82 | 1.8161 |
| geo_shift | rolling10 | 75.16 | 0.3608 |
| geo_shift | spci_qrf_lag3 | 71.99 | 0.4072 |
| geo_shift | acmcp_block_h1 | 90.05 | 0.6624 |
| lin_shift | field_online | 66.99 | 0.6660 |
| lin_shift | task_fixed | 78.60 | 1.8158 |
| lin_shift | rolling10 | 75.70 | 0.3602 |
| lin_shift | spci_qrf_lag3 | 71.94 | 0.4154 |
| lin_shift | acmcp_block_h1 | 89.95 | 0.6626 |
| local_shift | field_online | 68.23 | 0.6663 |
| local_shift | task_fixed | 78.71 | 1.7785 |
| local_shift | rolling10 | 74.57 | 0.3564 |
| local_shift | spci_qrf_lag3 | 70.86 | 0.4050 |
| local_shift | acmcp_block_h1 | 90.00 | 0.6519 |
| memory_gated | field_online | 72.63 | 0.6488 |
| memory_gated | task_fixed | 82.15 | 1.5632 |
| memory_gated | rolling10 | 73.92 | 0.3674 |
| memory_gated | spci_qrf_lag3 | 71.94 | 0.4234 |
| memory_gated | acmcp_block_h1 | 90.05 | 0.6588 |
| pod_rw | field_online | 68.71 | 0.6720 |
| pod_rw | task_fixed | 79.09 | 1.7650 |
| pod_rw | rolling10 | 74.25 | 0.3543 |
| pod_rw | spci_qrf_lag3 | 71.13 | 0.4110 |
| pod_rw | acmcp_block_h1 | 90.32 | 0.6583 |

## 解释边界

五任务平均值仅是同一尺度下的宏平均；逐任务`task_summary.csv`和逐工况`by_re.csv`同时保留。`paired_interval_gains.csv`按轨迹bootstrap，参考是相同点方法的field_online。所有结果探索性，未作独立外部确认或多重比较校正。标量区间对照的作用是揭示时序校准本身的收益，不能归因于新的物理潜状态，也不能替代整场ES/VS和自由滚动测试。

实际发报/完整反馈日志见`events.jsonl`；保存端点见`issued_intervals.pt`；独立索引AcMCP递推、原端点重放和未来残差干预审计见`audit.json`。源码、协议和下载的官方源码hash见`provenance.json`。这不是原作者依赖栈的逐字节复现，报告不会把适配结果冒充完整SPCI/AcMCP多步基准。
