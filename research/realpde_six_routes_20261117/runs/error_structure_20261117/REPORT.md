# RealPDE 有符号结构修正：阶段结果，完整目标仍在推进

本阶段得到一个小型可复现候选：warp_F_last（2967参数），直接移动/修正预测未来场。它在校准集从预定候选中选定，之后冻结。此结果仅支持有完整真实历史的20→20预测；无反馈长滚动存在退步。尚未完成附件中的全部六条研究路线。

## 诊断与方法

所有候选使用同一当前发布的四步 DynamicsDiffusion20、FP32、seed42。修正训练819窗口/46轨迹，主干原训练47轨迹前200帧；修正使用这些训练身份的较晚窗口。校准90窗口/10轨迹，开发108窗口/12轨迹，轨迹和Re与训练分离。late120、future72仍是同12开发轨迹，只是时间延伸。所有这些数据有历史研究使用记录，不能称新鲜独立确认。

下面的oracle读取真实未来，属于优化找到的诊断参照，不是可部署预测，也不保证达到非凸参数族的全局最优。位移限制±1网格、相位±1帧、幅值±25%、偏置±0.25归一化单位。局部56参数在2×4网格平滑插值；正向x/y位移是数组索引坐标，不能直接解释为物理流速方向。

| method | mse | gain_percent |
| --- | --- | --- |
| base | 9.32603e-05 | 1.11022e-14 |
| tangent_oracle | 8.83431e-05 | 5.27259 |
| warp_oracle | 8.82637e-05 | 5.35773 |
| four_segment_oracle | 8.61187e-05 | 7.65779 |
| amplitude | 8.50248e-05 | 8.83073 |
| combined | 7.57579e-05 | 18.7673 |
| geometry | 8.08994e-05 | 13.2542 |

局部训练头采用逐位置训练PCA32维、16隐藏通道、三层卷积。结构头2967参数，普通加性头2984参数，均2000步、batch8、同种子和数据顺序；仅用校准选修正强度。加性头输出2×4×2×4平滑残差，结构头输出7×2×4系数。同一P、同一信息时刻。global7另有54个岭回归/模态对照，完整结果见development_summary.csv和selection_grid.csv；不能将多个变体当成独立重复实验。

## 轨迹级结果

正百分比表示降低物理速度MSE。CI为基线MSE减候选MSE的绝对差，按完整轨迹重采样10000次；是单种子、重复使用数据上的描述性区间，同Re轨迹相关，不支持新Re总体置信保证。

| split | method | gain_percent | wins | n | ci_low | ci_high |
| --- | --- | --- | --- | --- | --- | --- |
| development | additive_F_last | 0.0356674 | 3 | 12 | -3.52963e-07 | 7.28707e-07 |
| development | warp_F_det4_1500 | 0.852061 | 11 | 12 | 3.16396e-07 | 2.92283e-06 |
| development | warp_F_det4_6000 | 0.703199 | 12 | 12 | 2.89811e-07 | 2.37967e-06 |
| development | warp_F_last | 0.791065 | 12 | 12 | 4.59071e-07 | 2.47456e-06 |
| development | warp_X_P | 0.319875 | 11 | 12 | -8.19064e-08 | 1.4793e-06 |
| late | additive_F_last | -0.238239 | 3 | 12 | -9.22464e-07 | 7.46266e-08 |
| late | warp_F_det4_1500 | 0.778508 | 12 | 12 | 2.79602e-07 | 2.44683e-06 |
| late | warp_F_det4_6000 | 0.607651 | 12 | 12 | 2.42212e-07 | 1.88139e-06 |
| late | warp_F_last | 0.721686 | 12 | 12 | 4.15207e-07 | 2.00638e-06 |
| late | warp_X_P | 0.542447 | 11 | 12 | 1.74442e-08 | 1.94822e-06 |
| future | additive_F_last | -0.52615 | 2 | 12 | -1.67902e-06 | -2.35949e-07 |
| future | warp_F_det4_1500 | 0.536689 | 11 | 12 | 2.91935e-07 | 1.6371e-06 |
| future | warp_F_det4_6000 | 0.247314 | 11 | 12 | 1.38599e-07 | 8.12637e-07 |
| future | warp_F_last | 0.45264 | 12 | 12 | 3.70456e-07 | 1.18391e-06 |
| future | warp_X_P | 0.318256 | 11 | 12 | 7.18366e-08 | 1.12809e-06 |

相对强控制的配对比较：

| split | reference | gain_percent | wins | ci_low | ci_high |
| --- | --- | --- | --- | --- | --- |
| development | warp_X_P | 0.472702 | 12 | 3.47468e-07 | 1.36396e-06 |
| development | warp_F_det4_1500 | -0.0615196 | 10 | -4.89449e-07 | 1.54943e-07 |
| development | warp_F_det4_6000 | 0.088489 | 11 | 1.73803e-09 | 2.79954e-07 |
| development | additive_F_last | 0.755667 | 12 | 7.42056e-07 | 1.94663e-06 |
| late | warp_X_P | 0.180216 | 10 | -1.13406e-08 | 5.39605e-07 |
| late | warp_F_det4_1500 | -0.0572679 | 10 | -4.45073e-07 | 1.61019e-07 |
| late | warp_F_det4_6000 | 0.114732 | 11 | 6.1396e-08 | 2.84221e-07 |
| late | additive_F_last | 0.957643 | 12 | 6.32614e-07 | 2.61725e-06 |
| future | warp_X_P | 0.134812 | 10 | -7.53913e-10 | 4.18252e-07 |
| future | warp_F_det4_1500 | -0.0845024 | 9 | -4.66751e-07 | 1.12914e-07 |
| future | warp_F_det4_6000 | 0.205835 | 10 | 8.63651e-08 | 6.56115e-07 |
| future | additive_F_last | 0.973667 | 12 | 6.7815e-07 | 2.83944e-06 |

主候选逐Re：

| split | Re | gain_percent |
| --- | --- | --- |
| development | 26700 | 0.65992 |
| development | 6300 | 1.68034 |
| development | 7575 | 1.3074 |
| late | 26700 | 0.593932 |
| late | 6300 | 1.42568 |
| late | 7575 | 1.30488 |
| future | 26700 | 0.286284 |
| future | 6300 | 1.39638 |
| future | 7575 | 1.25262 |

![结果及失败边界](figures/structure_results.png)

## 收益是否只是固定平滑

补充控制也是在首次开发结果后提出：56个常量局部参数只在train上优化，统一3×3平滑及两控制强度只在calibration选择。固定原主候选，不以这些控制重选。主模型分别仅保留几何/幅值/偏置系数作机制消融。分量存在交互，单项消融收益不应相加。

| split | method | gain_percent |
| --- | --- | --- |
| development | amplitude | -0.0457416 |
| development | base | 0 |
| development | bias | -0.0213366 |
| development | full | 0.791065 |
| development | geometry | 0.667684 |
| development | smooth | 0 |
| development | static56 | 0.0528258 |
| late | amplitude | -0.0791022 |
| late | base | 0 |
| late | bias | -0.0306704 |
| late | full | 0.721686 |
| late | geometry | 0.641015 |
| late | smooth | 0 |
| late | static56 | 0.106793 |
| future | amplitude | -0.167473 |
| future | base | 0 |
| future | bias | 0.0128534 |
| future | full | 0.45264 |
| future | geometry | 0.419427 |
| future | smooth | 0 |
| future | static56 | 0.0779253 |

几何部分承担主要收益；固定控制未达到同等效果。这仍不等同于已辨识了真实物理输运速度。

## F控制与样本效率

补训了完全相同输入（含历史增量）、相同网络形状和99→66→33→0四调用图的确定性模型，目标仅为终点真实未来MSE。同seed42、相同主干训练身份及前200帧。1500步近似匹配原6000步去噪训练的网络调用数；6000步匹配更新次数，训练计算量更高，不能称严格等训练成本。Fdet用于同一冻结DiPred的误差修正；不替换P，所以实际使用Fdet还需额外主干计算。

最初已冻结的主候选不变。Fdet-only局部头及25/50%局部学习曲线是看到首次开发结果后补充的机制控制，校准选强度，不重新选择主候选。它们不能伪装为全部预注册。各比例使用嵌套完整训练轨迹；PCA也仅在该子集拟合。X/XP使用相同32维PCA和同小头；不宣称冻结F的1.6M表征计算免费，实际推理计时见下。

25%/50%训练量下三种局部头均由校准选为零修正，因此**没有证明小标注样本效率优势**。确定性1500步F在三个时段的平均MSE均略优于本候选，配对区间跨零；不能宣称扩散目标具有独有优势。复用当前DiPred的F仍可省去另算确定性主干的推理成本。

| split | features | train_percent | gain_percent |
| --- | --- | --- | --- |
| development | Fdiff | 25 | 0 |
| development | Fdiff | 50 | 0 |
| development | Fdiff | 100 | 0.791065 |
| development | Fdet4 | 25 | 0 |
| development | Fdet4 | 50 | 0 |
| development | Fdet4 | 100 | 0.703199 |
| development | XP | 25 | 0 |
| development | XP | 50 | 0 |
| development | XP | 100 | 0.319875 |
| late | Fdiff | 25 | 0 |
| late | Fdiff | 50 | 0 |
| late | Fdiff | 100 | 0.721686 |
| late | Fdet4 | 25 | 0 |
| late | Fdet4 | 50 | 0 |
| late | Fdet4 | 100 | 0.607651 |
| late | XP | 25 | 0 |
| late | XP | 50 | 0 |
| late | XP | 100 | 0.542447 |
| future | Fdiff | 25 | 0 |
| future | Fdiff | 50 | 0 |
| future | Fdiff | 100 | 0.45264 |
| future | Fdet4 | 25 | 0 |
| future | Fdet4 | 50 | 0 |
| future | Fdet4 | 100 | 0.247314 |
| future | XP | 25 | 0 |
| future | XP | 50 | 0 |
| future | XP | 100 | 0.318256 |

## 长滚动与实际成本

独立任务：12条开发轨迹仅初始20帧是真实观测，连续自回归10个完整20帧块；同时间范围的observed控制每次读取真实历史。没有中途真值修正自由滚动输入。正收益不能从一个任务转移到另一个任务。

| mode | method | gain_percent | mse |
| --- | --- | --- | --- |
| free | warp_F_last | -0.416255 | 0.000310539 |
| free | additive_F_last | -2.13465 | 0.000315853 |
| observed | warp_F_last | 0.721364 | 0.000166049 |
| observed | additive_F_last | 0.00702341 | 0.000167244 |

| method | median_ms | p10_ms | p90_ms | peak_allocated_mb | repeats |
| --- | --- | --- | --- | --- | --- |
| base | 6.96637 | 6.92987 | 7.57575 | 17.8315 | 30 |
| warp_F_last | 8.09821 | 7.94601 | 8.4719 | 17.9707 | 30 |
| additive_F_last | 7.44013 | 7.39517 | 8.25217 | 17.9707 | 30 |

计时为A100、batch1、预载模型、30次，包括完整主干、特征提取、PCA与场修正，不包括磁盘和评分。小差异受测量噪声影响。Peak是实际torch分配峰值，不是nvidia-smi显存预留。

## 审计与使用

合成已知位移/相位/振幅恢复的最大系数误差为6.12e-06，独立NumPy物理MSE差为6.8e-12。实际观测直接推理与缓存重放通过，见inference_audit.json。真实流场位移—幅值分解未证明唯一，不把参数称为已辨识的物理状态。

从Workspace运行：

```python
import sys
sys.path.insert(0, 'error_structure_20261117')
from inference_rollout import Predictor
predictor = Predictor('warp_F_last')
future = predictor.predict(observed_standardized)  # [B,2,20,32,64], CUDA
```

归一化使用当前发布checkpoint中的mean/std；此入口不接受未来标签，不依赖评测缓存。

## 全目标剩余工作

1. 结构修正已有观测驱动收益及自由滚动反例；仍需审查同信息ResCrct/更直接残差控制、局部收益机制与扩展独立确认。
2. 低成本联合分布尚未完成；旧smooth仍依赖32条扩散路径。下一步优先从真实残差学习低秩/相关结构，与固定边际ECC、smooth、N-ECC比较，并实测成本。
3. 在线潜状态尚待有符号反馈、漂移与同信息控制实验。
4. 主动观测尚待固定新增观测时刻/预算的实际信息价值实验。
5. 计算分配尚待使用真实修正收益训练门控、相同预算对照和实际成本曲线。
6. 物理证书初步审查：calibration全部HDF5只含u/v/x/y/t/Re/aoa，无压力和出平面速度；本地inform.md亦说明压力零填充且不评分。仅凭这些观测不能构造有已核实前提的完整动量残差/伴随严格误差界。该结论不否定经验任务误差估计，后者仍待实验。

完整目标保持active。原始主线权重未替换，旧目录和外部仓库未发布改动。复现命令见源码目录README.md；所有负结果保留。
