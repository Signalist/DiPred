# RealPDE 六方向研究（2026-11-17）

本目录整理六条研究路线的主要代码、协议、数值汇总与审计。结论和边界见[总报告](runs/research_completion_20261117/REPORT.md)；逐条要求及来源见[要求映射](runs/research_completion_20261117/REQUIREMENTS.md)。

- [结构化误差](runs/error_structure_20261117/REPORT.md)及[ResCrct等直接对照](runs/direct_correction_controls_20261117/REPORT.md)
- [低成本联合分布](runs/lowcost_joint_20261117/REPORT.md)
- [在线有符号状态](runs/online_signed_state_20261117/REPORT.md)
- [主动观测](runs/active_observation_20261117/REPORT.md)
- [计算预算分配](runs/compute_allocation_20261117/REPORT.md)
- [物理任务误差](runs/goal_task_error_20261117/REPORT.md)
- [重新汇总的配对结果](runs/research_completion_20261117/recomputed_gains.csv)

本次只上传主要源代码、协议、小型数值表和审计记录；未上传原始RealPDE数据、主干/分位头权重、模型二进制、数GB中间缓存、逐窗口大表和图像。审计JSON中的路径、哈希、阶段状态是原实验记录，引用的全部资产并不都在本目录；报告中的部分本地文件或图像链接可能因精简未包含。请以本页和各报告内的数值表为阅读入口。

源代码保留实验时的Workspace布局与硬编码路径，不是安装包：直接运行还需要准备原RealPDE数据、原项目的Shared模块、冻结主干与49个边际头权重、对应缓存和依赖，并按各脚本调整路径。本目录的 `dependencies/Shared` 保存关键共享源码快照，不能单凭复制此目录完成复现。提交不包含这些缺失资产，不声称可开箱重跑或完整复核所有历史哈希。

开发、late和future使用同一组曾被反复探索的轨迹；本研究是探索性结果。区域均速、最大速度和自由滚动的退步，以及主动观测和物理证书的限制均在报告中保留。
