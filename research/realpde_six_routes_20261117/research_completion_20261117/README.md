# 最终整合入口

结果见[最终报告](../runs/research_completion_20261117/REPORT.md)，逐项边界见[43项验收映射](../runs/research_completion_20261117/REQUIREMENTS.md)。

在Workspace运行：

```bash
source env.sh
python research_completion_20261117/refresh_mechanics.py
python research_completion_20261117/audit.py
python research_completion_20261117/plot.py
python research_completion_20261117/finalize.py
```

第一步在GPU重新检查当前联合生成机制，只写整合目录，保留历史审计；第二步不导入模型实现，从CSV独立重算主要比较并核对现有哈希、协议与证据文件；第三步生成可导出的PNG/PDF；第四步确认清单未发生变化、文档链接有效，再写最终产物清单。均不重新训练、选模或覆盖默认主干。

实验各自的API/复现说明：[结构修正](../error_structure_20261117/README.md)、[直接强对照](../direct_correction_controls_20261117/README.md)、[联合生成](../lowcost_joint_20261117/README.md)、[在线状态](../online_signed_state_20261117/README.md)、[主动观测](../active_observation_20261117/README.md)、[计算分配](../compute_allocation_20261117/README.md)、[任务误差](../goal_task_error_20261117/README.md)。

早期阶段的完成状态和源码哈希是时间快照，不据此改写当时历史。最新汇总的84项源码/权重/协议哈希全部一致；旧版机制审计的源码差异由本目录新审计补证。数值验收不能证明独立泛化、理论保证或任意方法普遍失败。
