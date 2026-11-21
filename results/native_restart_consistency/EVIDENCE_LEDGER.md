# 固定误差算子＋共同位置修正支线：证据归档

归档参考提交76e3ec27416a0afeee0ee379d86b2ed7f7e56ed0。所有旧代码、权重、门槛和结果保留。以下“支持”均限于已用轨迹、有限模型与信息边界，不是新盲测或PDE因果结论。

| 轮次 | 已支持的有限结论 | 不支持或证据不足 | 未测试／不能外推 |
|---|---|---|---|
| 初始A/B/C成员残差诊断 | 在同轨迹、同成员、真上一帧残差可用时，上一帧与邻域对下一帧带符号残差有增量信息；成员打乱破坏对应 | 不能据离线标签收益证明发布时可改进预测 | 无当前真值的部署、真实输运速度、因果机制 |
| phase2持续性和方向性 | 复制解释大量一步价值；自由局部空间方向比对称控制更有效；原生成员时间对应对联合评分重要 | 纯一步关系转成完整无反馈20帧预测未获稳定收益；零位移相关不等于有物理速度 | 对成员相关性的诊断不等于训练了新的成员依赖 |
| phase3固定算子及共同偏差头 | CRPS训练头有小幅概率评分收益；误差分解及真实初值递推可作离线诊断 | 纯初值传播的原门槛未通过；MSE与通道存在代价 | 新物理动力学、任意新工况泛化 |
| phase3_followup冻结连续时段 | 固定三头在旧extension平均CRPS约改善0.24%、ES约0.19%；累计未解释项随lead增长 | 原尺度MSE约上升0.44%；未解释项的相关不证明其可由发布条件预测 | 新独立测试、多个独立事件正例轨迹、真实PDE缺陷 |
| conditional_innovation_pilot | 六次固定岭回归完成；弱正则下有工况集中的源项信号 | CRPS选定R1源项误差约退步0.37%，两发布候选均零；原gate=false | 当轮神经训练0次，不能称已比较非线性D/T；不能否定所有非线性模型 |
| nonlinear_parameterization_control | 独立协议九次×1000步完成，科学与发布选择均保留；梯度、时间上下文、感受野差异通过检查 | D科学选择全零；T_local相对D旧窗口CRPS退步、extension小幅变化区间跨零，两个时段均未优于T_diag点估计；效果集中Re3750并有MSE代价，无稳定局部传播增量支持 | 其他网络/算子/训练时段/联合分布改进，均未由这九次结果排除 |

上述**发布后处理支线仅向所有32个成员减去同一个预测场q**。成员对差、中心化成员场、方差与协方差结构在浮点舍入内保持不变，没有学到新的协方差或随机成员依赖。初期逐成员真残差是离线诊断，不应与后期共同平移部署模型混为一谈。固定A时任意q可重新写为q1及条件源项，这种表示不新增信息；T_local扩大空间感受野，收益即使出现也不自动证明物理输运。

本轮决策：归档这条固定算子＋共同位置修正支线，不追加旧头、loss、特征或搜索。原生物理时间重启一致性为独立的冻结模型分布探针，不是旧门槛复活；是否有分布差异及是否与预测质量相关需单独检验。

可审查来源：

- [初始A/B/C](../residual_dynamics/README.md)与[轨迹对齐](../residual_dynamics/alignment.json)。
- [phase2报告](../residual_dynamics_phase2/RESEARCH_REPORT.md)。
- [phase3报告](../residual_dynamics_phase3/DECISION_REPORT.md)与[门槛](../residual_dynamics_phase3/gates.json)。
- [冻结延伸报告](../residual_dynamics_phase3_followup/FOLLOWUP_DECISION.md)。
- [线性pilot报告](../conditional_innovation_pilot/DECISION_REPORT.md)与[gate=false](../conditional_innovation_pilot/selection_A.json)。
- [非线性对照报告](../nonlinear_parameterization_control/DECISION_REPORT.md)、[45个候选](../nonlinear_parameterization_control/candidates.csv)、[18项选择](../nonlinear_parameterization_control/selection.csv)、[配对效应](../nonlinear_parameterization_control/paired.csv)、[九次训练完成](../nonlinear_parameterization_control/training_complete.json)。
