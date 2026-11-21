# DiPred follow-up 状态审计

核验日期为 2026-11-21，UTC。任务已完成并通过复核，已有本地提交，尚未出现在远端 dynamics。

| 项目 | 实际状态 |
|---|---|
| 会话工作目录 | /cpfs01/projects-HDD/cfff-3469a2cbe57f_HDD/wangxiyun |
| 仓库与实验工作目录 | /cpfs01/projects-HDD/cfff-3469a2cbe57f_HDD/wangxiyun/DiPred+ResCrct/Workspace/github/DiPred |
| 当前分支 | dynamics |
| 实际 HEAD | 0d3e6cc2f6532b9264c7c676bdb0c00cdd24963b |
| 远端 dynamics，直接查询 | dca60cf90a7f5974bfa8338061ad5dea9c97879a |
| 本地 origin/dynamics 跟踪引用 | 与直接查询远端的值相同 |
| 提交差异 | 远端独有 0，本地独有 1 |
| 本地交付提交时间 | 2026-11-21 09:10:43 UTC |
| 关联 worktree | git worktree list 只列当前仓库一个 worktree，分支 dynamics |
| Git 公共目录 | 当前仓库 .git |
| 交付提交范围 | 54 个新增文件，其中代码 3 个、结果及报告 51 个 |
| 审计开始时已跟踪文件修改 | 无 |
| 本次 STATUS.md | 新增审计文件，尚未提交 |
| 路径缓存 | 20 个 .pt 共 2,831,225,280 字节，另有 20 个 JSON 身份文件 |
| 缓存 Git 状态 | .pt 被现有规则忽略，JSON 未跟踪；统一 cache_manifest.json 已提交 |

本地领先远端的提交主题为 Validate frozen phase3 heads over twenty windows and decompose missing propagation terms。对照远端基线，变更全部位于 research/residual_dynamics_phase3_followup/ 和 results/residual_dynamics_phase3_followup/。在 Workspace 内按 FOLLOWUP、phase3 和 STATUS 文件名搜索，结合 research/results 清单及 worktree 列表检查，定位到本目录这一套对应交付。

审计开始时原有未跟踪项目包括 reliability/extreme_predict.py、reliability/two_tail.py、research/5pde/、research/legacy_xpf16_feature_ablation/，以及 results/reliability 下的 5pde、fair_f_p_pf_xf、fair_f_p_pf_xf_sharedref、fair_x_xp_xpf、legacy_xpf16_feature_ablation 目录。本次项目内新增内容为本 STATUS.md，临时审计脚本和输出位于 /tmp。

**完成状态分类如下。**

| 状态类别 | 判定及证据 |
|---|---|
| 已实现但没有运行 | 本任务无此类必需实验；日志覆盖复现、延伸、诊断、测试及统计 |
| 已运行但未完成统计 | 无；明细、各级汇总、配对区间及反例已生成并核验一致 |
| 已完成并通过复核 | 三个冻结 CRPS 头的旧窗口复现、20 窗口延伸、传播分解、创新相关及共同平移检查 |
| 因缺数据、缓存或算力阻塞 | 当前任务无此类阻碍；源数据和完整路径缓存实际存在且哈希一致 |
| 已完成但尚未提交 | 实验代码和结果已提交；本次新增 STATUS.md 尚未提交 |
| 已完成但尚未出现在远端 | follow-up 提交 0d3e6cc 目前仅在本地 dynamics |

**本次核验直接读取了实际内容。**

53 个交付文件的 SHA-256 全部匹配 SHA256SUMS，12 个上游依赖匹配固定协议。9 个旧 checkpoint 均核对 selection.csv 与协议哈希，并使用 CPU 读取 selected、endpoint、scale、step、kind 字段。三个 CRPS 头均为第 1000 步、强度 1，scale 均为 0.012425202876329422。raw/std 的 selected 最后一层权重和偏置均为零，与原零输出检查相符。主干 checkpoint 哈希也与固定协议一致。

| 固定 CRPS checkpoint | 本次核对的 SHA-256 |
|---|---|
| crps_42.pt | 85f81f4879b26150a9b44aa0ae1775fc2143d8c0cfa44a62de0eaa003c36bec1 |
| crps_43.pt | 48c23d50bb49250a70d5922ac52033d4c3701b57e5b17e1bc66c59b69e54cef8 |
| crps_44.pt | 382fa01bffdc6851c724b81effaf5166947e1e7fdf2f48a35016507db2ffb598 |

20 个路径缓存逐一重新计算文件哈希，并读取真实张量和身份。每块均有 12×32×2×20×32×64 的原生路径和三个修正场。轨迹顺序、成员种子、路径张量哈希和历史哈希一致，后续块的历史哈希与前块已揭示真值哈希一致。12 个原始 HDF5 的全文件哈希均匹配。40 条发布／揭示日志与缓存身份对应，顺序为完整预测后揭示完整块真值。

960 行块评分覆盖完整的 12 轨迹×20 窗口×4 方法／种子组合，38,400 行 lead／通道记录覆盖每块全部 40 个单元。本次由明细重算块均值，CRPS 和 MSE 最大差异分别为 5.56×10⁻¹⁷、2.17×10⁻¹⁹；由块评分独立重算每条轨迹内的窗口及固定头评分平均、总体效应和 10,000 次轨迹 bootstrap，与 summary.csv、paired.csv 一致。480 行旧窗口复现评分与 phase3 原表逐值一致。

11,520 行传播分解和 10,944 行创新相关记录完整。分解表的 total=inherited+accumulated+cross 恒等式最大误差为 6.04×10⁻¹⁹。既有逐场 b=s+g 核验记录的最大误差为 2.50×10⁻¹⁶。tests.log 记录 5 项测试通过，包含真实冻结头在旧窗口的新推理入口复现。本次检查现有文件、缓存和统计，没有重新执行模型推理或训练。

第一次旧窗口复现曾发生模块绝对导入冲突，记录在 reproduce_import_failure.log。随后成功日志、完整评分及修复后的已提交代码相互对应，当前状态已完成。

**上轮实际执行命令与记录如下。**

以下命令来自本会话已执行工具记录，并与现存日志对应。工作目录为上表仓库根目录。

    source ../../env.sh
    export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
    python -u research/residual_dynamics_phase3_followup/run.py reproduce
    python -u research/residual_dynamics_phase3_followup/run.py extend
    python -u research/residual_dynamics_phase3_followup/run.py diagnose
    python research/residual_dynamics_phase3_followup/analyze.py
    python research/residual_dynamics_phase3_followup/tests.py

reproduce 在导入问题修复后成功执行，analyze 和 tests 在最终代码复核时再次执行。输出分别保存在 reproduce.log、extend.log、diagnose.log、analyze.log、tests.log。extend.log 逐项记录历史起点 440 至 820 完成 1/20 至 20/20，生成、评分和缓存写入约 50 秒。报告中的 run.py all 为组合复现入口，上轮实际按上述阶段调用。

本次执行了 pwd、git branch --show-current、git rev-parse HEAD、git status --short --branch、git worktree list --porcelain、git remote -v、git ls-remote origin refs/heads/dynamics、git rev-list --left-right --count dca60cf90a7f5974bfa8338061ad5dea9c97879a...HEAD、git diff --name-status dca60cf90a7f5974bfa8338061ad5dea9c97879a HEAD，以及文件搜索、读取和哈希检查。独立只读审计命令如下，其脚本和 JSON 输出保存在 /tmp。

    source ../../env.sh
    python /tmp/dipred_followup_status_audit.py

**经本次重算确认的关键结果如下。**

| 指标 | 原生 | 三个冻结头评分平均 | 修正减原生 | 95% 轨迹配对区间 | 改善轨迹 |
|---|---:|---:|---:|---|---:|
| 标准化经验 CRPS | 0.08686948 | 0.08666144 | −0.00020804 | [−0.00036267, −0.00004531] | 9/12 |
| 集合均值 MSE | 0.000114489 | 0.000114993 | +5.04×10⁻⁷ | [−3.41×10⁻⁸, +1.33×10⁻⁶] | 5/12 |
| 完整路径标准化 RMS ES | 0.16937492 | 0.16904568 | −0.00032924 | [−0.00063217, −0.00002417] | 8/12 |

旧窗口约 0.33% 的 CRPS 改善在后续时段缩小至约 0.24%，ES 改善约 0.19%，均值 MSE 增加约 0.44%。收益主要来自 Re=3750 和 v 通道；CRPS 退步轨迹为 13950_20.h5、13950_5.h5、22875_10.h5。

传播分解的全 20 帧平均继承项、累计未解释项、交叉项和总误差依次为 6.247×10⁻⁶、113.689×10⁻⁶、−5.447×10⁻⁶、114.489×10⁻⁶。第 20 帧累计未解释项约为继承项的 52 倍。仅看第 2–20 帧，真实初值传播后剩余误差相对原生的均方优势约 0.0096%。这部分依赖未来真值，支持对当前传播器缺项的离线解释。

240 个块有 7 个事件正例，全部来自 1 条原轨迹。已有轨迹和时段此前已被研究。新独立轨迹及更多独立正例轨迹仍是结论确认所需的数据，当前固定实验已完成。

**供审查的代码、报告和小型统计文件如下。** 所有列项已包含在本地 0d3e6cc 提交中，尺寸按实际文件计算。

| 用途 | 文件 | KiB |
|---|---|---:|
| 推理、复现及分解代码 | [run.py](../../research/residual_dynamics_phase3_followup/run.py) | 19.9 |
| 统计与核验代码 | [analyze.py](../../research/residual_dynamics_phase3_followup/analyze.py) | 9.7 |
| 测试代码 | [tests.py](../../research/residual_dynamics_phase3_followup/tests.py) | 3.8 |
| 研究报告 | [FOLLOWUP_DECISION.md](FOLLOWUP_DECISION.md) | 12.1 |
| 固定协议 | [PROTOCOL.md](PROTOCOL.md) | 2.6 |
| 配置与模型哈希 | [protocol.json](protocol.json) | 21.4 |
| 核心汇总 | [summary.csv](summary.csv) | 0.2 |
| 配对区间 | [paired.csv](paired.csv) | 22.2 |
| 各种子 | [per_seed.csv](per_seed.csv) | 0.4 |
| 各 Re | [per_Re.csv](per_Re.csv) | 0.5 |
| 各轨迹 | [per_trajectory.csv](per_trajectory.csv) | 2.0 |
| 轨迹配对差 | [trajectory_differences.csv](trajectory_differences.csv) | 1.1 |
| 各窗口 | [per_window.csv](per_window.csv) | 3.2 |
| 各 lead 与通道 | [per_lead_channel.csv](per_lead_channel.csv) | 4.2 |
| 分解汇总 | [decomposition_summary.csv](decomposition_summary.csv) | 0.3 |
| 逐 lead 分解 | [decomposition_by_lead.csv](decomposition_by_lead.csv) | 5.9 |
| 相关汇总 | [correlations_summary.csv](correlations_summary.csv) | 0.3 |
| 事件统计 | [events_summary.json](events_summary.json) | 2.3 |
| 旧窗口复核 | [reproduction.json](reproduction.json) | 2.5 |
| 完整验证 | [verification.json](verification.json) | 1.0 |
| 分解恒等式验证 | [decomposition_audit.json](decomposition_audit.json) | 3.9 |
| 文件哈希 | [SHA256SUMS](SHA256SUMS) | 6.6 |
| 缓存身份与哈希 | [cache_manifest.json](cache_manifest.json) | 50.3 |
| 源数据哈希 | [data_hashes.json](data_hashes.json) | 1.0 |

完整明细另见 scores.csv、lead_channel.csv、decomposition.csv、innovation_correlations.csv，运行日志均在本目录。原三个冻结 CRPS 权重位于 ../residual_dynamics_phase3/weights/。大路径缓存保存在本地 cache/，其统一身份和哈希已提交。

**剩余事项为审查与发布状态处理。** 固定研究任务没有待运行实验或统计步骤。远端尚缺一个已完成的本地提交，本次 STATUS.md 尚未提交。本轮状态核验已结束；后续如授权发布，可围绕已有提交及本状态文件处理。

本次内容审计完成时间为 2026-11-21T09:52:29Z。临时审计结果位于 /tmp/dipred_followup_status_audit.json。
