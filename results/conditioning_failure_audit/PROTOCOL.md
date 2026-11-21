# 冻结条件适配瓶颈核查：执行前协议

基线1d118a1d3a433a9111d42d5fa904a84eeb40cf69，dynamics。仅C_aug/seed42/step0、250、1000；k5/start400/repeat0/32成员。零优化更新，不搜索、不新读测试或extension评分，不修改旧文件，不push。

训练12条：按训练Re组编号数值升序，每组内文件名字典序；首轮各取首条（10组），再按同一Re顺序轮流取下一条补至12。开发原12条全部。先保存manifest/config及SHA256再执行，不查看本轮效果选样本。训练子集非全训练代表。

最多72个完整32成员前向组。step0优先复用完全同身份旧S5完整20帧缓存24组；非零新增48组=1536路径，192批网络调用。step0四次delta只运行零适配器读取，未重放冻结主干；其输出等价由旧预检24组证据、当前零层权重与相同sample/source/hash共同核验。非零通过adapter forward hook截取实际四次delta，不修改原sample数学行为；无第二次采样。所有参数eval/requires_gradFalse，torch.no_grad，无优化器。

仅载入有效根目录checkpoints.json指定3权重及hash；作废批次只引用原成本表，不加载其权重或评分。旧缓存文件/历史/父A/S5张量、成员身份、子实际noise32 hash、源HDF5与归一化核验。父不重生，不排序或平均成员。预测及delta完成、输出hash写事件后才读取真值[start425,end445)。每条相同预测同时计算full20=[425,445)、overlap15=[425,440)、extra5=[440,445)，没有另一个前向。

CRPS在冻结训练通道尺度下float64计算完整32成员经验分数，保存first=mean|p−y|与pair_half=sumij|pi−pj|/(2M²)，CRPS=first−pair_half。均值MSE原尺度及标准化均保留，overall/u/v逐轨迹。验证full=.75overlap+.25extra，对CRPS、两项及MSE均用绝对1e−12；排序公式与显式成员对小数组数值测试1e−10。旧开发相同身份overlap CRPS/MSE/SMSE与旧有效development.csv绝对1e−12（沿用旧重算标准）；step0与旧native quality.csv相同身份对照1e−12，旧输出物理复现阈值2e−6不扩大。任何不一致立即写STOP.json并停止后续组，不补样本、不放宽容差。

delta单位为标准化历史坐标；观测区域历史前15位置、生成区域最后5位置，各自按区域元素数求均方再开方，保存99/66/33/0、overall/u/v及轨迹。汇总取轨迹RMS等权均值，不把区域未归一化范数比较为损害。step0全部delta为0。

小表trajectory_scores.csv及delta_by_trajectory.csv；核心表1 score_summary.csv：划分×step×segment×channel绝对分数与相对0配对差、改善轨迹数、95%轨迹bootstrap10000/seed123；表2 delta_summary.csv：划分×step×denoise×region×channel的轨迹等权RMS和区域元素数。训练/开发分别12原轨迹，不以成员像素当独立样本。A固定训练full未改善；B训练full改善而overlap未改善；C训练overlap改善而开发overlap退步；混合/区间不支持则D证据不足。各step分开，禁止据此唯一归因或宣称过/欠拟合，在线随机loss不作固定性能曲线。

预算一块现有A100，最多72完整前向组（预计48新增）、墙钟15分钟、新增缓存≤256MiB。仅记录小统计/输入输出hash，不保存预测大缓存，不重算旧矩阵。缺资源/数值失败就报告明确缺口结束；不得追加训练或新测试。报告只提出一条下一动作或建议归档。成本记录真实墙钟/路径/缓存，与Codex费用无关。
