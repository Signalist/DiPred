# 精度恢复记录

原严格 FP32 门禁的误差为 3.5451353×10⁻⁴。旧实验 Python 3.8.10、PyTorch 1.14.0a0+44dac51、CUDA 12.0、A100-SXM4-80GB，恢复时相同，cuDNN 版本为 8700。旧 `configure()` 尝试关闭 TF32。本机 `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1` 在该调用后仍使实际 matmul 标志为 true。覆盖设为 0 后，实际 matmul 标志为 false。两种模式均禁用 cuDNN TF32、启用确定性 cuDNN。对照进程均取消 `CUBLAS_WORKSPACE_CONFIG`。

固定 `6300_0.h5`、start420、成员 500000，核对原始历史区间 [420,440)、标准化张量及归一化哈希、原始权重、完整开发 split 内生成的噪声切片，以及成员 500000–500031 的顺序。旧缓存保存标准化路径，按原归一化重算与旧逐窗得分相同。归档哈希和权重哈希均通过核验。[单成员旧设置](MEMBER_LEGACY.json) · [单成员严格 FP32](MEMBER_FP32.json) · [来源哈希](diagnostic/SOURCE_HASHES.json) · [样本及噪声哈希](diagnostic/CONDITION_NOISE_HASHES.json)

|实际 matmul TF32|批成员数|输出与旧缓存最大绝对差|
|---|---:|---:|
|开|1|3.0714273×10⁻⁴|
|关|1|3.0714273×10⁻⁴|
|开|32|0，N、C0-J、C1-J 均逐值一致|
|关|32|3.5451353×10⁻⁴，原生 N|

两次单成员输出的 SHA-256 相同，各自也与同环境的包装器逐值相同。单成员与旧 32 成员缓存的偏差随批形状变化，结论以匹配形状的对照为准。在匹配形状、权重和噪声下，仅改变 matmul TF32 即在失败与逐值复现间切换。2e-6 容差保持原值，恢复门禁要求零差。独立严格 FP32 探针重现原阻断的输出哈希。[批形状复现](diagnostic/CACHE_REPRODUCTION.json) · [同批形状严格 FP32](MATCHED_BATCH_FP32.json)

后续比较统一使用已复现旧缓存的有效设置，保留旧缓存的配对资格。每个新增完整前向都核对推理前、后的精度状态，C1 的 0/1 梯度端点在固定 6 窗复现 O/J 的输出哈希。输入哈希全程保持稳定，E_off/W_off 仅修改指定槽。未来标签置换后，发布输入和输出哈希均保持一致。所有权重哈希在前后复核一致。[验收记录](diagnostic/VERIFICATION.json) · [发布边界](diagnostic/PUBLICATION_AUDIT.json) · [权重哈希](diagnostic/WEIGHT_HASHES.json)

本次恢复执行 177 组完整 32 成员四步前向，其中固定诊断 176 组、独立严格 FP32 对照 1 组。加上原停止运行的 1 组累计 **178/192**。另有 **4** 次单成员四步前向。梯度图 **12/12**，真实 `autograd.grad` **24/24**，优化器实例及更新均 **0**。固定诊断墙钟 54.787 秒，三项精度对照合计 18.403 秒，均为运行计时。[合并成本](TOTAL_COST.json)
