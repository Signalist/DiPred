# Trust causal audit v2

本目录是修复后的唯一有效准备和审计入口。`research/error_guided_spatial_routing/` 及其结果完整保留为无效历史实验，不再运行旧 `run.py`、`prepare.py` 或占位测试，也不使用旧权重推断新定义模型的效果。

从 trust worktree 根目录运行，依赖本机现有 PyTorch/CUDA 和只读 `.deps/h5py`：

```bash
export PYTHONPATH="$PWD/../Forecast/Runtime/.deps${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export CUDA_CACHE_PATH="$PWD/.cache/cuda"
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
python research/trust_causal_audit_v2/run.py all
python research/trust_causal_audit_v2/verify.py
```

`run.py all` 无训练和评分模式，并拦截优化器构造；已有开始记录时检查原产物哈希而非重新冻结。重复审计只覆盖新审计目录，首次失败保留在 FAILURE.json；不会覆盖旧目录。`PROTECTED_BEFORE.json` 含旧大缓存哈希，可只读复核；`SOURCE_HASHES.json` 和 `SOURCE_DIFF.patch` 标明旧源码完全未改以及新增源码。

`data.py` 提供：显式 Key、CacheIndex.lookup、只接受历史的 prepare_inputs、完整原批次噪声重建 batch_noise；`model.py` 提供两种同信息路由器及 native_sample；`audit.py` 生成 keyed WINDOW_MANIFEST、risk_maps.npz 和训练独占 RISK_NORMALIZATION。所有未来标签与路由输入分离，测试清单只做元数据审计。

原生旧缓存由 TF32 执行生成，新实现采用 FP32。它们可作为已发布的过去预测反馈；新的原生/路由比较必须使用 `native_sample(native, same_history, same_noise)` 和相同精度，禁止把旧 TF32 输出当作 FP32 严格配对基线。NOISE_IDENTITIES 记录实际CUDA噪声哈希；运行环境变化时须先重做 D/真实小样本复现。

下一轮训练尚未获本目标授权。旧486参数checkpoint与1269参数修订接口不兼容，必须重新初始化，不能把旧权重解释为本协议训练结果。
