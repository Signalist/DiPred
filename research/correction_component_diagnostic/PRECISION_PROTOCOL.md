# 精度协议修订

原严格 FP32 门禁在 `6300_0.h5`、start420 的原生 32 成员输出上触发停止。复核先以同一窗口的成员 500000 在独立进程对照旧记录与严格 FP32，核对权重、历史、归一化、完整 split 批噪声及成员索引。单成员结果用于核查身份和包装器路径，旧缓存的复现门禁采用原 32 成员批形状。

匹配旧批形状的门禁要求 N、C0-J、C1-J **逐值一致**。旧有效精度为 float32 张量、CUDA matmul 允许 TF32、cuDNN TF32 关闭、确定性 cuDNN。旧精度在进程启动前设置，整个组件实验不在进程内切换。严格 FP32 对照只用于定位数字差异。旧缓存分解及本轮组件、幅度、阶段和消融均对应已通过复现的旧有效精度。容差、评分和原协议的中央差分保持原值。

从仓库根目录运行。使用新的输出路径保留停止记录及此前输出。

```bash
export PYTHONPATH="$PWD/../Forecast/Runtime/.deps${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
out=/absolute/path/to/new-results
mkdir -p "$out"
env -u CUBLAS_WORKSPACE_CONFIG TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1 \
  python -m research.correction_component_diagnostic.precision_probe legacy "$out/MEMBER_LEGACY.json"
env -u CUBLAS_WORKSPACE_CONFIG TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 \
  python -m research.correction_component_diagnostic.precision_probe fp32 "$out/MEMBER_FP32.json"
env -u CUBLAS_WORKSPACE_CONFIG TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1 \
  python -m research.correction_component_diagnostic.run --legacy-precision \
    --prior-cost results/correction_component_diagnostic/COST.json --output "$out/diagnostic"
env -u CUBLAS_WORKSPACE_CONFIG TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 \
  python -m research.correction_component_diagnostic.precision_probe fp32 "$out/MATCHED_BATCH_FP32.json" --matched-batch
```

完整入口在三项批形状匹配的缓存门禁成功后才执行组件诊断。恢复计数以此前 1 组已完成前向为起点，完整诊断 176 组，独立严格 FP32 对照 1 组，总计 178 组。单成员前向 4 次单记。
