# Residual & Uncertainty Conditioned Diffusion

从 trust worktree 根目录运行，本机现有CUDA/PyTorch及只读h5py依赖：

```bash
export PYTHONPATH="$PWD/../Forecast/Runtime/.deps${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
python -m research.residual_uncertainty_diffusion.prepare
python -m research.residual_uncertainty_diffusion.tests
python -m research.residual_uncertainty_diffusion.run
python -m research.residual_uncertainty_diffusion.verify
```

准备及训练拒绝覆盖已开始记录。旧源码和产物不改、不重新运行旧审计。新E/W与预测完整张量在`.cache/residual_uncertainty_diffusion/`；记录哈希及明确身份，无需进入Git。小checkpoint、协议、日志、逐轨迹评分、选择、消融及中文报告在新results目录。

本轮输出头零初始化是G的最后输出层，不将整个G零初始化。仅native.net.out可微调，其他主干及XPF冻结。有效训练至多C0/C1各500更新；任何中途错误不得重启完整训练追加更新。原始8成员fair目标包含排除自配对的成员距离项，开发用32经验CRPS。精度FP32/TF32关闭。
