# 完整测试集平均 CDF

`cdf_u.npz`、`cdf_v.npz` 和对应的 CSV 包含完全相同的对称 `x` 网格、六个平均 CDF 及各通道观测数量；`config.json` 记录轨迹、窗口、尺度和处理规则。读取示例：

```python
import numpy as np
import matplotlib.pyplot as plt

data = np.load('cdf_u.npz')
for name in ('xpf16_1', 'xpf16_02', 'noise8', 'noise32', 'noise256', 'noise2048'):
    plt.plot(data['x'], data[name], label=name)
plt.xlabel('forecast value − truth (dataset original scale)')
plt.ylabel('full-test mean CDF')
plt.legend()
plt.show()
```

横坐标在所有方法和两个通道上均是数值完全相同的 `z - y_true`，先反转训练归一化，不对方法单独缩放；数据源未提供可确认的速度物理单位。固定网格为 −0.35 至 +0.35，共 701 点，边界外的质量不被丢弃，会正确反映在端点 CDF 中。样本数按轨迹、窗口、未来帧和网格点计数，不是独立同分布样本数。

冷启动窗口起点 0 只用于揭示整个 20 帧后的第一次在线更新，不计入平均；后续起点 20、40、…、820 的全部测试轨迹和空间点均计入。每条轨迹及每种方法使用各自的校准状态，读取真值只发生在预测已经发出后。

XPF16-1 与 XPF16-0.2 分别使用独立训练头输出的 98 和 498 个概率点（不含中位数），先逐位置按数值重排以修复跨模型交叉，随后对分位函数做线性插值。超出最外分位点的概率质量分别放在最外预测值处，这是一项明确的端点原子假设。Noise 方法直接对各自在线尺度校准后的实际随机样本计算经验 CDF，不假设高斯分布；8、32、256、2048 条路径各有独立校准状态。该校准是原研究的 90% 代理尺度校准，不能声称其确保所有概率水平覆盖。

## 复现

复现需要原始 HDF5 数据和 CUDA；不需要重新训练。先从仓库根目录重建分片权重，再调用独立重放脚本：

```bash
python research/xpf16_02grid_2048/restore_result_parts.py results/reliability/xpf16_02grid_2048 --output-directory /tmp/dipred_reconstructed
mkdir -p /tmp/dipred_grid_weights
tar -xzf /tmp/dipred_reconstructed/xpf16_grid_250_weights.tar.gz -C /tmp/dipred_grid_weights
python research/xpf16_02grid_2048/cdf/test_cdf_stream.py
TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 python research/xpf16_02grid_2048/cdf/replay_cdf.py --data-dir /path/to/train_real --weights-dir /tmp/dipred_grid_weights --output /tmp/dipred_cdf
```

脚本使用仓库中的测试清单、冻结 DiPred、全部 250 个独立头及各自的校准初始状态，并只在完整窗口揭示后更新状态；每一窗口的原始预测仅保留在内存中。由于底层 GPU 批运算的浮点舍入差异，不同设备上的结果可能有微小差别。测试轨迹在先前研究中已经使用，结果是探索性分析。
