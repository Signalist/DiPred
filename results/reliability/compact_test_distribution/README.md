# 测试集逐窗口评分与场快照

完整测试集逐窗口结果可直接下载和分析：

- [8/32-noise 原生 Ensemble CRPS](ensemble_crps_per_window.csv)
- [19 个分位点的 Pinball 与经验覆盖](quantile_pinball_per_window.csv)
- [5%–95% 截断分位评分](quantile_score_05_95_per_window.csv)，不是完整 CRPS
- [完整窗口索引](window_index.csv)、[3 个快照索引](snapshot_index.csv)、[尺度与说明](metadata.json)

3 个不同 Re 工况的对齐真实场、32 路扩散预测、19 分位预测快照，连同上述表格与说明，保存在 [parts](parts/) 内的 16 个分片中。克隆仓库后，从仓库根目录运行以下命令，会校验所有分片并在当前目录重组 ZIP：

```bash
python results/reliability/compact_test_distribution/reassemble.py
```

最终 ZIP 为 46,748,151 字节（约 44.6 MiB），SHA-256 为 `50d81b13fc2f975b805ffc1fed0ebf0e0c005bedc178e0bb93a4326027089c01`；包内另有逐文件 `SHA256SUMS.txt`。分片用于适应大文件上传环境，重组后与原精简包逐字节一致。

逐窗口统计覆盖 12 条测试轨迹 × 42 个完整预测窗口。0–760 的 39 个窗口为既往评估范围，780、800、820 为补充的完整尾部窗口；帧 860–867 不足以组成新的 20 帧目标。`window_index.csv` 和 `snapshot_index.csv` 保存轨迹、窗口与 Re 等对应关系。

快照数组为原始物理场数据尺度，SI 单位未独立确认；前 8 条路径即 8-noise。XPF 分位数使用因果在线校准，扩散样本为未在线校准的原生随机预测，二者的在线反馈条件不同。

快照仅覆盖 3 个样本，无法重算完整测试集逐点 PIT、完整分布 CRPS 或任意新的全场统计。19 个分位节点不指定两侧尾部和节点间插值；计算 PIT/CRPS 时需明确这些假设。历史测试轨迹已反复用于研究。
