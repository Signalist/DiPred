# 任务状态

完成旧共同位置修正支线归档和独立原生重启一致性探针；无未完成正式数据、无资源阻碍。

实际起始HEAD：76e3ec27416a0afeee0ee379d86b2ed7f7e56ed0；分支dynamics。工作目录：/cpfs01/projects-HDD/cfff-3469a2cbe57f_HDD/wangxiyun/DiPred+ResCrct/Workspace/github/DiPred。交付提交号可用 `git log -1 --format=%H -- results/native_restart_consistency/STATUS.md` 查询，最终远端提交号由最终回复核实。

- 旧结论归档：EVIDENCE_LEDGER.md，旧gate/权重/结果保持不变。
- 固定协议：PROTOCOL.md/config.json，k2/5/10、M32、两独立重复、70原轨迹、400/420窗口。
- 预检：6项单测；两条训练轨迹的原生入口重放、k0及缓存检查；冻结代码后才正式评分。
- 正式执行：280组、53760路径、1680批sampler调用、6720网络前向，全部完成。
- 统计：840分布比较、10080质量行、96320逐lead/通道行；逐轨迹/重复/k/Re及原轨迹bootstrap。
- 结论：分布超出采样差异支持；与原生难度有关，但新增诊断价值、因果归因和一致性训练效益未证明。
- 核验：verification.json/verify.log、cache_manifest.json、data_hashes.json、events.jsonl、FROZEN.json。
- 审查入口：DECISION_REPORT.md、distribution_intervals.csv、quality_paired.csv、associations.csv、history_summary.csv。
- 实际命令与复跑：REPRODUCE.md；日志tests/preflight/freeze/formal/analyze/verify.log。
- 大缓存：Workspace/runs/native_restart_consistency，约17.7GB，不随Git上传；源数据和旧主干依赖见复跑说明。

本轮没有训练任何网络；没有追加重启点、噪声种子或旧extension。缺少新独立轨迹、更充分状态控制及诊断增量验证，属于解释和未来研究缺口，不是本轮执行未完成项。没有自动启动后续研究。
