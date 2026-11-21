# 审查文件清单

建议先读DECISION_REPORT.md和STATUS.md，再读PROTOCOL.md/config.json、PREFLIGHT.json、COMPLETE.json、VERIFICATION.json。原始小型统计在本目录，约23MiB；大缓存4.61GiB不进Git，其SHA256、来源与绝对路径由cache_manifest.json保留。

关键统计入口：quality_paired.csv（质量配对及双区间）、response_intervals.csv（响应平方与交叉项）、signed_mse_by_trajectory.csv（逐轨迹有符号分解）、jvp_full_amplitude_criterion.csv（预设逐轨迹阈值）、jvp_intervals.csv（方向诊断的轨迹区间）。*_by_Re和lead_channel_*保留所有指定分层，不能仅看总体表。

以下为本轮交付文件及大小。ARTIFACT_SHA256.json列出代码、报告、配置、日志及表格的实际哈希；该文件自身不自引用。

| 文件 | bytes |
| --- | ---: |
| `research/restart_input_attribution/README.md` | 4053 |
| `research/restart_input_attribution/analyze.py` | 4793 |
| `research/restart_input_attribution/core.py` | 6303 |
| `research/restart_input_attribution/preflight.py` | 6201 |
| `research/restart_input_attribution/run.py` | 9599 |
| `research/restart_input_attribution/verify.py` | 6845 |
| `results/restart_input_attribution/ADDITIONAL_AUDIT.json` | 313 |
| `results/restart_input_attribution/ANALYSIS.json` | 324 |
| `results/restart_input_attribution/COMPLETE.json` | 743 |
| `results/restart_input_attribution/DECISION_REPORT.md` | 16471 |
| `results/restart_input_attribution/EXECUTED_COMMANDS.json` | 1346 |
| `results/restart_input_attribution/FROZEN.json` | 1029 |
| `results/restart_input_attribution/PREFLIGHT.json` | 52395 |
| `results/restart_input_attribution/PROTOCOL.md` | 6627 |
| `results/restart_input_attribution/RUN_STARTED.json` | 204 |
| `results/restart_input_attribution/STATUS.md` | 5572 |
| `results/restart_input_attribution/TASK.md` | 8872 |
| `results/restart_input_attribution/VERIFICATION.json` | 739 |
| `results/restart_input_attribution/analyze.log` | 69 |
| `results/restart_input_attribution/cache_manifest.json` | 2484504 |
| `results/restart_input_attribution/config.json` | 43812 |
| `results/restart_input_attribution/environment.json` | 119 |
| `results/restart_input_attribution/events.jsonl` | 240594 |
| `results/restart_input_attribution/freeze.log` | 7 |
| `results/restart_input_attribution/jvp.csv` | 469454 |
| `results/restart_input_attribution/jvp_by_Re.csv` | 22458 |
| `results/restart_input_attribution/jvp_by_trajectory.csv` | 94323 |
| `results/restart_input_attribution/jvp_full_amplitude_criterion.csv` | 4778 |
| `results/restart_input_attribution/jvp_intervals.csv` | 46728 |
| `results/restart_input_attribution/jvp_intervals_by_Re.csv` | 144093 |
| `results/restart_input_attribution/jvp_summary.csv` | 7359 |
| `results/restart_input_attribution/lead_channel.csv` | 7602228 |
| `results/restart_input_attribution/lead_channel_by_Re.csv` | 430934 |
| `results/restart_input_attribution/lead_channel_by_trajectory.csv` | 1817738 |
| `results/restart_input_attribution/lead_channel_summary.csv` | 139804 |
| `results/restart_input_attribution/plan.json` | 107991 |
| `results/restart_input_attribution/preflight.log` | 1470 |
| `results/restart_input_attribution/preflight_directions.csv` | 514 |
| `results/restart_input_attribution/preflight_fd.csv` | 7312 |
| `results/restart_input_attribution/preflight_increment_gradients.csv` | 512 |
| `results/restart_input_attribution/preflight_replay.csv` | 164 |
| `results/restart_input_attribution/quality.csv` | 780407 |
| `results/restart_input_attribution/quality_by_Re.csv` | 45102 |
| `results/restart_input_attribution/quality_by_trajectory.csv` | 190017 |
| `results/restart_input_attribution/quality_differences.csv` | 844176 |
| `results/restart_input_attribution/quality_differences_by_trajectory.csv` | 205978 |
| `results/restart_input_attribution/quality_intervals.csv` | 90045 |
| `results/restart_input_attribution/quality_intervals_by_Re.csv` | 276957 |
| `results/restart_input_attribution/quality_paired.csv` | 98965 |
| `results/restart_input_attribution/quality_paired_by_Re.csv` | 303888 |
| `results/restart_input_attribution/quality_summary.csv` | 14733 |
| `results/restart_input_attribution/response_by_Re.csv` | 11480 |
| `results/restart_input_attribution/response_by_trajectory.csv` | 48050 |
| `results/restart_input_attribution/response_ensemble.csv` | 161006 |
| `results/restart_input_attribution/response_ensemble_by_Re.csv` | 9183 |
| `results/restart_input_attribution/response_ensemble_by_trajectory.csv` | 38967 |
| `results/restart_input_attribution/response_ensemble_intervals.csv` | 17273 |
| `results/restart_input_attribution/response_ensemble_intervals_by_Re.csv` | 53147 |
| `results/restart_input_attribution/response_ensemble_summary.csv` | 2986 |
| `results/restart_input_attribution/response_intervals.csv` | 22510 |
| `results/restart_input_attribution/response_intervals_by_Re.csv` | 69385 |
| `results/restart_input_attribution/response_members.csv` | 6321703 |
| `results/restart_input_attribution/response_summary.csv` | 3787 |
| `results/restart_input_attribution/run.log` | 1539 |
| `results/restart_input_attribution/signed_mse.csv` | 225012 |
| `results/restart_input_attribution/signed_mse_by_Re.csv` | 12985 |
| `results/restart_input_attribution/signed_mse_by_trajectory.csv` | 55323 |
| `results/restart_input_attribution/signed_mse_intervals.csv` | 23195 |
| `results/restart_input_attribution/signed_mse_intervals_by_Re.csv` | 70431 |
| `results/restart_input_attribution/signed_mse_summary.csv` | 4181 |
| `results/restart_input_attribution/verify.log` | 190 |
