# 执行与故障记录

所有故障均保留原日志；没有增加训练配置、loss、特征、更新数或测试驱动选择。

- 首次预检的算子 float32/float64 一致性失败。环境的 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE 强制开关覆盖 PyTorch allow_tf32=False，最大误差约8.82e-4。新入口在导入torch前固定该环境变量为0，保持全精度；原失败见 preflight_tf32_failure.log。
- 随后七项预检均通过，但写入哈希证据时相对 __file__ 不能相对绝对仓库路径计算 relative_to。改为 resolve()，日志见 preflight_path_failure.log。
- 初次训练入口在 D/42 第0步写空 training.csv 时抛出 IndexError。没有任何 optimizer.step 执行。唯一checkpoint为D_42_0.pt，空训练表和全部相关文件保存在 preupdate_failure/。修复只是在trace为空时跳过CSV写入。此后七项预检再次通过，才执行九次正式训练；记录中共9000个更新，未额外训练或续训。

成功执行的入口为：

```bash
source ../../env.sh
python research/nonlinear_parameterization_control/tests.py
python -u research/nonlinear_parameterization_control/run.py train
python -u research/nonlinear_parameterization_control/run.py freeze
python -u research/nonlinear_parameterization_control/run.py evaluate
```

后续分析、只读验证及提交状态以 DECISION_REPORT.md、STATUS.md 和实际日志为准。此前旧pilot/phase3/follow-up文件始终只读。
