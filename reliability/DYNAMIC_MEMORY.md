# 动态风险记忆支线

推荐 `warp/memory_dynamic`，入口 `reliability/dynamic_predict.py`。固定四步 DiPred `[99,66,33,0]`，20帧输入，一次联合生成紧接20帧；没有10+10分块、32路径采样或点预测残差校正。权重和初始校准均已包含。

```bash
pip install -r requirements.txt
TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 python -m reliability.dynamic_predict \
  --input history.pt --output forecast.pt \
  --geometry 0.0034 -0.0034 0.02 --objective warp --strategy memory_dynamic
python analyze_dynamic_memory.py
TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 python -m reliability.verify_dynamic --device cpu
```

几何值只是示例，须替换为实际带符号dx、dy和正dt。输入是原数值尺度float32 `[B,2,20,32,64]`；输出prediction、lower、upper、risk_half_width同形状、同原数值尺度。CLI仅为冷启动；以下接口用于连续观测反馈：

```python
from reliability.dynamic_predict import DynamicRiskMemoryPredictor
model = DynamicRiskMemoryPredictor(geometry, objective="warp", strategy="memory_dynamic")
result = model.predict(history)  # 当前未来真值尚不可见
# 完整下一20帧揭示以后：
model.reveal(truth)
result_next = model.predict(truth)
model.reset()  # 切换轨迹身份或批次顺序之前
```

整个20帧揭示以后才可reveal，不允许重复预测、部分反馈或跨轨迹续接；上一误差使用实际保存的基础预测。网络在线冻结，仅校准状态更新。冷启动没有历史误差；初始校准逐候选重新计算，因此最终冷启动区间不保证与旧版本相同。

固定48,340参数风险核心，新增10,430参数模块处理上一窗口绝对误差的潜特征；共58,770风险参数。`warp`按全部20个未来步预测有界4像素位移并做双线性采样；`modulate`是相同结构、参数量、初始化和训练预算的纯幅度调整对照。它们既不修改点预测，也不表示物理流速或质量守恒输运。819训练对、2000更新、seed42；校准10个独立身份，开发/测试身份未用于拟合。共享预测器和原风险头权重均不变。

测试窗口2–39、Re/channel等权：Score从上次发布0.620441834降至0.620079902，约0.05833%；Width从0.474026420降至0.473896527。八项测试汇总均改善，但原96项仍83改善/1持平/12退步；相对同容量modulate仍44项退步。原机制门槛失败，不能归因于迁移独有优势。点MSE仍约1.28793e-4。

Score/Width是训练通道标准化无量纲指标；预测数组保留原数值尺度。90%逐点覆盖不代表完整窗口联合覆盖，后者实测为0。数据反复开发使用，结果是探索性，不能声称独立泛化、显著性或严格覆盖保证。

完整11232窗口/通道记录、61参考组合、366组96项比较、9536反例、逐轨迹分析和图位于 `results/reliability/dynamic_memory/`。运行标准库脚本 `python analyze_dynamic_memory.py` 可独立重算，不需要原数据。`research/dynamic_memory/` 为原研究源码归档，依赖原研究环境；可移植入口及权重是 `reliability/`。原始PDE数据和大型预测缓存未上传。
