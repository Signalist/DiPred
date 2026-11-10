# 仅三组：16通道、原28通道、32-noise

入口 `reliability/selected16_predict.py`，三组推理权重及初始校准已包含。默认16通道为真实裁剪，保留历史时间波动2、空间梯度2、预测时间变化2、去噪投影8和坐标2。移除历史/误差投影和位移模块，风险参数34,004；原28风险参数58,770。点预测器冻结，仍为99→66→33→0四次调用，20帧输入、完整后20帧联合预测。

```bash
pip install -r requirements.txt
TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 python -m reliability.selected16_predict \
  --input history.pt --output forecast.pt \
  --geometry 0.0034 -0.0034 0.02 --method selected16
python analyze_selected16_comparison.py
TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 python -m reliability.verify_selected16 --device cpu
```

几何值是示例，须替换为真实带符号dx/dy和正dt。输入为原数值尺度float32 `[B,2,20,32,64]`；prediction/lower/upper/risk_half_width同形状、同原数值尺度。CLI是无历史反馈的冷启动，不能代表完整在线结果。连续窗口用：

```python
from reliability.selected16_predict import RequestedThreePredictor
model = RequestedThreePredictor(geometry, method="selected16")
result = model.predict(history)
# 完整后20帧真值揭示以后：
model.reveal(truth)
next_result = model.predict(truth)
model.reset()  # 换轨迹身份或批次顺序之前
```

另外两组仅换成 `method="original28"` 或 `method="noise32"`，不再加入其他正式对照。原28是上轮完整输入架构的同预算续训版本（819对、2000步、seed42），并非上次发布的旧warp权重。16从相同旧warp保留/切片初始化，使用完全相同训练数据、步数、优化器和批次。校准10个身份独立于拟合，网络冻结后单独拟合各方法初始校准。

三组使用共同零噪声点预测中心。32-noise实际生成32条路径（seed500000–500031），用其标准差构造误差范围并独立校准；不使用32路径均值中心，也不从std虚构经验5/95分位数。16/28为4次主干调用；32-noise加共同中心为128＋4次。没有把缓存读取时间当部署速度，没有测量端到端加速。

测试窗口2–39、Re及u/v等权：

|方法|风险头参数|Score↓|Width↓|90%逐点覆盖|
|---|---:|---:|---:|---:|
|16通道|34004|0.626680|0.477147|89.932%|
|原28通道|58770|0.618357|0.473093|89.933%|
|32-noise|无额外神经头|0.659619|0.479577|89.985%|

16风险参数少42.14%，Score比28差1.346%，比32-noise低4.994%。原8指标×12工况/通道比较，16对28为23改善/73退步，对32为70改善/26退步。16对32区间稍窄，但覆盖及最差区域覆盖较低，不是同覆盖效率或全面优越性证明。完整28仍是三组中平均评分最好者。

16神经头忽略历史误差记忆，在线校准器仍保留原有误差反馈与部分重叠物理特征。返回的memory_available表示上一窗口反馈已可用，不代表16神经头使用该记忆。reveal只接受完整有限20帧，使用实际保存的P计算残差；网络不在线更新。

Score/Width为训练通道标准化无量纲指标，点MSE不变。覆盖是逐点边际量，非整个20帧联合覆盖。子集在重复使用的历史数据上经过前轮消融后确定，结果为探索性，不构成独立统计确认或严格覆盖保证。

`results/reliability/selected16_comparison/`含全部5616条记录、两组原96项比较、全部99条反例、逐轨迹配对及审计。标准库脚本可重算，不需原始数据或权重。`research/selected16_comparison/`是依赖原研究环境的历史源码归档，正式对照仅三组。原始PDE数据、采样缓存和辅助模型权重不上传。
