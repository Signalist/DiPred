# 训练与评估源码溯源

这里保存冻结实验源码，哈希在`source_hashes.json`，对应`metadata/reliability/continued_quantile/`的协议。文件保留原路径和导入，用于核对训练目标、批次、数据身份、校准顺序及评估；它们依赖未完整上传的本地研究缓存和辅助模块，不能当作仓库内一键重训入口。

`experiment_distribution_error_risk.py`为本次819对样本、2000次额外更新的匹配训练；`distribution_error_risk.py`定义Gaussian/Student全边际NLL。`experiment_point_error_scale_alignment.py`记录上一轮初始化来源。原密度门槛失败记录保留，`experiment_random_window_continuation_followup.py`是纯分位数控制的独立后续评估。

可直接运行的独立推理入口为`reliability/continued_predict.py`，只依赖仓库内代码和已包含权重；所有CSV比较可运行`analyze_continued_reliability.py`复算，不需要原始数据。
