# Error-guided spatial routing

从仓库根目录运行：

```bash
source ../Forecast/Runtime/env.sh
python research/error_guided_spatial_routing/tests.py
python research/error_guided_spatial_routing/run.py all
python research/error_guided_spatial_routing/verify.py
```

`prepare.py` 在缓存身份匹配时复用 400/420 原生路径，并为协议指定的 440/460/480 起点生成冻结 32 成员路径；缓存默认位于 `.cache/error_guided_spatial_routing`。结果、日志、候选权重和中文决策报告位于 `results/error_guided_spatial_routing/`。
