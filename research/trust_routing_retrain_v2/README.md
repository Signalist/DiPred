# Trust routing retrain v2

```bash
export PYTHONPATH="$PWD/../Forecast/Runtime/.deps${PYTHONPATH:+:$PYTHONPATH}"
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
python research/trust_routing_retrain_v2/train.py
python research/trust_routing_retrain_v2/verify.py
```

入口只导入并复用 v2 审计的数据键、风险准备、噪声规则、冻结主干和两种路由器。新FP32开发原生缓存位于 `.cache/trust_routing_retrain_v2/development_native_fp32.pt`，不提交大张量；结果、权重、逐轨迹表和中文报告位于 `results/trust_routing_retrain_v2/`。两模型均选择原生时不生成测试结果。
