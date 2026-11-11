# Research source archive

The original fitting, evaluation and actual ensemble replay scripts are retained
for provenance. Replaying them requires the original workspace, RealPDE data,
frozen model/checkpoint dependencies and cached 20-frame features. These scripts
are not a standalone dataset-free training pipeline. The 19-quantile inference
entry point is `reliability/quantiles19_predict.py`. Aggregated released evidence
is independently checkable using `python analyze_quantiles19_native_crps.py`.

`training_source_original.py` is the exact source hashed at training. The later
`experiment_xpf16_quantiles19.py` adds evaluation-only resumption and fixes a
strict-versus-inclusive lower-tail tie check after training; it did not retrain
the head. Both versions and the failed first attempt's explanation are retained.
