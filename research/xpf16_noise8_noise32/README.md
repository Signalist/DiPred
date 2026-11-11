# Frozen research source archive

These readable sources record the exact local evaluation, shared online controller
and CRPS formulas. Full GPU replay requires the original project environment,
private source dependencies, dataset, frozen checkpoints and historical caches;
these files are not a standalone training or inference package. The evaluation's
`OriginalProvider` comes from the historical denoising-view study, but only its
frozen original F0/XPF16 head is used. No additional hidden-stage variants are
included in this comparison.

For dataset-free reproduction of all published aggregate results and counterexamples,
run `python analyze_xpf16_noise8_noise32.py` from the repository root. Existing
portable XPF16/32-noise inference remains `reliability/selected16_predict.py`;
this comparison-only release does not add an8-noise deployment CLI.
