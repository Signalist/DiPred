# Implementation audit notes

The frozen original PROTOCOL.md hash is stored in selection.json. The following implementation fixes do not refit or reselect the router:

- The single seed42 random realization uses SHA256(seed42, request id, sampling seed) to generate stateless priorities. Cached replay and the public live API now produce identical actions regardless of batch call order. The first scratch replay used a sequential random stream; that draw was not comparable to a stateless batch API. Exact expected-random loss, the main random control, is unaffected. Model choices and benefit/risk/width actions are unchanged.
- Initial sequential timings showed GPU clock/load drift across methods. Keep timing_sequential_diagnostic.csv; final timing interleaves seven repeats of all policies/budgets in a seed42 shuffled order. This seed controls measurement order, not training.
- The primary comparison remains all predeclared budgets. The 50% summary is not selected from held-out outcomes. Raw/fixed/online masks remain identical.
