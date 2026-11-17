# Low-cost identification control

The actual80-step registration cost is about87ms versus8.5ms for issuing a forecast, and the nonlinear geometry gains are small. Test the linear tangent identification already named in GEOMETRIC_PROTOCOL.md. It solves the7×7 regularized normal equation in the current P derivatives and amplitude/bias directions, clips to the identical coefficient bounds, and applies the same nonlinear warp at the next issue. Past truth remains available only after full20 reveal.

Use exactly the same seven-state families, train/cal split, seed42, calibration loss, shrink grid, and natural evaluation periods. Fit train scaling and VAR using train tangent labels; select on natural calibration before reading this variant's development results. Actual single-window direct inference is authoritative for all four evaluated splits from the outset. Keep80-step registration results as a separate cost/quality control. Do not select the cheaper method on heldout performance or call it a new algorithm.

Run direct streaming, prefix/isolation replay audit and cold/warm200-frame no-feedback rollout. All as-yet-unfinished probability, SPCI/AcMCP and controlled drift requirements remain outstanding.
