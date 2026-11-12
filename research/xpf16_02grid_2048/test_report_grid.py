"""Report generation test, using only isolated copied smoke arrays."""
import report_grid
from train_grid import OUT
root=OUT/'summary_smoketest'
(root/'independent_audit.json').write_text('{"passed":true,"synthetic_test_only":true}')
report_grid.OUT=root
report_grid.main()
assert (root/'summary/REPORT.md').exists()
print('GRID_REPORT_SMOKE_PASSED; synthetic report is not a study result')
