"""Check report rendering against isolated synthetic summaries."""
from pathlib import Path
import release_report
from train49 import OUT

root=OUT/'summary_smoketest'
(root/'independent_audit.json').write_text('{"passed": true, "synthetic_test_only": true}')
release_report.OUT=root
release_report.main()
assert (root/'summary/REPORT.md').exists()
print('REPORT_SMOKE_PASSED: synthetic input only')
