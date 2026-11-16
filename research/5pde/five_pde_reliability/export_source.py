"""Create a self-contained, portable source snapshot for the result release."""
import ast
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / 'github/DiPred/research/5pde'
DEST.mkdir(parents=True, exist_ok=True)
package = DEST / 'five_pde_reliability'
package.mkdir(exist_ok=True)
for source in Path(__file__).parent.glob('*.py'):
    shutil.copy2(source, package / source.name)
for name in ('run_all.sh',):
    shutil.copy2(Path(__file__).parent / name, package / name)
(p := package / 'run_all.sh').write_text(p.read_text().replace('source env.sh', 'if [[ -f env.sh ]]; then source env.sh; fi').replace('github/DiPred/results/reliability/5pde/', '../../results/reliability/5pde/'))
(package / '__init__.py').touch()
for name in ('fair_x_xp_xpf', 'xpf16_02grid_2048'):
    (DEST / name).mkdir(exist_ok=True)
    (DEST / name / '__init__.py').touch()
shutil.copy2(ROOT / 'fair_x_xp_xpf/model.py', DEST / 'fair_x_xp_xpf/model.py')
for name in ('calibration_grid.py', 'distribution_grid.py'):
    shutil.copy2(ROOT / 'xpf16_02grid_2048' / name, DEST / 'xpf16_02grid_2048' / name)
source = (ROOT / 'fair_x_xp_xpf/evaluate.py').read_text()
parsed = ast.parse(source)
selected = {'pair_quantiles', 'sorted_cdf_sum', 'quantile_crps', 'bounds'}
segments = [ast.get_source_segment(source, node) for node in parsed.body
            if isinstance(node, ast.FunctionDef) and node.name in selected]
metrics = ('import torch\nfrom calibration_grid import per_level_quantiles\n'
           'LEVELS = torch.arange(1, 500, dtype=torch.float64) / 500\nALPHA = LEVELS[:249]\n'
           'def _add_at(acc, indices, values):\n'
           '    acc.scatter_add_(0, indices.reshape(-1), values.reshape(-1).to(torch.float64))\n\n')
(DEST / 'fair_x_xp_xpf/evaluate.py').write_text(metrics + '\n\n'.join(segments) + '\n')
p = package / 'common.py'
s = p.read_text().replace("sys.path.insert(0,str(ROOT/'github/DiPred'))", "sys.path.insert(0,str(ROOT))\nsys.path.insert(0,str(ROOT.parents[1]))")
p.write_text(s)
for name in ('index_data.py', 'prepare_data.py'):
    p = package / name
    s = p.read_text()
    if name == 'index_data.py':
        s = s.replace('import json\n', 'import json\nimport os\n')
    s = s.replace("DATA=ROOT.parent/'OLD/DATASET-5PDE'", "DATA=Path(os.environ['FIVE_PDE_DATA']).resolve()")
    p.write_text(s)
p = package / 'summarize.py'
s = p.read_text().replace("DEST=ROOT/'github/DiPred/results/reliability/5pde'", "DEST=ROOT.parents[1]/'results/reliability/5pde'")
s = s.replace("(ROOT.parent/'OLD/DATASET-5PDE/download_manifest.json')", "(Path(os.environ['FIVE_PDE_DATA'])/'download_manifest.json')")
s = s.replace('import argparse,csv,hashlib,json,shutil,sys', 'import argparse,csv,hashlib,json,shutil,sys,os')
p.write_text(s)
p = package / 'release.py'
s = p.read_text().replace("DEST=ROOT/'github/DiPred/results/reliability/5pde'", "DEST=ROOT.parents[1]/'results/reliability/5pde'")
s = s.replace('research/5pde/verify_release.py', 'research/5pde/five_pde_reliability/verify_release.py')
p.write_text(s)
(DEST / 'README.md').write_text("""# Five-PDE reliability experiments

The six methods are X, XP, XPF, 8-noise, 32-noise and 256-noise. Data are
read-only; intermediate data and checkpoints may be very large. Set
`FIVE_PDE_DATA` to the original dataset root and install PyTorch, NumPy and
PyArrow. All commands below run from this directory, `research/5pde`.

```bash
export FIVE_PDE_DATA=/path/to/DATASET-5PDE
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
python five_pde_reliability/index_data.py
python five_pde_reliability/prepare_data.py
python five_pde_reliability/test_noise.py
for name in cylinder controlled_cylinder fsi foil combustion; do
  python five_pde_reliability/audit_fields.py "$name"
  python five_pde_reliability/train_backbone.py "$name"
  python five_pde_reliability/head_data.py "$name"
  python five_pde_reliability/train_heads.py "$name"
done
bash five_pde_reliability/run_all.sh
```

The runner resumes completed model evaluations and stops on audit failure.
Training checkpoints resume at the most recent saved step. Published
`config.json` and the scenario report contain the split and score details.
Independent verification of result tables requires no original data:

```bash
python research/5pde/five_pde_reliability/verify_release.py \
  results/reliability/5pde/<scenario>
```

Run this last command from the repository root. A CUDA GPU is required for
training and full evaluation; all trajectory bootstrap samples are reproducible.
""")
print('SOURCE_EXPORTED', DEST)
