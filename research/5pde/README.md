# Five-PDE reliability experiments

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
python research/5pde/five_pde_reliability/verify_release.py   results/reliability/5pde/<scenario>
```

Run this last command from the repository root. A CUDA GPU is required for
training and full evaluation; all trajectory bootstrap samples are reproducible.
