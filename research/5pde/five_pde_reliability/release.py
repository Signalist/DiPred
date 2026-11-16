"""Create a factual scenario report and complete release SHA-256 manifest."""
import argparse,csv,hashlib,json,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'runs/five_pde_reliability_20261114'
DEST=ROOT.parents[1]/'results/reliability/5pde'


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda:handle.read(1048576),b''):h.update(block)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('scenario');args=parser.parse_args();name=args.scenario;dest=DEST/name
    shutil.copy2(OUT/'field_audit'/f'{name}.json',dest/'field_audit.json')
    shutil.copy2(OUT/'audit'/f'{name}.json',dest/'audit.json')
    shutil.copy2(OUT/'noise_audit'/f'{name}.json',dest/'noise_audit.json')
    config=json.loads((dest/'config.json').read_text());audit=json.loads((dest/'audit.json').read_text())
    assert audit['passed']
    with (dest/'summary.csv').open(newline='') as handle:rows=list(csv.DictReader(handle))
    text=['# Five-PDE reliability experiment: '+name,'',
        f"Real-observation dataset only. Train/development/calibration/test trajectories: "+', '.join(f"{k}={sum(r['split']==k for r in config['source_split'])}" for k in ('train','development','calibration','test'))+'.',
        f"Evaluated {config['test_trajectories']} test trajectories, {config['full_test_windows_per_trajectory']} valid non-cold-start windows per trajectory, 20 future frames per window.",
        'One scenario-specific diffusion predictor is trained from scratch on training trajectories, then frozen. X, XP and XPF use the same quantile head, X-last-frame reference, training data, batch sequence, optimizer and three matched initialization seeds.',
        'Noise baselines are actual four-step diffusion trajectories; 8 and 32 paths are prefixes of 256 independently seeded paths. The same empirical quantile interpolation, fixed CQR and causal online calibration are applied, and finite-sample tail quantiles are inherently coarse.',
        'The shared head uses 12 spatiotemporal input channels: X=2, P=2 and an 8-channel projection of the 48-channel F; this is the fair shared-head design, not the historical XPF16 head. Missing P/F slots are masked.',
        'P and F are derived only from the observed history. F is the final hidden feature of the fourth actual denoising call. For combustion the unused second padded channel is excluded from losses and metrics.',
        'Original stored fields are strided to 32×64; physical coordinate spacing and control parameters are not provided as model inputs. The two source axes may be downsampled by different factors.',
        'Raw is the primary comparison; split CQR and causal online calibration are auxiliary. Calibration pools correlated space/time points and therefore does not by itself guarantee trajectory-level or distribution-shift coverage.',
        'Raw noise CDFs use the actual empirical samples. Fixed/online noise CDFs use calibrated quantile functions and are not claimed to be calibrated random ensembles.',
        'Each trajectory window is issued before its complete target is revealed. The starting window is used only to initialize the online state. Raw quantiles are monotone by construction; calibrated quantiles are monotonically rearranged before scoring.',
        'CRPS uses the same 499 levels at .002 spacing with trapezoidal 2×pinball integration and clamped tails. CDF x axes are in original data units, the same for all arms and calibration methods within the scenario.',
        'Per-window and 20-column per-frame CSVs are split by model/seed for transfer size; *_index.csv lists every part. All valid windows and field locations are represented.',
        'Bootstrap resamples trajectories; per-window intervals are unadjusted for multiple comparisons, and shared physical-parameter groups and spatial/time dependence limit their interpretation. Earlier project use of these trajectories is not fully known, so the test results are exploratory.',
        'The source archive has real and simulated domains; only the real domain is used here. In FSI, six official entries overlap remain/out-of-distribution lists and six IDs are unlisted; all such IDs are reserved for test.',
        '',f"Audit passed: {audit['passed']}.",'',
        '| Method | Seed | Channel | Raw CRPS | Raw 90% coverage | Raw interval score |',
        '|---|---:|---|---:|---:|---:|']
    for row in rows:
        if row['calibration']=='raw' and row['channel']!='all':
            text.append(f"| {row['method']} | {row['seed']} | {row['channel']} | {float(row['crps']):.8g} | {float(row['coverage90']):.6f} | {float(row['interval_score90']):.8g} |")
    (dest/'REPORT.md').write_text('\n'.join(text)+'\n')
    (dest/'README.md').write_text('''# Result layout

`summary.csv`, `paired_comparison.csv`, `paired_by_window.csv`, `quantile_metrics.csv`, channel-specific `cdf_*.csv`, `config.json`, `manifest.json` and the small prediction snapshot describe the complete held-out test evaluation. `per_window_index.csv` and `per_frame_index.csv` list all twelve model/seed shards (nine trained heads and three sampling baselines) under `runs/`. Per-frame tables have one row per window/channel and columns `crps_frame_01` through `crps_frame_20`; these values round-trip the original float32 scores.

The separate source code is under `research/5pde/`. To check a scenario without the large original PDE data, run `python research/5pde/five_pde_reliability/verify_release.py results/reliability/5pde/<scenario>`. `REPORT.md` explains normalization, the evaluation protocol, and limitations.
''')
    manifest_path=dest/'manifest.json';manifest=json.loads(manifest_path.read_text())
    manifest['files_sha256']={str(path.relative_to(dest)):sha(path) for path in sorted(dest.rglob('*')) if path.is_file() and path!=manifest_path}
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    print('RELEASE_COMPLETE',name,len(manifest['files_sha256']),flush=True)

if __name__=='__main__':main()
