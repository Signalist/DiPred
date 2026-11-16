"""Recreate the two publication figures directly from the delivered CSVs."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('directory', type=Path, help='directory containing the delivered CSVs')
args = parser.parse_args()
COLORS = {'X':'#777777', 'P':'#9467bd', 'F':'#0072b2', 'XP':'#b59a00', 'XF':'#009e73', 'PF':'#cc79a7', 'XPF':'#d55e00'}
d = args.directory
w = pd.read_csv(d / 'seven_features_per_window.csv')
n = pd.read_csv(d / 'noise_per_window.csv')
c = w[w['mode'] == 'online'].groupby(['method', 'window_index']).crps.mean().unstack(0)
assert c.shape == (41, 7) and np.isfinite(c.to_numpy()).all()
plt.figure(figsize=(11, 6))
for method in ['X', 'P', 'F', 'XP', 'XF', 'PF', 'XPF']:
    plt.plot(c.index, c[method], label=method, color=COLORS[method], linewidth=2.5 if method == 'XPF' else 1.4)
plt.xlim(2, 42); plt.xlabel('Window Index'); plt.ylabel('Standardized CRPS'); plt.legend(ncol=4); plt.tight_layout()
for ext in ('png', 'pdf'):
    plt.savefig(d / f'Seven_Features_CRPS_by_Window.{ext}', dpi=300)
plt.close()
plt.figure(figsize=(11, 6))
for ni, method in enumerate(['8-noise', '32-noise', '256-noise', '2048-noise']):
    s = n[(n.method == method) & (n['mode'] == 'online') & (n.estimator == 'empirical')]
    v = s.groupby('window_index').standardized_crps.mean()
    assert len(v) == 41 and np.isfinite(v).all()
    note = ' (250-grid run, proxy online)' if method == '2048-noise' else ' (proxy online)'
    plt.plot(v.index, v.values, label=method + note, color=['#333333','#666666','#999999','#444444'][ni], linestyle=['-','--',':','-.'][ni], linewidth=1.25)
for method in ('XPF', 'F'):
    plt.plot(c.index, c[method], label=method + ' (two-tail online)', color=COLORS[method], linewidth=2.5)
plt.xlim(2, 42); plt.xlabel('Window Index'); plt.ylabel('Standardized CRPS'); plt.legend(fontsize=8); plt.tight_layout()
for ext in ('png', 'pdf'):
    plt.savefig(d / f'Noise_XPF_F_CRPS_by_Window.{ext}', dpi=300)
plt.close()
