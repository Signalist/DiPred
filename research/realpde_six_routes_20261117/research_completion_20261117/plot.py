"""Standalone scientific overview; each panel keeps its own metric and baseline."""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

W = Path(__file__).resolve().parent.parent
OUT = W / 'runs/research_completion_20261117'
d = pd.read_csv(OUT / 'recomputed_gains.csv')
plt.rcParams.update({'font.size': 10, 'axes.titlesize': 11})
fig, ax = plt.subplots(2, 3, figsize=(14, 8.2))


def error_panel(a, frame, title, ylabel, labels, note):
    q = frame.set_index('split').loc[labels]
    x = np.arange(len(q))
    y = q.gain_percent.to_numpy()
    # CIs need not contain a ratio-of-means estimate in general; here all do.
    lo, hi = q.descriptive_ci_low.to_numpy(), q.descriptive_ci_high.to_numpy()
    a.errorbar(x, y, yerr=np.stack([y-lo, hi-y]), fmt='o', capsize=4, color='#155e75')
    a.axhline(0, color='#888888', lw=.8)
    a.set_xticks(x)
    a.set_xticklabels(['dev' if s == 'development' else s for s in labels])
    a.set_title(title, loc='left', fontweight='bold')
    a.set_ylabel(ylabel)
    a.grid(axis='y', alpha=.18)
    a.text(.02, .97, note, transform=a.transAxes, va='top', fontsize=8,
           bbox=dict(facecolor='white', edgecolor='none', alpha=.85))
    a.margins(y=.3)


splits = ['development', 'late', 'future']
error_panel(ax[0,0], d[(d.route == 'structured') & (d.reference == 'rescrct_original')],
            '1. Structured field correction', 'MSE gain vs cost-matched ResCrct, %', splits,
            'Observed history only; free rollout worsens')
t = pd.read_csv(W / 'runs/lowcost_joint_20261117/conditional_scores/scorefit/timing.csv').set_index('method')
a = ax[0,1]
values = [t.loc[m, 'median_ms'] for m in ['ar__mixed', 'smooth']]
a.bar(['AR + PCA', 'smooth'], values, color=['#155e75', '#8b98a8'], width=.55)
for i, v in enumerate(values):
    a.text(i, v+7, f'{v:.2f}', ha='center')
a.set_ylim(0, 340)
a.set_title('2. Low-cost joint distribution', loc='left', fontweight='bold')
a.set_ylabel('Measured batch-1 latency, ms')
a.text(.02, .97, '7.02x faster; max-speed CRPS worsens\nMacro tolerance is not statistical equivalence',
       transform=a.transAxes, va='top', fontsize=8)
a.grid(axis='y', alpha=.18)
error_panel(ax[0,2], d[d.route == 'injected_geometry'], '3. Delayed online error state',
            'Post-fault MSE gain vs faulted P, %', ['step', 'ramp'],
            'Injected forecast geometry; physical flow unchanged')
error_panel(ax[1,0], d[(d.route == 'observation_equal8') & (d.method == 'critic')],
            '4. Sparse active observation', 'Future-15 MSE gain vs risk-8, %', splits,
            'Eight new sensor sites; no stable VOI advantage')
error_panel(ax[1,1], d[(d.route == 'allocation_half') & (d.reference == 'random_expected')],
            '5. Expected-benefit compute routing', 'Functional CRPS gain vs equal random, %', splits,
            '50% expensive quota; dev/late CIs cross zero')
error_panel(ax[1,2], d[(d.route == 'task_error') & (d.method == 'direct/F_det4_1500/16')],
            '6. Empirical task-error readout', 'Four-task normalized MSE gain vs J(P), %', splits,
            'Region-u worsens; no field or PDE certificate claim')
fig.suptitle('RealPDE: six bounded explorations, single training seed 42', fontsize=16, y=.99)
fig.text(.5, .015, '95% CIs describe paired trajectories on reused data. Late/future share dev identities. Panel metrics are different; do not rank routes by percent.',
         ha='center', fontsize=9)
fig.tight_layout(rect=[0, .045, 1, .96], h_pad=3, w_pad=2.2)
fig.savefig(OUT / 'overview.png', dpi=180)
fig.savefig(OUT / 'overview.pdf')
plt.close(fig)
