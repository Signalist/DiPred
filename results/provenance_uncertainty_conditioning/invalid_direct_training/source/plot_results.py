"""Plot fixed development candidate tradeoffs; no metric or selection changes."""
from pathlib import Path
import os,csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(os.environ.get('DIPRED_CONDITION_OUT',str(ROOT/'results/provenance_uncertainty_conditioning')))
rows=list(csv.DictReader((OUT/'selection_candidates.csv').open()))
fig,axes=plt.subplots(1,2,figsize=(10.5,3.5),constrained_layout=True)
colors={'C_aug':'#2962a3','C_mask':'#d67a16','C_unc':'#27855a'}
for arm,color in colors.items():
    for seed,style in [('42','-'),('43','--'),('44',':')]:
        rr=sorted([r for r in rows if r['arm']==arm and r['seed']==seed],key=lambda r:int(r['step']))
        steps=[int(r['step']) for r in rr];base=float(rr[0]['development_restart_crps'])
        axes[0].plot(steps,[100*(float(r['development_restart_crps'])/base-1) for r in rr],style,color=color,marker='o',ms=3,label=f'{arm}, seed {seed}')
        axes[1].plot(steps,[100*(max(float(r[k]) for k in r if k.startswith(('mse_k','smse_k')))-1) for r in rr],style,color=color,marker='o',ms=3)
axes[0].axhline(0,color='black',lw=.9);axes[1].axhline(1,color='black',ls='--',lw=.9,label='1% protection limit')
axes[0].set_ylabel('Restart CRPS change vs native (%)');axes[1].set_ylabel('Worst MSE / SMSE increase (%)')
axes[0].set_title('All nonzero candidates worsen selection CRPS');axes[1].set_title('Protection checks: direct + three restart k')
for ax in axes:ax.set_xlabel('Optimizer updates');ax.set_xticks([0,250,500,750,1000]);ax.grid(alpha=.2)
axes[0].legend(fontsize=6,ncol=3);axes[1].legend(fontsize=8)
file=OUT/'development_tradeoffs.png';assert not file.exists();fig.savefig(file,dpi=180);plt.close(fig)
print('PLOTTED all36 nonzero development candidates; no test-based selection')
