from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
O=Path(__file__).resolve().parent.parent/'runs/recoverability_20261118'
s=pd.read_csv(O/'primary.csv');c=pd.read_csv(O/'cross_scene_summary.csv');r=pd.read_csv(O/'primary_by_re.csv');d=pd.read_csv(O/'decision_regret.csv');fig,ax=plt.subplots(2,2,figsize=(12.5,8))
q=s[(s.quota==.5)&(s.split.isin(['development','late','future','test']))];order=['development','late','future','test'];x=np.arange(4)
for j,seed in enumerate([42,123,2025]):
 a=q[q.seed==seed].set_index('split').loc[order];ax[0,0].plot(x,a.gain_percent,'o-',label=f'seed {seed}')
ax[0,0].axhline(0,color='gray',lw=.8);ax[0,0].set_xticks(x);ax[0,0].set_xticklabels(['dev','late','future','held-out Re']);ax[0,0].set_ylabel('MSE gain vs equal-budget random (%)');ax[0,0].set_title('A. In-domain routing (fixed backbone)');ax[0,0].legend(fontsize=8)
for j,seed in enumerate([42,123,2025]):
 a=c[c.seed==seed].set_index('scene').loc[['cylinder','fsi']];ax[0,1].plot([0,1],a.benefit_vs_random_pct,'o-',label=f'seed {seed}')
ax[0,1].axhline(0,color='gray',lw=.8);ax[0,1].set_xticks([0,1]);ax[0,1].set_xticklabels(['cylinder','FSI']);ax[0,1].set_ylabel('MSE gain vs equal-budget random (%)');ax[0,1].set_title('B. Cross-scene zero-shot stress test')
a=r[r.split.isin(['development','test'])];labels=[str(int(x)) for x in a.group];colors=['#b35348' if x<0 else '#176c78' for x in a.gain_percent];ax[1,0].bar(labels,a.gain_percent,color=colors);ax[1,0].axhline(0,color='gray',lw=.8);ax[1,0].set_ylabel('MSE gain (%)');ax[1,0].set_title('C. Three-seed average by Re (dev | test)')
t=pd.read_csv(O/'timing.csv').set_index('method');names=['base','warp','benefit50','rescrct_original'];vals=t.loc[names,'per_request_ms'];ax[1,1].bar(range(4),vals,color=['#999999','#176c78','#b35348','#666666']);ax[1,1].set_xticks(range(4));ax[1,1].set_xticklabels(['base','all warp','route 50%','ResCrct'],rotation=15);ax[1,1].set_ylabel('Median service ms/request');ax[1,1].set_title('D. End-to-end service timing (12 requests)')
for a in ax.flat:a.grid(axis='y',alpha=.18)
fig.suptitle('Recoverability: useful on held-out Re, not robust across scenarios',fontsize=14);fig.text(.5,.015,'Exploratory reused datasets; three heads share one backbone. Panels A–C report different samples; no independent confirmation.',ha='center',fontsize=8.5);fig.tight_layout(rect=[0,.04,1,.96]);fig.savefig(O/'summary.png',dpi=170);fig.savefig(O/'summary.pdf');plt.close(fig)
