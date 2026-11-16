"""Join aligned full-test online CDFs; preserve each method's calibration."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent.parent
DEST=ROOT/'github/DiPred/results/reliability/fair_f_p_pf_xf_sharedref'
FAIR=ROOT/'github/DiPred/results/reliability/fair_x_xp_xpf'
NOISE=ROOT/'github/DiPred/results/reliability/xpf16_02grid_2048/full_test_cdf'

def read(path):
    with path.open(newline='') as f:
        rows=list(csv.DictReader(f))
    return {k:np.array([float(v[k]) for v in rows]) for k in rows[0]}

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()

def main():
    current=json.loads((DEST/'config.json').read_text())
    old=json.loads((FAIR/'config.json').read_text())
    noise_config=json.loads((NOISE/'config.json').read_text())
    ids=lambda rows:[r['file'] for r in rows]
    assert ids(current['trajectory_split']['test'])==ids(old['splits']['test'])==ids(noise_config['trajectory_records'])
    assert current['test_starts']==old['evaluation_starts']==noise_config['starts']
    fig,axes=plt.subplots(1,2,figsize=(15,5),sharex=True,sharey=True)
    for channel,ax in zip(('u','v'),axes):
        new=read(DEST/f'cdf_{channel}.csv');old=read(FAIR/f'cdf_{channel}.csv');noise=read(NOISE/f'cdf_{channel}.csv')
        x=new['x'];assert np.allclose(x,old['x'],atol=1e-12) and np.allclose(x,noise['x'],atol=1e-12)
        curves={}
        for arm in ('F','P','PF','XF'):
            curves[arm]=np.mean([new[f'{arm}_{seed}_online'] for seed in (42,123,2026)],axis=0)
        for arm in ('X','XP','XPF'):
            curves[arm]=np.mean([old[f'{arm}_{seed}_online'] for seed in (42,123,2026)],axis=0)
        for n in (8,32,256,2048):curves[f'{n}-noise']=noise[f'noise{n}']
        with (DEST/f'cdf_{channel}_combined_online.csv').open('w',newline='') as f:
            w=csv.writer(f);w.writerow(['x']+list(curves))
            for i,v in enumerate(x):w.writerow([v]+[a[i] for a in curves.values()])
        for label,v in curves.items():ax.plot(x,v,label=label,linewidth=1.2)
        ax.set(title=f'{channel} channel',xlabel='Forecast value − truth (original data scale)',ylabel='Mean predictive CDF',xlim=(-.35,.35),ylim=(0,1));ax.grid(alpha=.2)
    axes[-1].legend(ncol=2,fontsize=8);fig.tight_layout()
    fig.savefig(DEST/'cdf_combined_online.png',dpi=220);fig.savefig(DEST/'cdf_combined_online.pdf');plt.close(fig)
    with (DEST/'README.md').open('a') as f:f.write('\nCombined online curves use the same test trajectories, target-centred grid and original numerical scale. Heads use per-tail online offsets; noise uses its previously published 90% proxy-scale calibration, so calibration procedures differ. All seven head methods use the last X frame as reference, and P/F themselves depend on X.\n')
    p=DEST/'manifest.json';m=json.loads(p.read_text());m['files_sha256']={str(v.relative_to(DEST)):sha(v) for v in DEST.rglob('*') if v.is_file() and v!=p};p.write_text(json.dumps(m,indent=2)+'\n')
    print('CDF_PLOT_COMPLETE',flush=True)
if __name__=='__main__':main()
