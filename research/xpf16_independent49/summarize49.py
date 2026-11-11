"""Summarize 49-way matched evaluations and write portable CSV/figures."""
import csv
import gzip
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from train49 import OUT, configs

METHODS=('XPF16','8-noise','32-noise','256-noise')
MODES=('raw','online')
def save_figure(fig, path):
    fig.savefig(path, dpi=220)
    fig.savefig(path.with_suffix('.pdf'))


COLS=('lower_coverage','upper_coverage','coverage','width','interval_score',
      'lower_pinball','upper_pinball','lower_miss','upper_miss')


def main():
    assert (OUT/'evaluation_complete.json').exists()
    folder=OUT/'summary'
    folder.mkdir(parents=True,exist_ok=True)
    allgroup=[]
    for split in ('development','test'):
        directory=OUT/'evaluation'/split
        metadata=json.loads((directory/'run.json').read_text())
        records=metadata['records']
        # k, method, mode, channel, frame, metric: separately retain Re groups.
        grouped={}
        dist=[]
        outfile=folder/f'{split}_per_window.csv.gz'
        with gzip.open(outfile,'wt',newline='') as f:
            writer=csv.writer(f)
            writer.writerow(['split','start','file','group_id','Re','aoa','channel','method','mode','experiment','target_coverage',*COLS])
            for start in metadata['starts']:
                data=np.load(directory/f'{start:04d}.npz')
                values=data['metrics'].mean(axis=-2) # mode,method,k,batch,channel,metrics
                for bi,record in enumerate(records):
                    group=str(record['group'])
                    if start>0:
                        if group not in grouped:
                            grouped[group]=[]
                        grouped[group].append(data['metrics'][:,:,:,bi])
                    for mode in range(2):
                        for method in range(4):
                            for k in range(49):
                                for c,ch in enumerate(('u','v')):
                                    writer.writerow([split,start,record['file'],group,record['re'],record['aoa'],ch,METHODS[method],MODES[mode],
                                                     f'XPF16-Q{k+1:02d}',1-2*(k+1)/100,*values[mode,method,k,bi,c]])
                if start>0:
                    row={'start':start}
                    for key in data.files:
                        if key!='metrics':
                            row[key]=float(data[key].mean()) if not key.endswith('pit_histogram') else data[key].tolist()
                    dist.append(row)
        rows=[]
        for group,items in grouped.items():
            v=np.stack(items).mean(0) # mode,method,k,channel,frame,metric
            for mode in range(2):
                for method in range(4):
                    for k,cfg in enumerate(configs()):
                        for channel,ch in enumerate(('u','v')):
                            for lead in range(20):
                                rows.append(dict(split=split,group=group,channel=ch,mode=MODES[mode],method=METHODS[method],
                                                 experiment=cfg['id'],alpha=cfg['lower'],target_coverage=cfg['target_coverage'],lead=lead+1,
                                                 **{name:float(v[mode,method,k,channel,lead,i]) for i,name in enumerate(COLS)}))
        with (folder/f'{split}_group_lead.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
        allgroup.extend(rows)
        (folder/f'{split}_distribution_windows.json').write_text(json.dumps(dist,indent=2)+'\n')
    # Equal weight Re groups and channels, then leads (not pixel-count weighting).
    groups={}
    for row in allgroup:
        key=(row['split'],row['mode'],row['method'],row['experiment'])
        groups.setdefault(key,[]).append(row)
    macro=[]
    for key,items in sorted(groups.items()):
        first=items[0]
        row={k:first[k] for k in ('split','mode','method','experiment','alpha','target_coverage')}
        row.update({col:float(np.mean([r[col] for r in items])) for col in COLS})
        row['coverage_error_pp']=100*(row['coverage']-row['target_coverage'])
        row['lower_error_pp']=100*(row['lower_coverage']-row['alpha'])
        row['upper_error_pp']=100*(row['upper_coverage']-(1-row['alpha']))
        macro.append(row)
    with (folder/'macro49.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=macro[0].keys());writer.writeheader();writer.writerows(macro)
    plot(folder,macro,allgroup)
    plot_extra(folder,allgroup)
    print('SUMMARY_COMPLETE',len(macro),flush=True)


def plot(folder,macro,group):
    figdir=folder/'figures';figdir.mkdir(exist_ok=True)
    test=[r for r in macro if r['split']=='test']
    def rows(method,mode):
        return sorted([r for r in test if r['method']==method and r['mode']==mode],key=lambda r:r['target_coverage'])
    fig,ax=plt.subplots(figsize=(7,5))
    for mode in MODES:
        r=rows('XPF16',mode);ax.plot([x['target_coverage'] for x in r],[x['coverage'] for x in r],label=mode)
    ax.plot([0,1],[0,1],'k--',label='ideal');ax.set(xlabel='Target coverage',ylabel='Empirical coverage');ax.legend();fig.tight_layout();save_figure(fig,figdir/'coverage_vs_target.png');plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,5))
    for mode in MODES:
        r=rows('XPF16',mode)
        x=[v['alpha'] for v in r]+[1-v['alpha'] for v in r]
        y=[v['lower_coverage'] for v in r]+[v['upper_coverage'] for v in r]
        order=np.argsort(x);ax.plot(np.array(x)[order],np.array(y)[order],label=mode)
    ax.plot([0,1],[0,1],'k--');ax.set(xlabel='Target quantile',ylabel='Empirical CDF');ax.legend();fig.tight_layout();save_figure(fig,figdir/'quantile_calibration.png');plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,5))
    for method in METHODS:
        for mode in MODES:
            r=rows(method,mode);ax.plot([v['target_coverage'] for v in r],[v['width'] for v in r],label=f'{method} {mode}',linestyle='-' if mode=='online' else '--')
    ax.set(xlabel='Target coverage',ylabel='Standardized width');ax.legend(fontsize=8);fig.tight_layout();save_figure(fig,figdir/'width_vs_target.png');plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,5))
    for mode in MODES:
        r=rows('XPF16',mode);x=[v['alpha'] for v in r]+[1-v['alpha'] for v in r];y=[v['lower_pinball'] for v in r]+[v['upper_pinball'] for v in r];ix=np.argsort(x);ax.plot(np.array(x)[ix],np.array(y)[ix],label=mode)
    ax.set(xlabel='Quantile',ylabel='Standardized pinball');ax.legend();fig.tight_layout();save_figure(fig,figdir/'pinball.png');plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,8))
    v=np.zeros((49,20))
    for k in range(1,50):
        for lead in range(1,21):
            r=[x for x in group if x['split']=='test' and x['mode']=='online' and x['method']=='XPF16' and x['experiment']==f'XPF16-Q{k:02d}' and x['lead']==lead]
            v[k-1,lead-1]=100*np.mean([x['coverage']-x['target_coverage'] for x in r])
    im=ax.imshow(v,aspect='auto',origin='lower',cmap='coolwarm');fig.colorbar(im,ax=ax,label='Coverage error (pp)');ax.set(xlabel='Future lead (index 0=frame 1)',ylabel='Q index (0=Q01)');fig.tight_layout();save_figure(fig,figdir/'coverage_lead_heatmap.png');plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,5))
    for method in METHODS:
        for mode in MODES:
            r=rows(method,mode);ax.plot([v['target_coverage'] for v in r],[v['coverage'] for v in r],label=f'{method} {mode}',linestyle='-' if mode=='online' else '--')
    ax.plot([0,1],[0,1],'k:');ax.set(xlabel='Target coverage',ylabel='Empirical coverage');ax.legend(fontsize=8);fig.tight_layout();save_figure(fig,figdir/'matched_coverage.png');plt.close(fig)

def plot_extra(folder,group):
    figdir=folder/'figures'
    # Per-Re and per-velocity marginal calibration using the same matched windows.
    rows=[r for r in group if r['split']=='test' and r['mode']=='online' and r['method']=='XPF16']
    fig,axes=plt.subplots(1,2,figsize=(11,4.8),sharey=True)
    for channel,ax in zip(('u','v'),axes):
        for regime in sorted({r['group'] for r in rows}):
            xs=[];ys=[]
            for k in range(1,50):
                values=[r for r in rows if r['group']==regime and r['channel']==channel and r['experiment']==f'XPF16-Q{k:02d}']
                xs.extend([k/100,1-k/100])
                ys.extend([np.mean([v['lower_coverage'] for v in values]),np.mean([v['upper_coverage'] for v in values])])
            ix=np.argsort(xs);ax.plot(np.array(xs)[ix],np.array(ys)[ix],label=regime)
        ax.plot([0,1],[0,1],'k:');ax.set(title=channel,xlabel='Target quantile');ax.legend(fontsize=7)
    axes[0].set_ylabel('Empirical CDF');fig.tight_layout();save_figure(fig,figdir/'Re_channel_quantile_calibration.png');plt.close(fig)
    windows=json.loads((folder/'test_distribution_windows.json').read_text())
    methods=['raw_unrearranged','raw_rearranged','online_unrearranged','online_rearranged']
    fig,axes=plt.subplots(1,3,figsize=(12,4))
    for j,metric in enumerate(('crps','wis','crossing')):
        axes[j].bar(range(4),[np.mean([w[f'{m}_{metric}'] for w in windows]) for m in methods])
        axes[j].set_xticks(range(4),['raw','raw sort','online','online sort'],rotation=35,ha='right');axes[j].set_title(metric)
    fig.tight_layout();save_figure(fig,figdir/'crps_wis_crossing.png');plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4.5))
    for mode in ('raw','online'):
        hist=np.sum([np.asarray(w[f'{mode}_rearranged_pit_histogram']) for w in windows],axis=0)
        hist=hist/hist.sum()
        ax.plot((np.arange(20)+.5)/20,hist,marker='.',label=mode)
    ax.axhline(.05,color='k',linestyle=':');ax.set(xlabel='Mid-PIT bin',ylabel='Frequency');ax.legend()
    fig.tight_layout();save_figure(fig,figdir/'pit_histogram.png');plt.close(fig)
    snap=np.load(OUT/'evaluation/test/snapshot.npz')
    # A grid point is a forecast quantile function, while the one observed
    # outcome is a vertical indicator; it is not a per-point empirical CDF.
    fig,ax=plt.subplots(figsize=(7,5))
    tau=np.r_[np.arange(1,50),np.arange(51,100)]/100
    for name in ('raw','online'):
        q=snap[name][:,0,0,0];ix=np.argsort(q)
        ax.plot(q[ix],tau,label=f'{name}: rearranged quantile CDF')
    pooled=np.sort(snap['truth'][:,0,:].reshape(-1));ax.step(pooled,np.arange(1,len(pooled)+1)/len(pooled),where='post',alpha=.6,label='pooled snapshot residual ECDF (descriptive)')
    truth=float(snap['truth'][0,0,0]);ax.axvline(truth,color='k',linestyle='--',label='observed value')
    ax.set(xlabel='Normalized residual',ylabel='Probability');ax.legend();fig.tight_layout();save_figure(fig,figdir/'example_quantile_cdf.png');plt.close(fig)

if __name__=='__main__': main()
