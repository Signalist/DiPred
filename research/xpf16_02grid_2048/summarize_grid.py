"""Aggregate all 499 level results and retain compact per-window arrays."""
import csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from train_grid import OUT
from evaluate_grid import INTERVAL_ARMS,QUANTILE_ARMS,METRICS

DIST_KEYS=('xpf_raw_crps','xpf_raw_wis','xpf_online_crps','xpf_online_wis',
           'xpf_online_rearranged_crps','xpf_online_rearranged_wis',
           'noise_two_tail_rearranged_crps','noise_two_tail_rearranged_wis',
           'noise_raw_quantiles_crps','noise_raw_quantiles_wis','noise_proxy_quantiles_crps','noise_proxy_quantiles_wis',
           'noise_raw_empirical','noise_raw_fair','noise_proxy_empirical','noise_proxy_fair',
           'xpf_crossing_fraction','xpf_any_crossing','noise_crossing_fraction','noise_any_crossing')

PIT_KEYS=('xpf_online_pit_histogram','noise_raw_pit_histogram','noise_proxy_pit_histogram','noise_two_tail_pit_histogram')


def main():
    assert (OUT/'evaluation_complete.json').exists()
    out=OUT/'summary';out.mkdir(exist_ok=True)
    allq=[];alli=[];distributions={};pit_summaries={}
    for split in ('development','test'):
        folder=OUT/'evaluation'/split
        meta=json.loads((folder/'complete.json').read_text());records=meta['records']
        groups=sorted({str(r['group']) for r in records})
        isums={g:[] for g in groups};qsums={g:[] for g in groups};dsums={g:[] for g in groups}
        w_intervals=[];w_marginals=[];w_dist={k:[] for k in DIST_KEYS};pit={k:[] for k in PIT_KEYS}
        for start in meta['starts']:
            d=np.load(folder/f'{start:04d}.npz')
            iv=d['intervals'];q=d['marginals']
            w_intervals.append(iv.mean(axis=-2));w_marginals.append(q.mean(axis=-2))
            for k in DIST_KEYS:w_dist[k].append(d[k].mean(axis=-1))
            if start>0:
                for k in PIT_KEYS:pit[k].append(d[k])
                for g in groups:
                    ix=[i for i,r in enumerate(records) if str(r['group'])==g]
                    isums[g].append(iv[:,:,ix].mean(axis=2))
                    qsums[g].append(q[:,:,ix].mean(axis=2))
                    dsums[g].append(np.stack([d[k][ix].mean(axis=0) for k in DIST_KEYS]))
            print('SUMMARIZE_GRID',split,start,flush=True)
        # Arrays retain per-trajectory u/v scores per window; frame grouping is separate.
        np.savez_compressed(out/f'{split}_per_window.npz',
            starts=np.array(meta['starts']),intervals=np.stack(w_intervals),marginals=np.stack(w_marginals),
            **{k:np.stack(v) for k,v in w_dist.items()})
        i_group=np.stack([np.stack(isums[g]).mean(axis=0) for g in groups])
        q_group=np.stack([np.stack(qsums[g]).mean(axis=0) for g in groups])
        d_group=np.stack([np.stack(dsums[g]).mean(axis=0) for g in groups])
        np.savez_compressed(out/f'{split}_group_lead.npz',groups=np.array(groups),intervals=i_group,marginals=q_group,
            **{key:d_group[:,j] for j,key in enumerate(DIST_KEYS)})
        # Equal weight for Re group, u/v channel and lead.
        iv=i_group.mean(axis=(0,3,4));qv=q_group.mean(axis=(0,3,4));dv=d_group.mean(axis=(0,2,3))
        # iv [5,249,9]; qv [7,499,2]; dv [16]
        for a,arm in enumerate(INTERVAL_ARMS):
            for k in range(249):
                alpha=(k+1)/500
                r=dict(split=split,method=arm,alpha=alpha,lower_pct=100*alpha,upper_pct=100*(1-alpha),target_coverage=1-2*alpha)
                r.update({name:float(iv[a,k,j]) for j,name in enumerate(METRICS)})
                r['coverage_error_pp']=100*(r['coverage']-r['target_coverage'])
                r['lower_error_pp']=100*(r['lower_coverage']-alpha)
                r['upper_error_pp']=100*(r['upper_coverage']-(1-alpha))
                alli.append(r)
        for a,arm in enumerate(QUANTILE_ARMS):
            for k in range(499):
                level=(k+1)/500
                allq.append(dict(split=split,method=arm,level=level,percent=100*level,coverage=float(qv[a,k,0]),
                                 error_pp=100*(float(qv[a,k,0])-level),pinball=float(qv[a,k,1])))
        distributions[split]={key:float(v) for key,v in zip(DIST_KEYS,dv)}
        pit_summaries[split]={k:(np.sum(v,axis=0)/np.sum(v)).tolist() for k,v in pit.items()}
    for filename,rows in (('quantiles499.csv',allq),('intervals249.csv',alli)):
        with (out/filename).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    (out/'distribution_summary.json').write_text(json.dumps(distributions,indent=2)+'\n')
    (out/'pit_summary.json').write_text(json.dumps(pit_summaries,indent=2)+'\n')
    plot(out,allq,alli)
    plot_diagnostics(out,distributions['test'])
    fig,ax=plt.subplots(figsize=(7,4.5))
    for key,value in pit_summaries['test'].items():ax.plot((np.arange(20)+.5)/20,value,marker='.',label=key)
    ax.axhline(.05,color='black',linestyle=':');ax.set(xlabel='Mid-PIT',ylabel='Frequency');ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(out/'pit.png',dpi=220);fig.savefig(out/'pit.pdf');plt.close(fig)
    print('GRID_SUMMARY_COMPLETE',flush=True)


def plot(out,allq,alli):
    q=[r for r in allq if r['split']=='test'];intervals=[r for r in alli if r['split']=='test']
    fig,axes=plt.subplots(1,3,figsize=(15,4.5))
    for arm in ('xpf_raw','xpf_online','xpf_online_rearranged','noise_raw','noise_proxy','noise_online_two_tail_rearranged'):
        r=[v for v in q if v['method']==arm]
        axes[0].plot([v['percent'] for v in r],[100*v['coverage'] for v in r],label=arm)
        axes[1].plot([v['percent'] for v in r],[v['error_pp'] for v in r],label=arm)
        axes[2].plot([v['percent'] for v in r],[v['pinball'] for v in r],label=arm)
    axes[0].plot([0,100],[0,100],'k:');axes[0].set(xlabel='Target quantile %',ylabel='Empirical CDF %')
    axes[1].axhline(0,color='k',linestyle=':');axes[1].set(xlabel='Target quantile %',ylabel='CDF error (pp)')
    axes[2].set(xlabel='Target quantile %',ylabel='Standardized pinball')
    axes[0].legend(fontsize=7);fig.tight_layout();fig.savefig(out/'quantile499.png',dpi=220);fig.savefig(out/'quantile499.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,4.5))
    for arm in INTERVAL_ARMS:
        r=sorted([v for v in intervals if v['method']==arm],key=lambda v:v['target_coverage'])
        axes[0].plot([100*v['target_coverage'] for v in r],[100*v['coverage'] for v in r],label=arm)
        axes[1].plot([100*v['target_coverage'] for v in r],[v['width'] for v in r],label=arm)
        axes[2].plot([100*v['target_coverage'] for v in r],[v['interval_score'] for v in r],label=arm)
    axes[0].plot([0,100],[0,100],'k:')
    for ax,label in zip(axes,('Actual coverage %','Standardized width','Interval score')):ax.set(xlabel='Target interval coverage %',ylabel=label)
    axes[0].legend(fontsize=7);fig.tight_layout();fig.savefig(out/'interval249.png',dpi=220);fig.savefig(out/'interval249.pdf');plt.close(fig)


def plot_diagnostics(out,distribution):
    group=np.load(out/'test_group_lead.npz')
    iv=group['intervals'];q=group['marginals'];groups=group['groups']
    target=1-2*np.arange(1,250)/500
    coverage=iv[:,1,:,:,:,2].mean(axis=(0,2))
    fig,ax=plt.subplots(figsize=(9,7))
    im=ax.imshow(100*(coverage-target[:,None]),origin='lower',aspect='auto',cmap='coolwarm')
    fig.colorbar(im,ax=ax,label='Coverage error (pp)')
    ax.set(xlabel='Forecast frame (0 = frame 1)',ylabel='Tail level index (0 = 0.2%)')
    fig.tight_layout();fig.savefig(out/'lead_heatmap.png',dpi=220);fig.savefig(out/'lead_heatmap.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),sharey=True)
    levels=np.arange(1,500)/500
    for c,ax in enumerate(axes):
        for g,label in enumerate(groups):
            coverage=q[g,1,:,c,:,0].mean(axis=1)
            ax.plot(levels,coverage,label=str(label))
        ax.plot([0,1],[0,1],'k:');ax.set(xlabel='Target quantile',title=('u','v')[c]);ax.legend(fontsize=7)
    axes[0].set_ylabel('Empirical CDF');fig.tight_layout();fig.savefig(out/'re_channel.png',dpi=220);fig.savefig(out/'re_channel.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,4.5))
    variants=('xpf_raw','xpf_online','xpf_online_rearranged')
    for ax,key in zip(axes,('crps','wis')):
        ax.bar(range(3),[distribution[v+'_'+key] for v in variants]);ax.set_xticks(range(3),variants,rotation=25,ha='right');ax.set_title(key)
    axes[2].bar([0,1],[distribution['xpf_crossing_fraction'],distribution['xpf_any_crossing']])
    axes[2].set_xticks([0,1],['adjacent pairs','any crossing'],rotation=25,ha='right');axes[2].set_title('Online crossing')
    fig.tight_layout();fig.savefig(out/'distribution_diagnostics.png',dpi=220);fig.savefig(out/'distribution_diagnostics.pdf');plt.close(fig)
    snap=np.load(OUT/'evaluation/test/0000.npz')
    # Direct empirical CDF of 2048 actual forecasts at the same selected point.
    samples=snap['sample_snapshot'][:,0,0,0]-snap['point_snapshot'][0,0,0]
    samples=np.sort(samples);value=float(snap['truth_snapshot'][0,0,0])
    xpf=snap['xpf_rearranged_snapshot'][:,0,0,0]
    fig,ax=plt.subplots(figsize=(7,5))
    ax.plot(xpf,levels,label='XPF online, monotone rearranged')
    ax.plot(samples,np.arange(1,len(samples)+1)/len(samples),label='2048 raw sample empirical CDF')
    ax.axvline(value,color='black',linestyle='--',label='observed residual')
    ax.set(xlabel='Normalized residual at one held-out point',ylabel='Probability');ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(out/'point_cdf.png',dpi=220);fig.savefig(out/'point_cdf.pdf');plt.close(fig)

if __name__=='__main__':main()
