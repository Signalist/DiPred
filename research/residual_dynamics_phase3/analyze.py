"""Collapse windows/seeds within original IDs before resampling; verify saved forecasts."""
from collections import defaultdict
import csv
from run import out,ref,read,dump,digest,csvwrite,bootstrap,records,np


def readrows(name):
    with (out/(name+'.csv')).open() as f:return list(csv.DictReader(f))


def group(rows,keys,metrics):
    acc=defaultdict(list)
    for r in rows:acc[tuple(r[k] for k in keys)].append([float(r[k]) for k in metrics])
    return [dict(zip(keys,k),**dict(zip(metrics,np.mean(v,0)))) for k,v in acc.items()]


def main():
    scores=readrows('scores'); detail=readrows('lead_channel'); sel=readrows('selection')
    assert len(detail)==40*len(scores)
    for r in sel:assert digest(out/'weights'/f"{r['kind']}_{r['seed']}.pt")==r['sha256']
    metrics=['mse','crps','energy','brier']
    index=['split','trajectory','start','method','seed']
    sums={tuple(r[k] for k in index):r for r in group(detail,index,['mse','crps'])}
    for r in scores:
        for k in ('mse','crps'):
            assert abs(sums[tuple(r[x] for x in index)][k]-float(r[k]))<1e-12
    for keys,filename in [(['split','method'],'summary'),(['split','method','seed'],'per_seed'),
                          (['split','method','Re'],'per_Re'),(['split','method','trajectory'],'per_trajectory')]:
        csvwrite(out/(filename+'.csv'),group(scores,keys,metrics))
    for r in detail:r['period']='first' if r['lead']=='1' else 'rest'
    periods=group(detail,['split','method','period'],['mse','crps'])
    leads=group(detail,['split','method','lead'],['mse','crps'])
    csvwrite(out/'periods.csv',periods);csvwrite(out/'leads.csv',leads)
    contrasts=[]
    for rows,ks,pairs in [(scores,metrics,[('crps','raw'),('crps','std'),('crps','native'),('raw_endpoint','crps_endpoint')]),
                           ([r for r in detail if r['lead']!='1'],['mse','crps'],[('linear_first','native'),('linear_first','copy_first'),('linear_teacher','linear_first')])]:
        for split in ('development','test'):
            tr=group([r for r in rows if r['split']==split],['trajectory','method'],ks)
            idx={(r['trajectory'],r['method']):r for r in tr}
            for a,b in pairs:
                if split=='development' and a.endswith('_endpoint'):continue
                for k in ks:
                    delta=[idx[r['file'],a][k]-idx[r['file'],b][k] for r in records[split]]
                    contrasts.append(dict(split=split,metric=k,contrast=a+' minus '+b,**bootstrap(delta)))
    dump(out/'paired.json',contrasts)
    # Independent references: zero recursion exactly native; teacher copies phase2 bias metric.
    for r in periods:
        if r['method']=='copy_teacher' and r['period']=='rest':
            with (ref/'component_prediction_summary.csv').open() as f:
                old=next(v for v in csv.DictReader(f) if v['split']==r['split'] and v['component']=='bias' and v['method']=='A1_persistence')
            assert abs(r['mse']-float(old['mse']))<1e-10
    base={(r['split'],r['start'],r['trajectory']):r for r in scores if r['method']=='native'}
    for r in scores:
        if r['method'].endswith('_zero') or r['method'] in ('raw','std') and any(
            v['kind']==r['method'] and v['seed']==r['seed'] and v['step']=='0' for v in sel):
            for k in metrics:assert r[k]==base[r['split'],r['start'],r['trajectory']][k]
    dump(out/'data_split.json',records)
    print('PASS cache-linked scores, zero recursion, checkpoint hashes, phase2 teacher reference, 40 lead/channel cells per block')
    for s in ['development','test']:
        positive=[r for r in base.values() if r['split']==s and int(r['event'])]
        print('event',s,len(positive),'blocks',len({r['trajectory'] for r in positive}),'IDs')


if __name__=='__main__':main()
