"""Stratified descriptive scores and paired original-trajectory effects; never select on test."""
from collections import defaultdict
from experiment import *


def contrasts():
    result=[]
    for selection in ('scientific','deployment'):
        for kind in ARMS:
            for base in ('native','phase3'):result.append((kind+'_'+selection,base))
        result += [('T_local_'+selection,'D_'+selection),('T_local_'+selection,'T_diag_'+selection)]
    return result


def pairs(rows,keys,metrics):
    trajectory=aggregate(rows,[*keys,'trajectory','method'],metrics)
    groups=defaultdict(dict)
    for r in trajectory:groups[tuple(r[k] for k in keys)][r['trajectory'],r['method']]=r
    table=[];deltas=[]
    for key,index in groups.items():
        for left,right in contrasts():
            names=sorted(t for t,m in index if m==left)
            if not names or not all((t,right) in index for t in names):continue
            for metric in metrics:
                delta=[index[t,left][metric]-index[t,right][metric] for t in names]
                stats=bootstrap(delta)
                table.append(dict(zip(keys,key),contrast=left+' minus '+right,metric=metric,mean_delta=stats['mean_delta'],ci95_low=stats['ci95'][0],ci95_high=stats['ci95'][1],improved=stats['improved'],trajectories=stats['trajectories']))
                if keys==['split']:
                    deltas += [dict(split=key[0],trajectory=t,contrast=left+' minus '+right,metric=metric,delta=d) for t,d in zip(names,delta)]
    return table,deltas


def main():
    frozen=json.loads((OUT/'FROZEN.json').read_text())
    scores=read_csv(OUT/'scores.csv');detail=read_csv(OUT/'detail.csv')
    assert len(scores)==264*23 and len(detail)==len(scores)*40
    for keys,name in [(['split','method'],'summary'),(['split','method','seed'],'per_seed'),(['split','method','trajectory','Re'],'per_trajectory'),(['split','method','Re'],'per_Re'),(['split','method','start'],'per_window')]:
        csvwrite(OUT/(name+'.csv'),aggregate(scores,keys,METRICS))
    for keys,name in [(['split','method','channel'],'per_channel'),(['split','method','lead','channel'],'per_lead_channel'),(['split','method','Re','channel'],'per_Re_channel'),(['split','method','Re','lead','channel'],'per_Re_lead_channel')]:
        csvwrite(OUT/(name+'.csv'),aggregate(detail,keys,CELL_METRICS))
    p,d=pairs(scores,['split'],METRICS);csvwrite(OUT/'paired.csv',p);csvwrite(OUT/'trajectory_differences.csv',d)
    csvwrite(OUT/'paired_Re.csv',pairs(scores,['split','Re'],METRICS)[0])
    csvwrite(OUT/'paired_channel.csv',pairs(detail,['split','channel'],CELL_METRICS)[0])
    # Native is scored once; repeat its score (not its prediction) for each fixed-seed paired comparison.
    seeded=[]
    for r in scores:
        if r['method'] in ('native','zero'):
            seeded += [dict(r,seed=str(s)) for s in SEEDS]
        else:seeded.append(r)
    csvwrite(OUT/'paired_seed.csv',pairs(seeded,['split','seed'],METRICS)[0])
    selected=[];dev=read_csv(OUT/'development_scores.csv')
    for row in frozen['selection']:
        selected += [dict(r,method=row['kind']+'_'+row['selection']) for r in dev if r['kind']==row['kind'] and r['seed']==row['seed'] and r['step']==row['step']]
    native=[dict(r,method='native',seed='0') for r in dev if r['kind']=='D' and r['seed']=='42' and r['step']=='0']
    selected+=native
    csvwrite(OUT/'development_selected_scores.csv',selected)
    csvwrite(OUT/'development_summary.csv',aggregate(selected,['method'],METRICS))
    csvwrite(OUT/'development_per_seed.csv',aggregate(selected,['method','seed'],METRICS))
    csvwrite(OUT/'development_per_Re.csv',aggregate(selected,['method','Re'],METRICS))
    dd=read_csv(OUT/'development_detail.csv');chosen=[]
    for row in frozen['selection']:
        chosen += [dict(r,method=row['kind']+'_'+row['selection']) for r in dd if r['kind']==row['kind'] and r['seed']==row['seed'] and r['step']==row['step']]
    chosen += [dict(r,method='native',seed='0') for r in dd if r['kind']=='D' and r['seed']=='42' and r['step']=='0']
    csvwrite(OUT/'development_per_channel.csv',aggregate(chosen,['method','channel'],CELL_METRICS))
    csvwrite(OUT/'development_per_lead_channel.csv',aggregate(chosen,['method','lead','channel'],CELL_METRICS))
    csvwrite(OUT/'development_paired.csv',pairs(selected,['split'],METRICS)[0])
    events={}
    for split in ('test','extension'):
        rr=[r for r in scores if r['split']==split and r['method']=='native']
        positive=[r for r in rr if int(r['event'])]
        events[split]=dict(blocks=len(rr),trajectories=len(set(r['trajectory'] for r in rr)),positive_blocks=len(positive),positive_trajectories=sorted(set(r['trajectory'] for r in positive)),threshold=CFG['event_threshold'])
    dump(OUT/'events_summary.json',events)
    # All source scores are diagnostic implied q[h+1]-Aq[h], not trained force labels.
    src=[r for r in detail if r['r_mse']!='']
    csvwrite(OUT/'source_by_lead_channel.csv',aggregate(src,['split','method','lead','channel'],['r_mse','r_smse']))
    inv=read_csv(OUT/'invariance.csv')
    invmax={k:max(float(r[k]) for r in inv) for k in inv[0] if k not in ('split','start','method','seed','trajectory')}
    dump(OUT/'analysis_checks.json',dict(rows=len(scores),detail=len(detail),invariance_maximum=invmax,events=events))
    print('ANALYZED',len(scores),'block scores,',len(detail),'lead/channel scores; paired units=original trajectories',flush=True)

if __name__=='__main__':main()
