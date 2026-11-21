"""Nine synchronized, full32-member CRPS fits and development-only selection."""
from core import *


@torch.no_grad()
def development(model,norm,models,bundles,step):
    rows=[];std=norm['std'].reshape(2,1,1,1)
    for entry,b in bundles:
        for k in CFG['eval_k']:
            h,mask,spread,z=prepare(b,k,norm)
            outputs={key:(b['paths']['B' if k==0 else 'S'+str(k)].cuda() if step==0 else decode(sample(model,a,h,mask,spread,z),norm)) for key,a in models.items()}
            # All outputs issued before current future target is requested.
            truth=truth_for(entry,k)[:,:20-k] if k else truth_for(entry,0)
            if step==0:
                baseline=next(iter(outputs.values()))[:,:,:20-k] if k else next(iter(outputs.values()))
                scores,_=native.quality(baseline,truth,std)
            for (arm,seed),p in outputs.items():
                if step: scores,_=native.quality(p[:,:,:20-k] if k else p,truth,std)
                rows.extend(dict(split='development',trajectory=entry['trajectory'],Re=entry['Re'],start=entry['start'],repeat=entry['repeat'],k=k,view='tail' if k else 'direct',arm=arm,seed=seed,step=step,**r) for r in scores)
    return rows


def select(rows,weights):
    # All groups balanced, but explicitly reduce each original trajectory first.
    tr=native.aggregate([r for r in rows if r['channel']=='overall'],['arm','seed','step','trajectory','k'],QUAL)
    avg=native.aggregate(tr,['arm','seed','step','k'],QUAL)
    ix={(r['arm'],r['seed'],r['step'],r['k']):r for r in avg};selected=[];candidates=[]
    for seed in CFG['seeds']:
        for arm in CFG['arms']:
            available=sorted(set(r['step'] for r in avg if r['seed']==seed and r['arm']==arm))
            for step in available:
                objective=float(np.mean([ix[arm,seed,step,k]['crps'] for k in CFG['k']]))
                ratios={f'{metric}_k{k}':ix[arm,seed,step,k][metric]/ix[arm,seed,0,k][metric] for k in CFG['eval_k'] for metric in ['mse','smse']}
                candidates.append(dict(arm=arm,seed=seed,step=step,development_restart_crps=objective,eligible=all(v<=CFG['noninferiority_ratio'] for v in ratios.values()),**ratios))
            own=[r for r in candidates if r['arm']==arm and r['seed']==seed]
            sci=min(own,key=lambda r:(r['development_restart_crps'],r['step']))
            zero=next(r for r in own if r['step']==0)
            good=[r for r in own if r['eligible'] and r['development_restart_crps']<zero['development_restart_crps']]
            guarded=min(good,key=lambda r:(r['development_restart_crps'],r['step'])) if good else zero
            for policy,r in [('scientific',sci),('guarded',guarded)]:
                item=next(w for w in weights if (w['arm'],w['seed'],w['step'])==(arm,seed,r['step']))
                selected.append(dict(policy=policy,**r,checkpoint=item))
    csvwrite(OUT/'selection_candidates.csv',candidates)
    dump(OUT/'selection.json',selected)
    csvwrite(OUT/'selection.csv',[{k:v for k,v in r.items() if k!='checkpoint'} for r in selected])
    return selected


def main():
    configure();check();pre=json.loads((OUT/'PREFLIGHT.json').read_text());assert pre['passed']
    for p,h in pre['code'].items():assert digest(ROOT/p)==h,p
    assert not (OUT/'TRAINING_STARTED.json').exists(),'Never automatically overwrite or repeat a started fit'
    model,norm=load_model();basehash=state_hash(model)
    train_entries=entries('train');dev_entries=entries('development')
    assert len(train_entries)==184 and len(dev_entries)==48
    sourcehashes=json.loads((native.DELIVERY/'data_hashes.json').read_text())
    for split in ['train','development']:
        for record in CFG['records'][split]:assert digest(native.DATA/record['file'])==sourcehashes[record['file']]
    train_bundles=[(e,prior.load_bundle(e)) for e in train_entries]
    dev_bundles=[(e,prior.load_bundle(e)) for e in dev_entries]
    catalog=[dict(source_index=i,trajectory=e['trajectory'],start=e['start'],repeat=e['repeat'],k=k) for i,e in enumerate(train_entries) for k in CFG['train_k']]
    assert CFG['train_k']==[2,5,10] and all(r['k'] in [2,5,10] for r in catalog)
    indices=np.random.default_rng(CFG['schedule_seed']).integers(len(catalog),size=CFG['steps']).tolist()
    schedule=[dict(step=i+1,**catalog[n]) for i,n in enumerate(indices)]
    dump(OUT/'batch_schedule.json',schedule)
    dump(OUT/'data_sources.json',dict(train=train_entries,development=dev_entries,source_hdf5={r['file']:sourcehashes[r['file']] for split in ['train','development'] for r in CFG['records'][split]}))
    dump(OUT/'TRAINING_FROZEN.json',dict(code=code_hashes(),protocol_sha256=CFG['protocol_sha256'],config_sha256=digest(DELIVERY/'config.json'),preflight_sha256=digest(OUT/'PREFLIGHT.json'),schedule_sha256=digest(OUT/'batch_schedule.json'),data_sha256=digest(OUT/'data_sources.json'),utc=time.time()))
    models={(arm,seed):Adapter(arm,seed).cuda() for seed in CFG['seeds'] for arm in CFG['arms']}
    optim={key:torch.optim.Adam(a.parameters(),lr=CFG['lr']) for key,a in models.items()}
    for seed in CFG['seeds']:assert len({state_hash(models[arm,seed]) for arm in CFG['arms']})==1
    logs={key:(OUT/f'train_{key[0]}_{key[1]}.jsonl').open('x') for key in models}
    trainrows=Table(OUT/'training.csv');devrows=[];weights=[];cost=[]
    started=time.monotonic();torch.cuda.reset_peak_memory_stats()
    dump(OUT/'TRAINING_STARTED.json',dict(utc=time.time(),frozen_sha256=digest(OUT/'TRAINING_FROZEN.json')))
    dump(OUT/'environment.json',dict(torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),tf32=torch.backends.cuda.matmul.allow_tf32,normalization={k:v.tolist() for k,v in norm.items()}))
    def validate(step):
        t=time.monotonic()
        for (arm,seed),a in models.items():weights.append(checkpoint(a,seed,step))
        new=development(model,norm,models,dev_bundles,step);devrows.extend(new)
        csvwrite(OUT/'development.csv',devrows);dump(OUT/'checkpoints.json',weights)
        cost.append(dict(step=step,development_seconds=time.monotonic()-t));csvwrite(OUT/'development_cost.csv',cost)
        event('development_complete',step=step,rows=len(new),backbone_state_sha256=state_hash(model))
        assert state_hash(model)==basehash
        print('VALIDATED',step,'candidates',len(weights),'elapsed',round(time.monotonic()-started,2),flush=True)
    validate(0);last_boundary=time.monotonic();last_block=None;steps_done=0
    for step,row in enumerate(schedule,1):
        entry,b=train_bundles[row['source_index']];k=row['k'];assert k in [2,5,10]
        h,mask,spread,z=prepare(b,k,norm)
        y=(truth_for(entry,k)-norm['mean'][0])/norm['std'][0]
        for (arm,seed),a in models.items():
            t=time.monotonic();op=optim[arm,seed];op.zero_grad(set_to_none=True)
            prediction=sample(model,a,h,mask,spread,z)
            loss,first,pair=crps_parts(prediction,y);assert torch.isfinite(loss)
            loss.backward();gn=float(sum(p.grad.square().sum() for p in a.parameters()).sqrt());assert np.isfinite(gn)
            op.step();torch.cuda.synchronize()
            record=dict(arm=arm,seed=seed,step=step,trajectory=entry['trajectory'],start=entry['start'],repeat=entry['repeat'],k=k,crps=float(loss),member_truth=float(first),member_pair_half=float(pair),gradient_norm=gn,seconds=time.monotonic()-t)
            trainrows.write([record]);logs[arm,seed].write(json.dumps(record)+'\n')
        steps_done=step
        if step%50==0:
            trainrows.flush()
            for f in logs.values():f.flush()
            print('TRAIN',step,'/1000 x9','elapsed',round(time.monotonic()-started,2),flush=True)
        if step in CFG['candidates']:
            validate(step);elapsed=time.monotonic()-started;last_block=time.monotonic()-last_boundary;last_boundary=time.monotonic()
            if step<CFG['steps'] and elapsed+last_block*1.2>CFG['max_training_seconds']:
                dump(OUT/'TRAINING_RESOURCE_STOP.json',dict(common_steps=step,elapsed=elapsed,estimated_next_block=last_block*1.2));break
    trainrows.close()
    for f in logs.values():f.close()
    assert state_hash(model)==basehash and all(p.grad is None for p in model.parameters())
    selected=select(devrows,weights)
    check()
    dump(OUT/'TRAINING_COMPLETE.json',dict(runs=9,steps_each=steps_done,total_updates=steps_done*9,full_requested_budget_complete=steps_done==1000,seconds=time.monotonic()-started,peak_memory_allocated=torch.cuda.max_memory_allocated(),peak_memory_reserved=torch.cuda.max_memory_reserved(),backbone_state_sha256=basehash,checkpoints=len(weights),schedule_sha256=digest(OUT/'batch_schedule.json'),utc=time.time()))
    dump(OUT/'FROZEN_SELECTION.json',dict(selection_sha256=digest(OUT/'selection.json'),candidate_sha256=digest(OUT/'selection_candidates.csv'),checkpoints_sha256=digest(OUT/'checkpoints.json'),training_complete_sha256=digest(OUT/'TRAINING_COMPLETE.json'),config_sha256=digest(DELIVERY/'config.json'),checkpoints=[r['checkpoint'] for r in selected],utc=time.time()))
    print('TRAINING COMPLETE',steps_done*9,'updates; selections frozen before historical test/extension',flush=True)

if __name__=='__main__':main()
