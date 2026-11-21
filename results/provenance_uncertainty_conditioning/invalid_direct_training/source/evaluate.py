"""Frozen selected candidates; all histories/outputs issued before scoring truth."""
from core import *
from research.provenance_uncertainty_conditioning.metrics import scores,structure,EXTRA
import shutil


def extension_plan():
    return [dict(split='extension',trajectory=r['file'],Re=r['group'],start=start,repeat=repeat) for start in CFG['extension'] for repeat in range(2) for r in CFG['records']['test']]


@torch.no_grad()
def create_extension(model,norm,entry):
    h=prior.frames(entry['trajectory'],entry['start'],entry['start']+20).cuda()
    original=h[None].expand(32,-1,-1,-1,-1).contiguous();paths={};calls={}
    for family in ['A','B','C']:
        seeds=native.seeds_for('test',entry['trajectory'],entry['start'],entry['repeat'],family,0)
        paths[family],calls[family]=native.generate(model,norm,original,seeds)
    for k in CFG['k']:
        history=native.restart_history(h,paths['A'],k)
        seeds=native.seeds_for('test',entry['trajectory'],entry['start'],entry['repeat'],'S',k)
        paths['S'+str(k)],calls['S'+str(k)]=native.generate(model,norm,history,seeds)
    ident=dict(**entry,seed_split='test',member_ids=list(range(32)),parent_child_ids=[[i,i] for i in range(32)],physical={str(k):native.physical_indices(entry['start'],k) for k in CFG['train_k']})
    b=dict(identity=ident,history=h.cpu(),paths={key:v.cpu() for key,v in paths.items()},calls=calls,history_sha256=tensor_hash(h))
    p=CACHE/'native_extension'/f"{entry['trajectory']}_{entry['start']}_{entry['repeat']}.pt"
    assert not p.exists();p.parent.mkdir(parents=True,exist_ok=True);torch.save(b,p)
    e=dict(**entry,path=str(p),sha256=digest(p),bytes=p.stat().st_size,history_sha256=tensor_hash(h),outputs={key:call['output_sha256'] for key,call in calls.items()},calls=calls)
    return b,e


def frozen_inputs():
    selection=json.loads((OUT/'selection.json').read_text());frozen=json.loads((OUT/'FROZEN_SELECTION.json').read_text())
    assert digest(OUT/'selection.json')==frozen['selection_sha256']
    assert digest(OUT/'checkpoints.json')==frozen['checkpoints_sha256']
    items={}
    for r in selection:
        item=r['checkpoint'];key=f"{item['arm']}_{item['seed']}_{item['step']}"
        if item['step']:items[key]=item
    return selection,items


def save_append_manifest(f,entry):f.write(json.dumps(entry)+'\n');f.flush()


@torch.no_grad()
def main():
    configure();check();selection,items=frozen_inputs()
    assert json.loads((OUT/'EVALUATION_TESTS.json').read_text())['passed']
    assert not (OUT/'EVALUATION_STARTED.json').exists(),'Do not overwrite/repeat an evaluation'
    model,norm=load_model();basehash=state_hash(model);std=norm['std'].reshape(2,1,1,1)
    adapters={key:read_checkpoint(item) for key,item in items.items()}
    sourcehashes=json.loads((native.DELIVERY/'data_hashes.json').read_text())
    for record in CFG['records']['test']:assert digest(native.DATA/record['file'])==sourcehashes[record['file']]
    CACHE.mkdir(parents=True,exist_ok=True);assert shutil.disk_usage(CACHE).free>CFG['max_cache_bytes']
    plan=[{key:e[key] for key in META if key!='k'} for e in entries('test')]+extension_plan()
    dump(OUT/'evaluation_plan.json',plan)
    dump(OUT/'EVALUATION_FROZEN.json',dict(code=code_hashes(),selection_frozen_sha256=digest(OUT/'FROZEN_SELECTION.json'),evaluation_tests_sha256=digest(OUT/'EVALUATION_TESTS.json'),plan_sha256=digest(OUT/'evaluation_plan.json'),source_hashes={r['file']:sourcehashes[r['file']] for r in CFG['records']['test']},utc=time.time()))
    dump(OUT/'EVALUATION_STARTED.json',dict(utc=time.time(),frozen_sha256=digest(OUT/'EVALUATION_FROZEN.json'),unique_nonzero_checkpoints=len(adapters)))
    score_table=Table(OUT/'scores.csv.gz');lead_tables={(split,seed):Table(OUT/f'leads_{split}_{seed}.csv.gz') for split in ['test','extension'] for seed in ['native','42','43','44']}
    manifest_file=(OUT/'evaluation_manifest.jsonl').open('x');native_file=(OUT/'extension_manifest.jsonl').open('x')
    old_entries={(e['trajectory'],e['start'],e['repeat']):e for e in entries('test')}
    started=time.monotonic();bytecount=0;new_paths=0;native_paths=0;generation_seconds=0;completed=0;peak=0
    for entry in plan:
        if time.monotonic()-started>CFG['max_evaluation_seconds']-60 or bytecount>CFG['max_cache_bytes']-1024**3:
            dump(OUT/'EVALUATION_RESOURCE_STOP.json',dict(completed=completed,next_item=entry,elapsed=time.monotonic()-started,cache_bytes=bytecount));break
        if entry['split']=='test':
            source=old_entries[entry['trajectory'],entry['start'],entry['repeat']];b=prior.load_bundle(source)
        else:
            b,source=create_extension(model,norm,entry);save_append_manifest(native_file,source);bytecount+=source['bytes'];native_paths+=192
        # Build every k's mask/std before reading the current future, then issue all predictions.
        inputs={k:prepare(b,k,norm) for k in CFG['train_k']}
        conditions_record={str(k):dict(history_sha256=tensor_hash(h),mask_sha256=tensor_hash(mask),spread_sha256=tensor_hash(spread),noise_sha256=[tensor_hash(v) for v in z],noise_seeds=b['calls']['B' if k==0 else 'S'+str(k)]['seeds']) for k,(h,mask,spread,z) in inputs.items()}
        event('conditions_issued',**entry,conditions=conditions_record)
        issued={};hashes={};seconds={}
        audit=entry['repeat']==0 and entry['start'] in [400,440,820]
        for key,a in adapters.items():
            for k,args in inputs.items():
                torch.cuda.synchronize();t=time.monotonic();p=decode(sample(model,a,*args),norm);torch.cuda.synchronize()
                calltime=time.monotonic()-t;generation_seconds+=calltime;seconds[f'{key}/k{k}']=calltime
                issued[key,k]=p;hashes[f'{key}/k{k}']=tensor_hash(p);new_paths+=32
        record=dict(**entry,source=source['path'],source_sha256=source['sha256'],conditions=conditions_record,outputs=hashes,seconds=seconds)
        if audit:
            file=CACHE/'audit'/f"{entry['split']}_{entry['trajectory']}_{entry['start']}_{entry['repeat']}.pt";file.parent.mkdir(parents=True,exist_ok=True);assert not file.exists()
            torch.save(dict(identity=entry,outputs={f'{key}/k{k}':v.cpu() for (key,k),v in issued.items()},output_sha256=hashes),file)
            bytecount+=file.stat().st_size;record.update(audit_path=str(file),audit_sha256=digest(file))
        event('predictions_issued',**entry,outputs=hashes,audit_sha256=record.get('audit_sha256'))
        truth=prior.frames(entry['trajectory'],entry['start']+20,entry['start']+40).cuda()
        event('truth_revealed',**entry,truth_sha256=tensor_hash(truth))
        matrices={};parent=b['paths']['A'].cuda()
        for k in CFG['train_k']:
            native_s=b['paths']['B' if k==0 else 'S'+str(k)].cuda();direct=b['paths']['B'].cuda();independent=b['paths']['C'].cuda()
            views=['direct'] if k==0 else ['tail','joint']
            for view in views:
                def transform(p):
                    if view=='direct':return p
                    if view=='tail':return p[:,:,:20-k]
                    return torch.cat([parent[:,:,:k],p[:,:,:20-k]],2)
                native_output=transform(native_s)
                direct_view=direct[:,:,k:] if view=='tail' else direct
                independent_view=independent[:,:,k:] if view=='tail' else independent
                target=truth[:,k:] if view=='tail' else truth
                refs=(direct_view,native_output,independent_view)
                structs=tuple(structure(p.double()/std) for p in refs)
                computed={}
                candidates=[('B',direct_view)]
                if k:candidates.append(('S',native_output))
                candidates += [(key,transform(issued[key,k])) for key in adapters]
                for name,p in candidates:
                    rr,cc,mat=scores(p,target,std,view,k,parent,refs,structs)
                    computed[name]=(rr,cc);matrices[f'{name}/k{k}/{view}']=mat
                records=[dict(policy='native',arm=name,seed=-1,step=0,key=name) for name in (['B','S'] if k else ['B'])]
                for chosen in selection:
                    item=chosen['checkpoint'];key=f"{item['arm']}_{item['seed']}_{item['step']}" if item['step'] else ('S' if k else 'B')
                    records.append(dict(policy=chosen['policy'],arm=item['arm'],seed=item['seed'],step=item['step'],key=key))
                for method in records:
                    rr,cc=computed[method['key']]
                    meta=dict(**entry,k=k,view=view,policy=method['policy'],arm=method['arm'],seed=method['seed'],step=method['step'])
                    score_table.write([dict(**meta,**r) for r in rr])
                    leadmeta=dict(meta)
                    lead_tables[entry['split'],'native' if method['seed']==-1 else str(method['seed'])].write([dict(**leadmeta,physical_frame=entry['start']+20+(k if view=='tail' else 0)+r['tail_lead']-1,**r) for r in cc])
        mp=CACHE/'temporal'/f"{entry['split']}_{entry['trajectory']}_{entry['start']}_{entry['repeat']}.pt";mp.parent.mkdir(parents=True,exist_ok=True);assert not mp.exists();torch.save(matrices,mp)
        bytecount+=mp.stat().st_size;record.update(temporal_path=str(mp),temporal_sha256=digest(mp),truth_sha256=tensor_hash(truth))
        save_append_manifest(manifest_file,record);completed+=1;peak=max(peak,torch.cuda.max_memory_allocated())
        if completed%12==0:
            score_table.flush()
            for table in lead_tables.values():table.flush()
            print('EVALUATED',completed,'/528',entry['split'],entry['start'],'adapter paths',new_paths,'seconds',round(time.monotonic()-started,1),flush=True)
    score_table.close()
    for table in lead_tables.values():table.close()
    manifest_file.close();native_file.close()
    assert state_hash(model)==basehash;check()
    dump(OUT/'EVALUATION_COST.json',dict(groups=completed,requested_groups=528,unique_nonzero_checkpoints=len(adapters),adapter_full20_paths=new_paths,native_extension_full20_paths=native_paths,adapter_network_calls=new_paths//32*4,native_extension_network_calls=native_paths//32*4,adapter_generation_seconds=generation_seconds,wall_seconds=time.monotonic()-started,cache_bytes=bytecount,peak_memory_allocated=peak,score_rows=score_table.count,lead_rows={str(k):t.count for k,t in lead_tables.items()},utc=time.time()))
    if completed==528:dump(OUT/'EVALUATION_COMPLETE.json',dict(completed=528,selection_sha256=digest(OUT/'selection.json'),cost_sha256=digest(OUT/'EVALUATION_COST.json'),utc=time.time()))
    print('EVALUATION END',completed,'/528',flush=True)

if __name__=='__main__':main()
