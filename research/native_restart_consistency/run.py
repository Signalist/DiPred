"""Protocol-bound generation/cache first, truth scoring second. No optimizer exists."""
import argparse,shutil
from core import *


def event(action,**kw):
    with (OUT/'events.jsonl').open('a') as f:f.write(json.dumps(dict(action=action,utc=time.time(),**kw))+'\n')


def load_model():
    check_dependencies();model,norm,frozen=load_backbone(SimpleNamespace(workspace=WORK));assert frozen==CFG['frozen']
    assert all(not p.requires_grad for p in model.parameters())
    return model,norm


def train_history_reference(norm):
    histories=torch.cat([raw(DATA,RECORDS['train'],start) for start in CFG['starts']])
    values=history_features(histories,norm)
    return dict(mean=values.mean(0),std=values.std(0,unbiased=False).clamp_min(1e-8)),values


def identity(split,record,start,repeat):
    return dict(split=split,trajectory=record['file'],Re=record['group'],start=start,repeat=repeat,config_sha256=CONFIG_HASH,history_indices=physical_indices(start,0)['history'],direct_indices=physical_indices(start,0)['direct'],k={str(k):physical_indices(start,k) for k in CFG['k']},member_ids=list(range(32)),parent_child_ids=[[i,i] for i in range(32)],batch_order='member0..31',frozen=CFG['frozen'])


@torch.no_grad()
def create_bundle(model,norm,history,ident,ks=None):
    ks=CFG['k'] if ks is None else ks;paths={};calls={}
    h=history[None].expand(32,-1,-1,-1,-1).contiguous()
    for family in ['A','B','C']:
        seeds=seeds_for(ident['split'],ident['trajectory'],ident['start'],ident['repeat'],family,0)
        paths[family],calls[family]=generate(model,norm,h,seeds)
    for k in ks:
        hp=restart_history(history,paths['A'],k)
        seeds=seeds_for(ident['split'],ident['trajectory'],ident['start'],ident['repeat'],'S',k)
        paths['S'+str(k)],calls['S'+str(k)]=generate(model,norm,hp,seeds)
    return dict(identity=ident,history=history.cpu(),paths={k:v.cpu() for k,v in paths.items()},calls=calls,history_sha256=tensor_hash(history))


@torch.no_grad()
def preflight():
    assert json.loads((OUT/'UNIT_TESTS.json').read_text())['passed']
    model,norm=load_model();CACHE.mkdir(parents=True,exist_ok=True);all_calls=[];replay=[];k0=[]
    for record in RECORDS['train'][:2]:
        history=raw(DATA,[record],400)[0];ident=identity('preflight',record,400,0)
        bundle=create_bundle(model,norm,history,ident,ks=[0,2,5,10])
        # Exact input noise and original rollout; batch rounding only is tolerated.
        seed=bundle['calls']['A']['seeds'][0];h=(history[None]-norm['mean'])/norm['std']
        direct=model.rollout(h,seed,sign=1)*norm['std']+norm['mean']
        error=float((direct[0].cpu()-bundle['paths']['A'][0]).abs().max());assert error<2e-6
        replay.append(dict(trajectory=record['file'],rollout_max_error=error))
        path=CACHE/('preflight_'+record['file']+'.pt');torch.save(bundle,path);loaded=torch.load(path,map_location='cpu',weights_only=False)
        assert loaded['identity']==ident and torch.equal(loaded['paths']['A'],bundle['paths']['A'])
        hashes=[v for c in bundle['calls'].values() for v in c['noise_sha256']];assert len(hashes)==len(set(hashes))
        std=norm['std'].cpu().reshape(2,1,1,1)
        b=bundle['paths']['B'].double()/std;c=bundle['paths']['C'].double()/std;s=bundle['paths']['S0'].double()/std
        k0.append(dict(trajectory=record['file'],bc_u=energy_distance(b,c)['u'],bs0_u=energy_distance(b,s)['u']))
        all_calls.append(dict(path=str(path),sha256=digest(path),calls=bundle['calls']))
    reference,values=train_history_reference(norm)
    dump(OUT/'training_history_reference.json',dict(features=FEATURES,mean=reference['mean'].tolist(),std=reference['std'].tolist(),histories=len(values),normalization={k:v.tolist() for k,v in norm.items()}))
    dump(OUT/'data_hashes.json',{r['file']:digest(DATA/r['file']) for records in RECORDS.values() for r in records})
    dump(OUT/'PREFLIGHT.json',dict(passed=True,config_sha256=CONFIG_HASH,rollout_replay=replay,k0_sampling_diagnostic=k0,caches=all_calls,network_calls=14*4+2*4,samples=14*32+2,code={str(p.relative_to(ROOT)):digest(p) for p in Path(__file__).resolve().parent.glob('*.py')},utc=time.time()))
    print('PASS native sampler replay, two train histories, cache roundtrip, unique actual noise, k0 same condition; no outcome selection',flush=True)


def freeze():
    assert not (OUT/'FROZEN.json').exists();pre=json.loads((OUT/'PREFLIGHT.json').read_text());assert pre['passed']
    check_dependencies()
    for name,sha in pre['code'].items():assert digest(ROOT/name)==sha
    dump(OUT/'FROZEN.json',dict(config_sha256=CONFIG_HASH,protocol_sha256=CFG['protocol_sha256'],code={str(p.relative_to(ROOT)):digest(p) for p in Path(__file__).resolve().parent.glob('*.py')},preflight_sha256=digest(OUT/'PREFLIGHT.json'),training_history_sha256=digest(OUT/'training_history_reference.json'),data_sha256=digest(OUT/'data_hashes.json'),utc=time.time()))
    print('FROZEN protocol, model, seeds, code and history reference before formal scoring',flush=True)


def history_rows(bundle,norm,reference):
    rows=[];h=bundle['history'].cuda();parent=bundle['paths']['A'].cuda()
    for k in [0]+CFG['k']:
        histories=h[None] if k==0 else restart_history(h,parent,k)
        f=history_features(histories,norm);z=(f-reference['mean'])/reference['std']
        for i in range(len(histories)):
            for ch in range(2):
                rows.append(dict(k=k,parent_id=i if k else -1,channel='uv'[ch],**{name:float(f[i,ch,j]) for j,name in enumerate(FEATURES)},max_abs_z=float(z[i,ch].abs().max()),fraction_abs_z_above3=float((z[i,ch].abs()>3).double().mean()),**{name+'_z':float(z[i,ch,j]) for j,name in enumerate(FEATURES)}))
    return rows


@torch.no_grad()
def formal():
    assert not (OUT/'RUN_STARTED.json').exists(),'Completed/started fixed experiment cannot automatically restart'
    frozen=json.loads((OUT/'FROZEN.json').read_text());assert frozen['config_sha256']==CONFIG_HASH
    for name,sha in frozen['code'].items():assert digest(ROOT/name)==sha
    model,norm=load_model();std=norm['std'].reshape(2,1,1,1)
    ref=json.loads((OUT/'training_history_reference.json').read_text());reference={k:torch.tensor(ref[k],device='cuda',dtype=torch.double) for k in ['mean','std']}
    CACHE.mkdir(parents=True,exist_ok=True);assert shutil.disk_usage(CACHE).free>CFG['max_cache_bytes']
    before=time.monotonic();torch.cuda.reset_peak_memory_stats()
    dump(OUT/'RUN_STARTED.json',dict(utc=time.time(),frozen_sha256=digest(OUT/'FROZEN.json')))
    dump(OUT/'environment.json',dict(torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),tf32=torch.backends.cuda.matmul.allow_tf32))
    manifest=[];distribution=[];quality_rows=[];cells=[];history_stats=[];calls=0;samples=0;bytes_written=0
    for split in ['train','development','test']:
        for start in CFG['starts']:
            for repeat in range(2):
                for record in RECORDS[split]:
                    if time.monotonic()-before>CFG['max_seconds'] or bytes_written>CFG['max_cache_bytes']:
                        dump(OUT/'RESOURCE_STOP.json',dict(elapsed=time.monotonic()-before,next_item=[split,start,repeat,record['file']],completed=len(manifest),cache_bytes=bytes_written));return
                    ident=identity(split,record,start,repeat);file=CACHE/f"{split}_{record['file']}_{start}_{repeat}.pt"
                    assert not file.exists(),'New seed identity already exists; review before reuse instead of overwriting'
                    h=raw(DATA,[record],start)[0]
                    bundle=create_bundle(model,norm,h,ident)
                    torch.save(bundle,file);sha=digest(file);bytes_written+=file.stat().st_size
                    for c in bundle['calls'].values():calls+=c['network_calls'];samples+=c['samples']
                    metadata=dict(split=split,trajectory=record['file'],Re=record['group'],start=start,repeat=repeat)
                    entry=dict(**metadata,path=str(file),sha256=sha,bytes=file.stat().st_size,history_sha256=bundle['history_sha256'],outputs={k:c['output_sha256'] for k,c in bundle['calls'].items()},calls=bundle['calls'])
                    manifest.append(entry);dump(OUT/'cache_manifest.json',manifest)
                    event('issued_all_paths',**metadata,cache_sha256=sha,outputs=entry['outputs'])
                    # No future truth opened before every compared path is safely cached.
                    truth=raw(DATA,[record],start+20)[0]
                    event('truth_revealed',**metadata,truth_sha256=tensor_hash(truth))
                    for r in history_rows(bundle,norm,reference):history_stats.append(dict(**metadata,**r))
                    for k in CFG['k']:
                        b=bundle['paths']['B'][:,:,k:].cuda().double()/std
                        c=bundle['paths']['C'][:,:,k:].cuda().double()/std
                        s=bundle['paths']['S'+str(k)][:,:,:20-k].cuda().double()/std
                        actual=discrepancy(b,s);baseline=discrepancy(b,c)
                        distribution.append(dict(**metadata,k=k,length=20-k,**{'restart_'+m:v for m,v in actual.items()},**{'sampling_'+m:v for m,v in baseline.items()},**{'excess_'+m:actual[m]-baseline[m] for m in actual},native_spread=float(b.var(0,unbiased=False).mean()),native_temporal_rms=float(b.mean(0).diff(dim=1).square().mean().sqrt())))
                        for family in ['A','B','C','S']:
                            path=bundle['paths'][family if family!='S' else 'S'+str(k)].cuda()
                            tail=path[:,:,k:] if family!='S' else path[:,:,:20-k]
                            rr,cc=quality(tail,truth[:,k:],std)
                            quality_rows += [dict(**metadata,k=k,length=20-k,method=family,**r) for r in rr]
                            cells += [dict(**metadata,k=k,method=family,physical_frame=start+20+k+r['tail_lead']-1,**r) for r in cc]
                    if len(manifest)%4==0:
                        csvwrite(OUT/'distribution.csv',distribution);csvwrite(OUT/'quality.csv',quality_rows);csvwrite(OUT/'lead_channel.csv',cells);csvwrite(OUT/'history_stats.csv',history_stats)
                        print('PROGRESS',len(manifest),'/280',split,start,repeat,record['file'],'elapsed',round(time.monotonic()-before,1),flush=True)
    csvwrite(OUT/'distribution.csv',distribution);csvwrite(OUT/'quality.csv',quality_rows);csvwrite(OUT/'lead_channel.csv',cells);csvwrite(OUT/'history_stats.csv',history_stats)
    dump(OUT/'COMPLETE.json',dict(groups=len(manifest),trajectories=70,network_calls=calls,batched_sampler_calls=calls//4,samples=samples,peak_memory_allocated=torch.cuda.max_memory_allocated(),peak_memory_reserved=torch.cuda.max_memory_reserved(),wall_seconds=time.monotonic()-before,generation_seconds=sum(c['seconds'] for m in manifest for c in m['calls'].values()),cache_bytes=bytes_written,config_sha256=CONFIG_HASH,frozen_sha256=digest(OUT/'FROZEN.json'),utc=time.time()))
    print('COMPLETE',len(manifest),'groups',samples,'paths',calls,'network calls',flush=True)

if __name__=='__main__':
    configure();OUT.mkdir(parents=True,exist_ok=True)
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['preflight','freeze','formal']);args=parser.parse_args();globals()[args.stage]()
