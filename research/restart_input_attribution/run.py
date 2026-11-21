"""One fixed pass: prefix-only intervention, issue paths, then reveal scoring tail."""
from core import *
import argparse,shutil


def freeze():
    check();pre=json.loads((OUT/'PREFLIGHT.json').read_text());assert pre['replay_passed']
    # Preflight implementation must remain unchanged; additional execution/analysis files are allowed.
    for p,h in pre['code'].items():assert digest(ROOT/p)==h,p
    assert not (OUT/'FROZEN.json').exists()
    dump(OUT/'FROZEN.json',dict(config_sha256=digest(DELIVERY/'config.json'),protocol_sha256=CFG['protocol_sha256'],code=code_hashes(),preflight_sha256=digest(OUT/'PREFLIGHT.json'),jvp_enabled=pre['jvp_passed'],old_manifest_sha256=digest(old.DELIVERY/'cache_manifest.json'),utc=time.time()))
    print('FROZEN',flush=True)


def jvp_group(model,norm,r,s,z,rtail,stail,k):
    members=CFG['jvp_members'];x=normalize(r[None].expand(4,-1,-1,-1,-1),norm).contiguous()
    e=normalize(s[members],norm)-x;noise=z[members];length=20-k
    assert torch.count_nonzero(e[:,:,:20-k])==0
    fn=lambda h:sampler(model,h,noise)[:,:,:length]
    before=time.monotonic()
    base,linear=torch.autograd.functional.jvp(fn,x,e,strict=True)
    with torch.no_grad():
        full=fn(x+e);half=fn(x+.5*e)
    std=norm['std'].reshape(2,1,1,1)
    actual=(stail[members]-rtail[members])/std
    half_response=half-base
    replay_r=float((decode(base,norm)-rtail[members]).abs().max())
    replay_s=float((decode(full,norm)-stail[members]).abs().max())
    replay_passed=max(replay_r,replay_s)<=CFG['replay_atol']
    rows=[]
    for n,member in enumerate(members):
        for ch,name in [(slice(None),'overall'),(slice(0,1),'u'),(slice(1,2),'v')]:
            for dose,lin,act,direction in [('full',linear,actual,e),('half',.5*linear,half_response,.5*e)]:
                rows.append(dict(member=member,channel=name,dose=dose,**tangent_metrics(lin[n,ch],act[n,ch],direction[n,ch],k),replay_R_maxabs=replay_r,replay_S_maxabs=replay_s,replay_passed=int(replay_passed)))
    fields=dict(input_error=e.cpu(),linear=linear.cpu(),half=half.cpu(),actual=actual.cpu(),base=base.cpu(),full=full.cpu(),member_ids=members,replay_R_maxabs=replay_r,replay_S_maxabs=replay_s,replay_passed=replay_passed)
    torch.cuda.synchronize()
    return rows,fields,time.monotonic()-before


def quality_full(paths,truth,std):
    rr,cc=old.quality(paths,truth,std)
    # Complete lead-level diagnostics; time-mean CRPS at a single lead equals CRPS.
    pn=paths.double()/std;yn=truth.double()/std
    for row in cc:
        ch='uv'.index(row['channel']);t=row['tail_lead']-1;scale=float(std[ch,0,0,0])
        row.update(energy=float(old.energy_score(pn[:,ch:ch+1,t:t+1],yn[ch:ch+1,t:t+1])),standardized_variance=row['variance']/scale**2,standardized_width90=row['width90']/scale,time_mean_crps=row['crps'])
    return rr,cc


def formal():
    check();frozen=json.loads((OUT/'FROZEN.json').read_text())
    assert frozen['config_sha256']==digest(DELIVERY/'config.json')
    for p,h in frozen['code'].items():assert digest(ROOT/p)==h,p
    assert not (OUT/'RUN_STARTED.json').exists(),'Never overwrite or implicitly repeat a started experiment'
    model,norm=model_load();std=norm['std'].reshape(2,1,1,1)
    CACHE.mkdir(parents=True,exist_ok=True);assert not list(CACHE.glob('*.pt'));assert shutil.disk_usage(CACHE).free>CFG['max_cache_bytes']
    plan=[dict(**{key:e[key] for key in META if key!='k'},k=k,source=e['path'],source_sha256=e['sha256']) for e in entries() if e['split'] in ['development','test'] for k in CFG['k']]
    assert len(plan)==288
    dump(OUT/'plan.json',plan);before=time.monotonic();torch.cuda.reset_peak_memory_stats()
    dump(OUT/'RUN_STARTED.json',dict(utc=time.time(),frozen_sha256=digest(OUT/'FROZEN.json'),plan_sha256=digest(OUT/'plan.json')))
    dump(OUT/'environment.json',dict(torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),tf32=torch.backends.cuda.matmul.allow_tf32,threads=torch.get_num_threads()))
    manifest=[];q=[];leads=[];resp=[];ens=[];signed=[];jrows=[];jseconds=0.;gseconds=0.;cache_bytes=0;jgroups=0
    def save_tables():
        for name,rows in [('quality',q),('lead_channel',leads),('response_members',resp),('response_ensemble',ens),('signed_mse',signed),('jvp',jrows)]:
            if rows:csvwrite(OUT/(name+'.csv'),rows)
        dump(OUT/'cache_manifest.json',manifest)
    for entry in [e for e in entries() if e['split'] in ['development','test']]:
        b=load_bundle(entry)
        for k in CFG['k']:
            meta={key:entry[key] for key in META if key!='k'};meta['k']=k;length=20-k
            estimate=(64*2*length*32*64+2*2*20*32*64)*4+20*1024**2
            if time.monotonic()-before>CFG['max_seconds']-30 or cache_bytes+estimate>CFG['max_cache_bytes']:
                save_tables();dump(OUT/'RESOURCE_STOP.json',dict(completed=len(manifest),next_item=meta,seconds=time.monotonic()-before,cache_bytes=cache_bytes));return
            prefix=oracle_prefix(entry['trajectory'],entry['start'],k)
            r,m,s=inputs(b['history'],b['paths']['A'],prefix,k)
            event('oracle_prefix_revealed',**meta,indices=list(range(entry['start']+20,entry['start']+20+k)),sha256=tensor_hash(prefix))
            r,m,s=r.cuda(),m.cuda(),s.cuda();z=noise_for(b,k,s)
            outputs={};calls={}
            with torch.no_grad():
                for group,h in [('R',r),('M',m)]:
                    hn=normalize(h[None].expand(32,-1,-1,-1,-1),norm)
                    torch.cuda.synchronize();t=time.monotonic();p=decode(sampler(model,hn,z),norm);torch.cuda.synchronize()
                    seconds=time.monotonic()-t;gseconds+=seconds
                    outputs[group]=p[:,:,:length].contiguous().cpu()
                    calls[group]=dict(seconds=seconds,network_calls=4,samples=32,full_output_sha256=tensor_hash(p),tail_sha256=tensor_hash(outputs[group]),history_sha256=tensor_hash(h),noise_sha256=b['calls']['S'+str(k)]['noise_sha256'],seeds=b['calls']['S'+str(k)]['seeds'])
            path=CACHE/(f"{entry['split']}_{entry['trajectory']}_{entry['start']}_{entry['repeat']}_k{k}.pt")
            assert not path.exists()
            bundle=dict(identity=meta,source=entry['path'],source_sha256=entry['sha256'],history_R=r.cpu(),history_M=m.cpu(),paths=outputs,calls=calls,compared_frames=list(range(entry['start']+20+k,entry['start']+40)))
            torch.save(bundle,path);cache_bytes+=path.stat().st_size
            item=dict(**meta,path=str(path),sha256=digest(path),bytes=path.stat().st_size,source=entry['path'],source_sha256=entry['sha256'],calls=calls)
            manifest.append(item)
            event('paths_issued',**meta,sha256=item['sha256'])
            # Only now does the scoring-tail interface run.
            truth=frames(entry['trajectory'],entry['start']+20+k,entry['start']+40).cuda()
            event('scoring_tail_revealed',**meta,indices=bundle['compared_frames'],sha256=tensor_hash(truth))
            rt,mt=outputs['R'].cuda(),outputs['M'].cuda();st=b['paths']['S'+str(k)][:,:,:length].cuda()
            with torch.no_grad():
                for group,p in [('R',rt),('M',mt),('S',st),('B',b['paths']['B'][:,:,k:].cuda())]:
                    rr,cc=quality_full(p,truth,std)
                    q.extend(dict(**meta,method=group,**row) for row in rr)
                    leads.extend(dict(**meta,method=group,physical_frame=entry['start']+20+k+row['tail_lead']-1,**row) for row in cc)
                rr,ee=response_rows(rt,mt,st,std)
                resp.extend(dict(**meta,**row) for row in rr);ens.extend(dict(**meta,**row) for row in ee)
                signed.extend(dict(**meta,**row) for row in signed_rows(rt,st,truth,std))
            if entry['start']==400 and entry['repeat']==0 and frozen['jvp_enabled']:
                rr,fields,seconds=jvp_group(model,norm,r,s,z,rt,st,k);jseconds+=seconds;jgroups+=1
                jrows.extend(dict(**meta,**row) for row in rr)
                jp=path.with_name(path.stem+'_jvp.pt');assert not jp.exists();torch.save(fields,jp);cache_bytes+=jp.stat().st_size
                item.update(jvp_path=str(jp),jvp_sha256=digest(jp),jvp_seconds=seconds,jvp_replay_passed=fields['replay_passed'])
            if len(manifest)%12==0:
                save_tables();print('PROGRESS',len(manifest),'/288','paths',len(manifest)*64,'jvp_directions',jgroups*4,'seconds',round(time.monotonic()-before,2),flush=True)
    save_tables()
    check();model_load() # frozen checkpoint hashes and packaged/original equality again
    dump(OUT/'COMPLETE.json',dict(groups=len(manifest),paths=len(manifest)*64,trajectories=24,jvp_groups=jgroups,jvp_directions=jgroups*4,jvp_reverse_sweeps=jgroups*2,main_network_calls=len(manifest)*8,main_sample_equivalent_network_calls=len(manifest)*64*4,jvp_forward_network_calls=jgroups*4,jvp_replay_and_half_network_calls=jgroups*8,jvp_forward_sample_equivalent_network_calls=jgroups*4*4,jvp_replay_and_half_sample_equivalent_network_calls=jgroups*8*4,half_paths=jgroups*4,lambda1_replay_paths=jgroups*4,seconds=time.monotonic()-before,generation_seconds=gseconds,jvp_seconds=jseconds,cache_bytes=cache_bytes,peak_memory_allocated=torch.cuda.max_memory_allocated(),peak_memory_reserved=torch.cuda.max_memory_reserved(),jvp_replay_all_passed=all(e.get('jvp_replay_passed',True) for e in manifest),utc=time.time()))
    print('COMPLETE',len(manifest),'groups',len(manifest)*64,'new full paths',flush=True)

if __name__=='__main__':
    configure();parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['freeze','formal']);args=parser.parse_args();globals()[args.stage]()
