"""One bounded frozen audit; original sampler is observed using a forward hook."""
import os
os.environ['TORCH_ALLOW_TF32_CUBLAS_OVERRIDE']='0'
import sys,json,time,hashlib,csv,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from research.provenance_uncertainty_conditioning import core as old
OUT=Path(os.environ.get('DIPRED_FAILURE_OUT',str(ROOT/'results/conditioning_failure_audit')))
SPEC=ROOT/'results/conditioning_failure_audit'
CFG=json.loads((SPEC/'config.json').read_text())
dump,csvwrite,digest=old.dump,old.csvwrite,old.digest
METRICS=['crps','member_truth','member_pair_half','mse','smse']


def check():
    assert digest(SPEC/'PROTOCOL.md')==CFG['protocol_sha256']
    assert digest(SPEC/'manifest.json')==CFG['manifest_sha256']
    for p,h in CFG['dependencies'].items():assert digest(ROOT/p)==h,p


def score(p,y,std):
    p,y=p.double(),y.double();pn,yn=p/std,y/std
    loss,first,pair=old.crps_parts(pn,yn)
    e=p.mean(0)-y
    return dict(crps=float(loss),member_truth=float(first),member_pair_half=float(pair),mse=float(e.square().mean()),smse=float((e/std).square().mean()))


def event(action,**kw):
    with (OUT/'events.jsonl').open('a') as f:f.write(json.dumps(dict(action=action,utc=time.time(),**kw))+'\n')


def main():
    old.configure();OUT.mkdir(parents=True,exist_ok=True);assert not (OUT/'STARTED.json').exists()
    start=time.monotonic();check()
    manifest=json.loads((SPEC/'manifest.json').read_text());assert len(manifest)==24
    assert all(e['split'] in ['train','development'] and e['start']==400 and e['repeat']==0 for e in manifest)
    assert {e['trajectory'] for e in manifest if e['split']=='train'}==set(CFG['train_files'])
    selections=json.loads((old.DELIVERY/'selection.json').read_text());assert len(selections)==18 and all(r['step']==0 for r in selections)
    model,norm=old.load_model();model.eval().requires_grad_(False);basehash=old.state_hash(model)
    adapters={w['step']:old.read_checkpoint(w).eval().requires_grad_(False) for w in CFG['weights']}
    assert set(adapters)=={0,250,1000}
    for w in CFG['weights']:
        ck=torch.load(ROOT/w['path'],map_location='cpu',weights_only=False)
        assert ck['config_sha256']==digest(old.DELIVERY/'config.json')
    assert torch.count_nonzero(adapters[0].out.weight)==torch.count_nonzero(adapters[0].out.bias)==0
    pre=json.loads((old.DELIVERY/'PREFLIGHT.json').read_text());assert pre['passed']
    # Pure score arithmetic test needs no model forward.
    x=torch.arange(48,dtype=torch.double).reshape(4,2,3,2)/17;y=x.mean(0)+.13
    crps=old.crps_parts(x,y)[0];explicit=(x-y).abs().mean()-.5*(x[:,None]-x[None,:]).abs().mean()
    assert abs(float(crps-explicit))<1e-10
    dev=old.read_csv(old.DELIVERY/'development.csv')
    devix={(r['trajectory'],int(r['step']),r['channel']):r for r in dev if r['arm']=='C_aug' and r['seed']=='42' and r['start']=='400' and r['repeat']=='0' and r['k']=='5'}
    nativeq=old.read_csv(old.native.DELIVERY/'quality.csv')
    nativeix={(r['split'],r['trajectory'],r['channel']):r for r in nativeq if r['split'] in ['train','development'] and r['start']=='400' and r['repeat']=='0' and r['k']=='5' and r['method']=='S'}
    sourcehash=json.loads((old.native.DELIVERY/'data_hashes.json').read_text())
    frozen=dict(code_sha256=digest(Path(__file__)),config_sha256=digest(SPEC/'config.json'),protocol_sha256=CFG['protocol_sha256'],manifest_sha256=CFG['manifest_sha256'],weights=CFG['weights'],backbone_state_sha256=basehash,normalization={k:v.tolist() for k,v in norm.items()},utc=time.time())
    dump(OUT/'FROZEN.json',frozen);dump(OUT/'STARTED.json',dict(utc=time.time(),optimizer_updates=0))
    rows=[];deltas=[];identities=[];checks=[];forward=0;reused=0;generation_seconds=0.;scoremax=0.;identitymax=0.
    try:
        with torch.no_grad():
            for entry in manifest:
                assert time.monotonic()-start<CFG['max_seconds'],'time budget'
                assert digest(old.native.DATA/entry['trajectory'])==sourcehash[entry['trajectory']]
                b=old.prior.load_bundle(entry)
                assert b['identity']['member_ids']==list(range(32)) and b['identity']['parent_child_ids']==[[i,i] for i in range(32)]
                h,mask,spread,z=old.prepare(b,5,norm)
                assert old.tensor_hash(old.native.restart_history(b['history'].cuda(),b['paths']['A'].cuda(),5))==b['calls']['S5']['history_sha256']
                assert not set(b['calls']['S5']['seeds'])&set(b['calls']['A']['seeds'])
                predictions={0:b['paths']['S5'].cuda()};delta_fields={}
                for step,a in adapters.items():
                    captured=[]
                    if step==0:
                        for label in [99,66,33,0]:captured.append((label,a(h,mask,spread,label)))
                        assert all(torch.count_nonzero(v)==0 for _,v in captured);reused+=1
                    else:
                        def capture(module,args,output):captured.append((int(args[3]),output.detach().clone()))
                        hook=a.register_forward_hook(capture);torch.cuda.synchronize();t=time.monotonic()
                        try:predictions[step]=old.decode(old.sample(model,a,h,mask,spread,z),norm)
                        finally:hook.remove()
                        torch.cuda.synchronize();generation_seconds+=time.monotonic()-t;forward+=1
                        assert forward+reused<=72
                    assert [label for label,v in captured]==[99,66,33,0]
                    for label,v in captured:
                        for region,sl,nframes in [('observed',slice(0,15),15),('generated',slice(15,20),5)]:
                            for channel,ch,nch in [('overall',slice(None),2),('u',slice(0,1),1),('v',slice(1,2),1)]:
                                data=v[:,ch,sl].double();assert data.numel()==32*nch*nframes*32*64
                                deltas.append(dict(split=entry['split'],trajectory=entry['trajectory'],Re=entry['Re'],step=step,denoise=label,region=region,channel=channel,elements=data.numel(),rms=float(data.square().mean().sqrt())))
                meta={k:entry[k] for k in ['split','trajectory','Re','start','repeat']}
                outputs={str(k):old.tensor_hash(v) for k,v in predictions.items()}
                event('predictions_complete',**meta,outputs=outputs,source_sha256=entry['sha256'])
                # Single full20 label read after all predictions and diagnostic deltas exist.
                truth=old.prior.frames(entry['trajectory'],425,445).cuda()
                event('truth_revealed',**meta,indices=list(range(425,445)),truth_sha256=old.tensor_hash(truth))
                std=norm['std'].reshape(2,1,1,1)
                for step,p in predictions.items():
                    for channel,ch in [('overall',slice(None)),('u',slice(0,1)),('v',slice(1,2))]:
                        part={}
                        for segment,sl in [('full',slice(None)),('overlap',slice(0,15)),('extra',slice(15,20))]:
                            values=score(p[:,ch,sl],truth[ch,sl],std[ch]);part[segment]=values
                            rows.append(dict(**meta,step=step,segment=segment,channel=channel,**values))
                        for metric in METRICS:
                            err=abs(part['full'][metric]-.75*part['overlap'][metric]-.25*part['extra'][metric]);identitymax=max(identitymax,err);assert err<=1e-12,(meta,step,metric,err)
                        references=[]
                        if step==0:references.append(('native',nativeix[entry['split'],entry['trajectory'],channel]))
                        if entry['split']=='development':references.append(('development',devix[entry['trajectory'],step,channel]))
                        for name,ref in references:
                            for metric in ['crps','mse','smse']:
                                err=abs(part['overlap'][metric]-float(ref[metric]));scoremax=max(scoremax,err)
                                checks.append(dict(**meta,step=step,channel=channel,reference=name,metric=metric,maxabs=err))
                                assert err<=CFG['score_atol'],(meta,step,channel,metric,err)
                identities.append(dict(**meta,source=entry['path'],source_sha256=entry['sha256'],history_sha256=b['history_sha256'],parent_sha256=entry['outputs']['A'],noise_seeds=b['calls']['S5']['seeds'],noise_sha256=b['calls']['S5']['noise_sha256'],outputs=outputs,truth_sha256=old.tensor_hash(truth)))
                csvwrite(OUT/'trajectory_scores.csv',rows);csvwrite(OUT/'delta_by_trajectory.csv',deltas);csvwrite(OUT/'score_replay_checks.csv',checks);dump(OUT/'identities.json',identities)
                print('AUDITED',len(identities),'/24',entry['split'],entry['trajectory'],flush=True)
        assert forward==48 and reused==24 and len(rows)==648 and len(deltas)==1728
        assert old.state_hash(model)==basehash
        assert all(not p.requires_grad and p.grad is None for a in [model,*adapters.values()] for p in a.parameters())
        for w in CFG['weights']:assert old.state_hash(adapters[w['step']])==w['state_sha256']
        check()
        dump(OUT/'COMPLETE.json',dict(passed=True,trajectories=24,new_forward_groups=forward,reused_step0_groups=reused,new_paths=forward*32,reused_paths=reused*32,new_network_batch_calls=forward*4,optimizer_updates=0,score_rows=len(rows),delta_rows=len(deltas),existing_score_maxabs=scoremax,segment_identity_maxabs=identitymax,generation_seconds=generation_seconds,wall_seconds=time.monotonic()-start,prediction_cache_bytes=0,all_parameters_frozen=True,no_test_extension_scored=True,old_files_unchanged=True,utc=time.time()))
    except Exception as ex:
        dump(OUT/'STOP.json',dict(reason=str(ex),completed_trajectories=len(identities),new_forward_groups=forward,reused_step0_groups=reused,optimizer_updates=0,wall_seconds=time.monotonic()-start,traceback=traceback.format_exc(),utc=time.time()))
        raise

if __name__=='__main__':main()
