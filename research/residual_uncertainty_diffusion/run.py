"""Complete fixed experiment; training has a hard no-restart marker."""
import time
from .core import *
from .experiment import *
from .prepare import build


def verify_protected():
    for path,sha in read(OUT/'PROTECTED.json').items():assert digest(ROOT/path)==sha,path


def output_change(model,original):
    deltas=[]
    for name,value in model.native.net.out.state_dict().items():deltas.append((value-original[name].cuda()).double().flatten())
    return float(torch.cat(deltas).square().mean().sqrt())


def main():
    assert not (OUT/'TRAINING_STARTED.json').exists(),'Training already started: never automatically repeat updates'
    pre=read(OUT/'PREFLIGHT.json');assert pre['passed']
    for path,sha in pre['code_hashes'].items():assert digest(ROOT/path)==sha,path
    verify_protected()
    native,norm=load_backbone();scales=read(OUT/'CONDITION_SCALES.json')
    train=torch.load(CACHE/'train_conditions.pt',map_location='cpu',weights_only=False)
    dev=torch.load(CACHE/'development_conditions.pt',map_location='cpu',weights_only=False)
    noise,identities=make_noise(train+dev);plan=schedule(train)
    dump(OUT/'NOISE_MANIFEST.json',dict(identities=identities,split_batch_files={s:[r['file'] for r in records()[s]] for s in ('train','development')},
        generator='v2 batch_noise CUDA FP32',schedule=plan))
    dump(OUT/'FROZEN.json',dict(protocol_sha256=digest(CODE/'PROTOCOL.md'),code_hashes=code_hashes(),
        preflight_sha256=digest(OUT/'PREFLIGHT.json'),condition_scales_sha256=digest(OUT/'CONDITION_SCALES.json'),
        data_manifest_sha256=digest(OUT/'DATA_MANIFEST.json'),noise_sha256=digest(OUT/'NOISE_MANIFEST.json'),
        initial_backbone_state=state_hash(native),frozen_non_out=freeze_hash(native,True),seed=42,steps=500))
    initial_out={k:v.detach().cpu().clone() for k,v in native.net.out.state_dict().items()}
    models={a:CorrectedDiffusion(native) for a in ARMS}
    assert state_hash(models['C0'])==state_hash(models['C1'])
    base_rows,_,base_seconds=evaluate(native,None,'native',0,dev,scales,noise,store=True)
    csvwrite(OUT/'DEVELOPMENT.csv',base_rows)
    optimizers={a:torch.optim.Adam([dict(params=models[a].correction.parameters(),lr=3e-4),
                                   dict(params=models[a].native.net.out.parameters(),lr=1e-5)]) for a in ARMS}
    logs={a:(OUT/f'train_{a}.jsonl').open('x') for a in ARMS}
    rows=list(base_rows);delta_rows=[];weights=[];updates=0;cost=[];started=time.perf_counter()
    dump(OUT/'TRAINING_STARTED.json',dict(updates=0,seed=42,plan_sha256=digest(OUT/'NOISE_MANIFEST.json'),utc=time.time()))
    try:
        for step,item in enumerate(plan,1):
            entry=train[item['index']];full=noise[('train',entry['key'].trajectory_id)]
            z=full[item['member_indices']].cuda()
            target=(entry['target'].cuda()-norm['mean'])/norm['std']
            for arm in ARMS:
                model=models[arm];inputs=condition_inputs(entry,scales,z,arm);t=time.perf_counter()
                optimizer=optimizers[arm];optimizer.zero_grad(set_to_none=True)
                p=model.sample(**inputs);loss,first,pair=crps(p,target,True,True)
                assert torch.isfinite(loss)
                loss.backward()
                gg=float(torch.sqrt(sum(v.grad.double().square().sum() for v in model.correction.parameters() if v.grad is not None)))
                go=float(torch.sqrt(sum(v.grad.double().square().sum() for v in model.native.net.out.parameters())))
                assert np.isfinite(gg) and np.isfinite(go)
                optimizer.step();updates+=1;torch.cuda.synchronize()
                record=dict(arm=arm,step=step,key=entry['cache_key'],member_ids=item['member_ids'],fair8_crps=float(loss),
                     member_truth=float(first),member_pair_half=float(pair),G_gradient_norm=gg,out_gradient_norm=go,
                     noise_sha256=tensor_hash(z),seconds=time.perf_counter()-t,updates_total=updates)
                logs[arm].write(json.dumps(record)+'\n');logs[arm].flush()
            if step%50==0:print('TRAIN paired step',step,'updates',updates,flush=True)
            if step in (250,500):
                for arm in ARMS:
                    weights.append(checkpoint(models[arm],arm,step))
                    vals,amplitudes,secs=evaluate(native,models[arm],arm,step,dev,scales,noise,store=True)
                    rows.extend(vals);delta_rows.extend(amplitudes);cost.append(dict(arm=arm,step=step,development_seconds=secs))
                csvwrite(OUT/'DEVELOPMENT.csv',rows);csvwrite(OUT/'DELTA_RMS.csv',delta_rows);dump(OUT/'CHECKPOINTS.json',weights)
                assert all(freeze_hash(models[a].native,True)==freeze_hash(native,True) for a in ARMS)
        selection,tr,macro=select(rows)
        csvwrite(OUT/'DEVELOPMENT_TRAJECTORY.csv',tr);csvwrite(OUT/'DEVELOPMENT_MACRO.csv',macro)
        dump(OUT/'SELECTION.json',selection)
        training=dict(steps_each=500,updates=updates,seed=42,frozen_non_out=freeze_hash(native,True),
            backbone_out_initial_hash=state_hash(native.net.out),backbone_out_initial_tensor_hashes={k:tensor_hash(v) for k,v in initial_out.items()},
            output_changes={a:dict(out_state_hash=state_hash(models[a].native.net.out),out_delta_rms=output_change(models[a],initial_out),
                frozen_non_out_hash=freeze_hash(models[a].native,True)) for a in ARMS})
        assert updates==1000
        dump(OUT/'TRAINING_COMPLETE.json',training)
        costs=dict(training_wall_seconds=time.perf_counter()-started,baseline_development_seconds=base_seconds,
                  checkpoint_evaluation=cost,updates=updates,train_path_members=8,development_path_members=32,
                  XPF_cost=read(OUT/'XPF_CONDITION_AUDIT.json'),cache_bytes=sum(p.stat().st_size for p in CACHE.glob('*.pt')))
        dump(OUT/'COST.json',costs)
        # Freeze selection before any test target access or test generation.
        dump(OUT/'SELECTION_FROZEN.json',dict(selection_sha256=digest(OUT/'SELECTION.json'),checkpoints_sha256=digest(OUT/'CHECKPOINTS.json'),
             protocol_sha256=digest(CODE/'PROTOCOL.md'),code_hashes=code_hashes(),utc=time.time()))
        # Minimal development-only diagnostics on fixed two s420 records.
        diagnostic_entries=[e for e in dev if e['key'].start==420][:2];diagnostics=[]
        for chosen in selection:
            if chosen['chosen_arm']=='native':continue
            arm=chosen['arm'];model=CorrectedDiffusion(native);ck=torch.load(ROOT/next(w['path'] for w in weights if w['arm']==arm and w['step']==chosen['step']),weights_only=False)
            model.correction.load_state_dict(ck['correction']);model.native.net.out.load_state_dict(ck['out'])
            modes=['G_off']+(['E_off','W_off'] if arm=='C1' else [])
            for mode in modes:
                vals,_,_=evaluate(native,model,arm,chosen['step'],diagnostic_entries,scales,noise,enabled=mode!='G_off',ablate=mode[0] if mode in ('E_off','W_off') else None)
                diagnostics.extend(dict(diagnostic=mode,**r) for r in vals)
        csvwrite(OUT/'DEVELOPMENT_ABLATIONS.csv',diagnostics)
        test_rows=[]
        if any(s['chosen_arm']!='native' for s in selection):
            xpf=FrozenXPF();test,_=build('test',xpf,scales);test_noise,test_ids=make_noise(test)
            dump(OUT/'TEST_NOISE_MANIFEST.json',test_ids)
            native_test,_,seconds=evaluate(native,None,'native',0,test,scales,test_noise,store=True);test_rows.extend(native_test)
            for chosen in selection:
                if chosen['chosen_arm']=='native':continue
                arm=chosen['arm'];model=CorrectedDiffusion(native);ck=torch.load(ROOT/next(w['path'] for w in weights if w['arm']==arm and w['step']==chosen['step']),weights_only=False)
                model.correction.load_state_dict(ck['correction']);model.native.net.out.load_state_dict(ck['out'])
                vals,amps,seconds=evaluate(native,model,arm,chosen['step'],test,scales,test_noise,store=True);test_rows.extend(vals)
            csvwrite(OUT/'TEST.csv',test_rows)
        else:dump(OUT/'TEST_STOP.json',dict(reason='Both scientific selections native',test_inference_groups=0))
        (OUT/'DECISION_REPORT.md').write_text(report(rows,selection,costs,training,test_rows))
        verify_protected()
        print('FIXED EXPERIMENT COMPLETE',selection,flush=True)
    except Exception as exc:
        dump(OUT/'STOP.json',dict(error=repr(exc),actual_updates=updates,remaining_updates_not_authorized_for_restart=True))
        raise
    finally:
        for f in logs.values():f.close()

if __name__=='__main__':main()
