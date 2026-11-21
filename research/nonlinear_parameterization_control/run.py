"""Nine bounded CRPS fits, dual dev selection, freeze, then exploratory scoring."""
import argparse
from experiment import *


def train():
    assert not (OUT/'training_started.json').exists(),'Never automatically repeat or resume formal training'
    pre=json.loads((OUT/'PREFLIGHT.json').read_text());assert pre['passed'] and pre['config_sha256']==CONFIG_HASH
    for name,sha in pre['code'].items():assert digest(ROOT/name)==sha
    check_dependencies();configure()
    train,dev=load_data('train'),load_data('development')
    generator=torch.Generator(device='cpu').manual_seed(CFG['sampling_seed'])
    schedule=torch.randint(len(train.x),(CFG['steps'],CFG['batch']),generator=generator)
    dump(OUT/'batch_schedule.json',schedule.tolist())
    started=time.monotonic()
    dump(OUT/'training_started.json',dict(utc=time.time(),config_sha256=CONFIG_HASH,schedule_sha256=digest(OUT/'batch_schedule.json')))
    dump(OUT/'environment.json',dict(torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),tf32=torch.backends.cuda.matmul.allow_tf32))
    (OUT/'weights').mkdir(exist_ok=True)
    candidates=[];selections=[];records=[];cells=[];trace=[];cost=[];sens=[];runs=[];maps={}
    native_rows,native_cells=full_scores(dev,torch.zeros_like(dev.b),'native')
    native=mean_metrics(native_rows)
    for kind in ARMS:
        for seed in SEEDS:
            model=make_model(kind,seed,train.std)
            init_hash=state_hash(model.state_dict());optimizer=torch.optim.Adam(model.parameters(),lr=CFG['lr'])
            run_start=time.monotonic();run_id=kind+'_'+str(seed);own=[]
            for step in range(CFG['steps']+1):
                if time.monotonic()-started>CFG['max_training_seconds']:raise RuntimeError('fixed training time budget exhausted; no automatic continuation')
                if step:
                    model.train();idx=schedule[step-1].to('cuda');q=model(train.x[idx])
                    value=loss(train.p[idx],train.y[idx],q,train.std)
                    assert torch.isfinite(value),'nonfinite loss; stop, no extra configuration'
                    optimizer.zero_grad(set_to_none=True);value.backward()
                    grad=float(torch.stack([p.grad.detach().square().sum() for p in model.parameters()]).sum().sqrt())
                    assert np.isfinite(grad),'nonfinite gradient'
                    trace.append(dict(kind=kind,seed=seed,step=step,crps_without_constant=float(value),gradient_l2=grad,q_rms=float(q.detach().square().mean().sqrt()),q_standardized_rms=float((q.detach()/train.std).square().mean().sqrt())))
                    optimizer.step()
                if step%CFG['eval_every']:continue
                model.eval()
                with torch.no_grad():
                    q=predict(model,dev.x)
                    rr,cc=full_scores(dev,q,run_id,seed)
                vals=mean_metrics(rr)
                file=OUT/'weights'/f'{run_id}_{step}.pt'
                torch.save(dict(state=cpu_state(model),kind=kind,seed=seed,step=step,config_sha256=CONFIG_HASH),file)
                candidate=dict(kind=kind,seed=seed,step=step,**vals,admissible=admissible(vals,native),file=file.name,sha256=digest(file),initial_state_sha256=init_hash)
                candidates.append(candidate);own.append(candidate)
                for r in rr+cc:r.update(kind=kind,step=step)
                records+=rr;cells+=cc
                csvwrite(OUT/'candidates.csv',candidates)
                if trace:csvwrite(OUT/'training.csv',trace)
                print('CHECKPOINT',kind,seed,step,json.dumps(vals),flush=True)
            elapsed=time.monotonic()-run_start
            for selection,protected in [('scientific',False),('deployment',True)]:
                best=choose(own,protected)
                selections.append(dict(selection=selection,**best))
                ck=torch.load(OUT/'weights'/best['file'],map_location='cuda',weights_only=False)
                model.load_state_dict(ck['state']);model.eval()
                rr,mm=sensitivity(model,train.x[:1],selection);sens += [dict(seed=seed,selected_step=best['step'],**r) for r in rr]
                maps[run_id+'_'+selection]=mm
            # Fixed batch2 inference microbenchmark, excludes conditions/cache/backbone.
            with torch.no_grad():
                for _ in range(3):model(train.x[:2])
                torch.cuda.synchronize();t=time.perf_counter()
                for _ in range(20):model(train.x[:2])
                torch.cuda.synchronize();seconds=(time.perf_counter()-t)/40
            cost.append(dict(kind=kind,seed=seed,params=sum(p.numel() for p in model.parameters()),training_and_validation_seconds=elapsed,inference_seconds_per_block=seconds,batch=2,repetitions=20))
            runs.append(dict(kind=kind,seed=seed,steps=1000,initial_state_sha256=init_hash,schedule_sha256=digest(OUT/'batch_schedule.json')))
            csvwrite(OUT/'selection.csv',selections);csvwrite(OUT/'cost.csv',cost);csvwrite(OUT/'sensitivity.csv',sens)
            csvwrite(OUT/'development_scores.csv',records);csvwrite(OUT/'development_detail.csv',cells)
            dump(OUT/'runs.json',runs)
    torch.save(maps,OUT/'sensitivity_maps.pt')
    dump(OUT/'training_complete.json',dict(runs=len(runs),steps=9000,seconds=time.monotonic()-started,config_sha256=CONFIG_HASH,utc=time.time()))


def freeze():
    assert not (OUT/'FROZEN.json').exists()
    done=json.loads((OUT/'training_complete.json').read_text());assert done['runs']==9 and done['steps']==9000
    selection=read_csv(OUT/'selection.csv');assert len(selection)==18
    check_dependencies()
    dump(OUT/'FROZEN.json',dict(config_sha256=CONFIG_HASH,protocol_sha256=CFG['protocol_sha256'],selection=selection,weights={p.name:digest(p) for p in sorted((OUT/'weights').glob('*.pt'))},training=done,code={str(p.relative_to(ROOT)):digest(p) for p in Path(__file__).resolve().parent.glob('*.py')},utc=time.time()))
    print('FROZEN 9 runs / 18 selections before test access',flush=True)


def baseline_tables(split,method):
    # Read already audited pilot baselines with full standardization diagnostics.
    directory=ROOT/'results/conditional_innovation_pilot'
    key=lambda r:(r['trajectory'],r['start'],r['seed'])
    block=[r for r in read_csv(directory/'scores.csv') if r['split']==split and r['method']==method]
    detail=[r for r in read_csv(directory/'detail.csv') if r['split']==split and r['method']==method]
    return {key(r):r for r in block},{(*key(r),r['lead'],r['channel']):r for r in detail}


@torch.no_grad()
def evaluate():
    assert not (OUT/'evaluation_started.json').exists(),'Do not repeat completed exploratory evaluation'
    frozen=json.loads((OUT/'FROZEN.json').read_text());assert frozen['config_sha256']==CONFIG_HASH
    for name,sha in frozen['weights'].items():assert digest(OUT/'weights'/name)==sha
    for name,sha in frozen['code'].items():assert digest(ROOT/name)==sha
    check_dependencies()
    dump(OUT/'evaluation_started.json',dict(utc=time.time(),frozen_sha256=digest(OUT/'FROZEN.json')))
    norm=torch.load(ROOT/'reliability/checkpoints/current.pt',map_location='cuda',weights_only=False)['normalization']
    std=norm['std'].reshape(2,1,1);models={};selected={}
    for row in frozen['selection']:
        kind=row['kind'];seed=int(row['seed']);name=kind+'_'+row['selection']
        ck=torch.load(OUT/'weights'/row['file'],map_location='cuda',weights_only=False)
        m=make_model(kind,seed,std);m.load_state_dict(ck['state']);m.eval().requires_grad_(False)
        models[name,seed]=m;selected[name,seed]=int(row['step'])
    records=[];cells=[];audit=[];invariants=[]
    manifest=json.loads((PRIOR/'cache_manifest.json').read_text())
    from research.residual_dynamics_phase3_followup.run import shift_invariance
    operator,_=operators()
    for split,starts in [('test',CFG['starts']),('extension',CFG['extension'])]:
        baseline={method:baseline_tables(split,method) for method in ('native','phase3')}
        for start in starts:
            if split=='extension':
                entry=next(r for r in manifest if r['identity']['start']==start)
                file=PRIOR/'cache'/f'issued_{start}.pt';assert digest(file)==entry['sha256']
                bundle=torch.load(file,map_location='cuda',weights_only=False)
                history,paths=bundle['history'],bundle['paths']
                assert tensor_hash(history)==entry['identity']['history_sha256']
                assert tensor_hash(paths)==entry['paths_sha256']
                old={seed:q.transpose(1,2) for seed,q in bundle['biases'].items()}
            else:
                file=EXTRA/f'test_{start}_native.pt'
                entry=next(r for r in json.loads((ROOT/'results/residual_dynamics_phase2/alignment_audit.json').read_text())['entries'] if r['split']=='test' and r['start']==start)
                assert digest(file)==entry['native_sha256']
                bundle=torch.load(file,map_location='cuda',weights_only=False);paths=bundle['paths']
                history=raw(DATA,RECORDS['test'],start);old=None
            data=SimpleNamespace(p=paths.transpose(2,3),history=history,norm=norm,std=std,x=conditions(history,paths,norm),split=split,meta=[dict(trajectory=r['file'],Re=r['group'],start=start) for r in RECORDS['test']])
            zero=torch.zeros(len(paths),20,2,32,64,device='cuda')
            predictions={('native',0):zero,('zero',0):zero}
            predictions.update({key:predict(m,data.x) for key,m in models.items()})
            if old is None:old=old_heads(data)
            predictions.update({('phase3',seed):q for seed,q in old.items()})
            issued={name+':'+str(seed):tensor_hash(q) for (name,seed),q in predictions.items()}
            # Evidence of ordering is persisted before the target reader executes.
            with (OUT/'release_events.jsonl').open('a') as stream:stream.write(json.dumps(dict(event='issued',split=split,start=start,utc=time.time(),corrections=issued))+'\n')
            truth=raw(DATA,RECORDS['test'],start+20)
            if split=='extension':assert tensor_hash(truth)==entry['truth_sha256']
            data.y=truth.transpose(1,2);data.b=data.p.mean(1)-data.y
            with (OUT/'release_events.jsonl').open('a') as stream:stream.write(json.dumps(dict(event='truth_revealed',split=split,start=start,utc=time.time(),truth_sha256=tensor_hash(truth)))+'\n')
            audit.append(dict(split=split,start=start,cache=str(file),cache_sha256=digest(file),issued=issued,truth_sha256=tensor_hash(truth),frozen_sha256=digest(OUT/'FROZEN.json')))
            for (name,seed),q in predictions.items():
                alias=name in ('native','zero') or selected.get((name,seed),-1)==0
                ref='native' if alias else 'phase3' if name=='phase3' else None
                if ref:
                    blocks,detail=baseline[ref];lookupseed=seed if ref=='phase3' else 0
                    rr=[];cc=[]
                    for i,meta in enumerate(data.meta):
                        key=(meta['trajectory'],str(start),str(lookupseed))
                        r=dict(blocks[key]);r.update(method=name,seed=seed)
                        variance=(data.p[i]-q[i,None]).double().var(0,unbiased=False).mean((-1,-2));nv=variance/std.reshape(1,2).double().square()
                        r.update(spread=float(variance.mean()),sspread=float(nv.mean()));rr.append(r)
                        for lead in range(20):
                            for ch in range(2):
                                c=dict(detail[(*key,str(lead+1),'uv'[ch])]);c.update(method=name,seed=seed,spread=float(variance[lead,ch]),sspread=float(nv[lead,ch]));cc.append(c)
                else:rr,cc=full_scores(data,q,name,seed)
                records+=rr;cells+=cc
                if name not in ('native','zero','phase3'):
                    for i,meta in enumerate(data.meta):invariants.append(dict(split=split,start=start,method=name,seed=seed,trajectory=meta['trajectory'],**shift_invariance(paths[i],paths[i]-q[i].transpose(0,1)[None])))
            csvwrite(OUT/'scores.csv',records);csvwrite(OUT/'detail.csv',cells);csvwrite(OUT/'invariance.csv',invariants);dump(OUT/'evaluation_audit.json',audit)
            print('EVALUATED',split,start,'methods',len(predictions),flush=True)
    dump(OUT/'evaluation_complete.json',dict(blocks=264,rows=len(records),detail=len(cells),utc=time.time(),frozen_sha256=digest(OUT/'FROZEN.json')))

if __name__=='__main__':
    configure();OUT.mkdir(parents=True,exist_ok=True)
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['train','freeze','evaluate']);args=parser.parse_args()
    globals()[args.stage]()
