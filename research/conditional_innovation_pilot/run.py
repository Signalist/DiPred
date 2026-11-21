"""Bounded fitting on train/development, then one frozen exploratory evaluation."""
import argparse
import time

from core import *


def audit_data(data, operator):
    source = encode_source(data.b.double(), operator)
    rebuilt = reconstruct(source, operator)
    assert float((rebuilt-data.b.double()).abs().max()) < CFG["numeric"]["reconstruction_double_atol"]
    rows = []
    for lead in range(20):
        for channel, name in enumerate("uv"):
            value = source[:, lead, channel]
            rows.append(dict(split=data.split, target="b1" if lead==0 else "r", lead=lead+1,
                channel=name, mean=float(value.mean()), rms=float(value.square().mean().sqrt()),
                normalized_rms=float((value/data.std[channel]).square().mean().sqrt()),
                samples=len(value), finite=bool(torch.isfinite(value).all())))
    return rows


@torch.no_grad()
def stage_a():
    assert not (OUT/"selection_A.json").exists(), "stage A already completed; use saved selection"
    record_environment()
    operator, diagonal = operators()
    train, dev = load_data("train"), load_data("development")
    csvwrite(OUT/"data_audit.csv", audit_data(train,operator)+audit_data(dev,operator))
    mean_bias = train.b.double().mean(0,keepdim=True)
    mean_source = encode_source(train.b.double(),operator).mean(0,keepdim=True)
    assert float((reconstruct(mean_source,operator)-mean_bias).abs().max()) < 1e-11
    weights=OUT/"weights"
    weights.mkdir(exist_ok=True)
    baseline=dict(bias_mean=mean_bias,source_mean=mean_source,std=train.std)
    dump_checkpoint(weights/"baseline.pt",baseline)
    fits=fit_ridges(train,operator)
    native=select_score(dev,torch.zeros_like(dev.b))
    predictions={"native":torch.zeros_like(dev.b),"R0":mean_bias.float().expand_as(dev.b)}
    candidates=[]
    selected={}
    for kind in ("R1","D1"):
        choices=[]
        for reg in CFG["ridge"]:
            model=fits[kind,reg]
            q=model(dev.x)
            values=select_score(dev,q)
            file=weights/f"{kind}_{reg}.pt"
            checkpoint_hash=dump_checkpoint(file,model.checkpoint)
            candidates.append(dict(kind=kind,ridge=reg,**values,sha256=checkpoint_hash,
                                   admissible=admissible(values,native)))
            choices.append((values["crps"],reg,q))
        _,reg,q=min(choices,key=lambda x:x[0])
        predictions[kind+"_ridge"]=q
        # R0 is nested and zero is always an admissible deployment candidate.
        deployed="native"
        best=native
        for name,value in [("R0",predictions["R0"]),(kind+"_ridge",q)]:
            stats=select_score(dev,value)
            if admissible(stats,native) and stats["crps"]<best["crps"]:
                deployed,best=name,stats
        selected[kind]=dict(ridge=reg,deployment=deployed,development=best)
        predictions[kind]=predictions[deployed]
    records,cells=[],[]
    for name,q in predictions.items():
        a,b=score(dev,q,name,operator)
        records+=a;cells+=b
    # Gate uses the fixed nonzero R1 chosen by whole-block development CRPS.
    r0=[row for row in records if row["method"]=="R0"]
    r1=[row for row in records if row["method"]=="R1_ridge"]
    gain=1-np.mean([r["r_smse"] for r in r1])/np.mean([r["r_smse"] for r in r0])
    def group_gains(field,values,source):
        result={}
        for value in values:
            means={}
            for method in ("R0","R1_ridge"):
                means[method]=np.mean([r["r_smse"] for r in source if r["method"]==method and r[field]==value and r["r_smse"]!=""])
            result[str(value)]=1-means["R1_ridge"]/means["R0"]
        return result
    windows=group_gains("start",CFG["starts"],records)
    channels=group_gains("channel",["u","v"],cells)
    reynolds=group_gains("Re",sorted({r["group"] for r in RECORDS["development"]}),records)
    gate=bool(gain>=CFG["source_gate"]["relative_gain"] and min(windows.values())>0
              and min(channels.values())>=0 and sum(v>0 for v in reynolds.values())>=2)
    selection=dict(selected=selected,gate=gate,r_gain=gain,window_gains=windows,channel_gains=channels,
                   Re_gains=reynolds,config_sha256=CONFIG_HASH,neural_runs_allowed=9 if gate else 0)
    dump(OUT/"selection_A.json",selection)
    csvwrite(OUT/"ridge_candidates.csv",candidates)
    csvwrite(OUT/"development_scores_A.csv",records)
    csvwrite(OUT/"development_detail_A.csv",cells)
    print("stage A selection",json.dumps(selection,indent=2),flush=True)


def stage_b():
    record_environment()
    selection=json.loads((OUT/"selection_A.json").read_text())
    if not selection["gate"]:
        dump(OUT/"stage_B.json",dict(executed=False,reason="prespecified conditional source gate not met",
                                    neural_runs=0,config_sha256=CONFIG_HASH))
        print("stage B skipped by fixed development gate",flush=True)
        return
    assert not (OUT/"stage_B.json").exists(), "stage B already completed"
    operator,diagonal=operators()
    train,dev=load_data("train"),load_data("development")
    with torch.no_grad():
        native=select_score(dev,torch.zeros_like(dev.b))
    started=time.monotonic()
    training=[];selections=[];rows=[];cells=[]
    for kind in ("D","T_local","T_diag"):
        for seed in CFG["seeds"]:
            assert time.monotonic()-started<CFG["budget"]["max_seconds"],"fixed resource limit exhausted"
            torch.manual_seed(seed)
            model=SmallModel(kind,operator,diagonal,train.std).cuda()
            optimizer=torch.optim.Adam(model.parameters(),lr=CFG["lr"])
            generator=torch.Generator(device="cuda").manual_seed(CFG["sampling_seed"])
            best=native.copy();beststep=0
            beststate={key:value.detach().cpu().clone() for key,value in model.state_dict().items()}
            for step in range(1,CFG["steps"]+1):
                index=torch.randint(len(train.x),(CFG["batch"],),device="cuda",generator=generator)
                q=model(train.x[index])
                loss=((train.p[index]-train.y[index,None]-q[:,None])/train.std).abs().mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                if step%CFG["eval_every"]==0:
                    with torch.no_grad():
                        values=select_score(dev,predict(model,dev.x))
                    training.append(dict(kind=kind,seed=seed,step=step,**values,
                                         admissible=admissible(values,native)))
                    if admissible(values,native) and values["crps"]<best["crps"]:
                        best=values;beststep=step
                        beststate={key:value.detach().cpu().clone() for key,value in model.state_dict().items()}
                    print("train",kind,seed,step,values,"selected",beststep,flush=True)
            file=OUT/"weights"/f"{kind}_{seed}.pt"
            torch.save(dict(selected=beststate,step=beststep,kind=kind,seed=seed,config_sha256=CONFIG_HASH),file)
            model.load_state_dict(beststate)
            model.eval().requires_grad_(False)
            with torch.no_grad():
                q=predict(model,dev.x)
                a,b=score(dev,q,kind,operator,seed)
            rows+=a;cells+=b
            torch.cuda.synchronize()
            before=time.perf_counter()
            with torch.no_grad():predict(model,dev.x)
            torch.cuda.synchronize()
            selections.append(dict(kind=kind,seed=seed,step=beststep,params=sum(v.numel() for v in model.parameters()),
                seconds_per_block=(time.perf_counter()-before)/len(dev.x),sha256=digest(file),**best))
            csvwrite(OUT/"training_B.csv",training)
            csvwrite(OUT/"selection_B.csv",selections)
    csvwrite(OUT/"development_scores_B.csv",rows)
    csvwrite(OUT/"development_detail_B.csv",cells)
    dump(OUT/"stage_B.json",dict(executed=True,neural_runs=9,seconds=time.monotonic()-started,
                               config_sha256=CONFIG_HASH))


def freeze():
    assert not (OUT/"FROZEN.json").exists(),"already frozen"
    selection=json.loads((OUT/"selection_A.json").read_text())
    status=json.loads((OUT/"stage_B.json").read_text())
    weights={p.name:digest(p) for p in sorted((OUT/"weights").glob("*.pt"))}
    dump(OUT/"FROZEN.json",dict(config_sha256=CONFIG_HASH,stage_A=selection,stage_B=status,weights=weights,
         frozen_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())))
    print("frozen before test",len(weights),"weight files",flush=True)


def final_models(std):
    frozen=json.loads((OUT/"FROZEN.json").read_text())
    assert frozen["config_sha256"]==CONFIG_HASH
    for name,sha in frozen["weights"].items():assert digest(OUT/"weights"/name)==sha
    operator,diagonal=operators()
    base=torch.load(OUT/"weights/baseline.pt",map_location="cuda",weights_only=False)
    models={}
    for kind,spec in frozen["stage_A"]["selected"].items():
        ck=torch.load(OUT/"weights"/f"{kind}_{spec['ridge']}.pt",map_location="cuda",weights_only=False)
        models[kind+"_ridge",0]=Ridge(ck,operator)
    if frozen["stage_B"]["executed"]:
        for kind in ("D","T_local","T_diag"):
            for seed in CFG["seeds"]:
                ck=torch.load(OUT/"weights"/f"{kind}_{seed}.pt",map_location="cuda",weights_only=False)
                model=SmallModel(kind,operator,diagonal,std).cuda().eval().requires_grad_(False)
                model.load_state_dict(ck["selected"])
                models[kind,seed]=model
    return frozen,base,models


def baseline_tables(split,method):
    directory=PRIOR if split=="extension" else ROOT/"results/residual_dynamics_phase3"
    block=read_csv(directory/"scores.csv")
    detail=read_csv(directory/"lead_channel.csv")
    block=[r for r in block if r["method"]==method and (split=="extension" or r["split"]==split)]
    detail=[r for r in detail if r["method"]==method and (split=="extension" or r["split"]==split)]
    key=lambda r:(r["trajectory"],r["start"],r["seed"])
    return {key(r):r for r in block},{(*key(r),r["lead"],r.get("channel",r.get("ch"))):r for r in detail}


@torch.no_grad()
def evaluate():
    record_environment()
    assert not (OUT/"evaluation_complete.json").exists(),"frozen evaluation already completed"
    norm=torch.load(ROOT/"reliability/checkpoints/current.pt",map_location="cuda",weights_only=False)["normalization"]
    std=norm["std"].reshape(2,1,1)
    frozen,base,models=final_models(std)
    operator,_=operators()
    rows=[];cells=[];audit=[];invariants=[]
    manifest=json.loads((PRIOR/"cache_manifest.json").read_text())
    from research.residual_dynamics_phase3_followup.run import shift_invariance
    for split,starts in [("test",CFG["starts"]),("extension",CFG["extension"])]:
        prior={method:baseline_tables(split,method) for method in ("native","crps")}
        for start in starts:
            if split=="extension":
                entry=next(r for r in manifest if r["identity"]["start"]==start)
                file=PRIOR/"cache"/f"issued_{start}.pt"
                assert digest(file)==entry["sha256"]
                bundle=torch.load(file,map_location="cuda",weights_only=False)
                history,paths=bundle["history"],bundle["paths"]
                assert tensor_hash(history)==entry["identity"]["history_sha256"]
                assert tensor_hash(paths)==entry["paths_sha256"]
                frozen_bias={seed:value.transpose(1,2) for seed,value in bundle["biases"].items()}
            else:
                file=EXTRA/f"test_{start}_native.pt"
                entry=next(r for r in json.loads((ROOT/"results/residual_dynamics_phase2/alignment_audit.json").read_text())["entries"]
                           if r["split"]=="test" and r["start"]==start)
                assert digest(file)==entry["native_sha256"]
                bundle=torch.load(file,map_location="cuda",weights_only=False)
                paths=bundle["paths"]
                history=raw(DATA,RECORDS["test"],start)
                frozen_bias=None
            data=SimpleNamespace(p=paths.transpose(2,3),history=history,norm=norm,std=std,
                x=conditions(history,paths,norm),split=split,
                meta=[dict(trajectory=r["file"],Re=r["group"],start=start) for r in RECORDS["test"]])
            q0=torch.zeros(len(paths),20,2,32,64,device="cuda")
            predictions={("native",0):q0,("zero",0):q0,("R0",0):base["bias_mean"].float().expand_as(q0)}
            predictions.update({key:predict(model,data.x) for key,model in models.items()})
            for kind,spec in frozen["stage_A"]["selected"].items():
                predictions[kind,0]=predictions[spec["deployment"],0]
            if frozen_bias is None:frozen_bias=old_heads(data)
            predictions.update({("phase3",seed):q for seed,q in frozen_bias.items()})
            # Complete all corrections before opening the current target.
            issued={f"{key[0]}:{key[1]}":tensor_hash(q) for key,q in predictions.items()}
            truth=raw(DATA,RECORDS["test"],start+20)
            if split=="extension":assert tensor_hash(truth)==entry["truth_sha256"]
            data.y=truth.transpose(1,2)
            data.b=data.p.mean(1)-data.y
            audit.append(dict(split=split,start=start,cache=str(file),cache_sha256=digest(file),
                              issued_biases=issued,truth_sha256=tensor_hash(truth),
                              frozen_selection_sha256=digest(OUT/"FROZEN.json")))
            for (name,seed),q in predictions.items():
                baseline="native" if name in ("native","zero") else "crps" if name=="phase3" else None
                existing,existing_detail=prior[baseline] if baseline else (None,None)
                a,b=score(data,q,name,operator,seed,existing,existing_detail)
                rows+=a;cells+=b
                if name not in ("native","zero","phase3"):
                    for index,r in enumerate(RECORDS["test"]):
                        invariants.append(dict(split=split,start=start,method=name,seed=seed,trajectory=r["file"],
                            **shift_invariance(paths[index],paths[index]-q[index].transpose(0,1)[None])))
            csvwrite(OUT/"scores.csv",rows)
            csvwrite(OUT/"detail.csv",cells)
            csvwrite(OUT/"invariance.csv",invariants)
            dump(OUT/"evaluation_audit.json",audit)
            print("evaluated",split,start,"methods",len(predictions),flush=True)
    dump(OUT/"evaluation_complete.json",dict(blocks=264,rows=len(rows),detail=len(cells),
         config_sha256=CONFIG_HASH,frozen_sha256=digest(OUT/"FROZEN.json")))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("stage",choices=["stage_a","stage_b","freeze","evaluate"])
    args=parser.parse_args()
    globals()[args.stage]()


if __name__=="__main__":
    main()
