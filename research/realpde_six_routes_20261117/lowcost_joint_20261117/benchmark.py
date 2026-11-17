"""Real end-to-end latency and target-free replay of selected joint models."""
from study import *
import gc

@torch.no_grad()
def main():
    setup();selection=json.loads((OUT/'selection.json').read_text());selected=selection['selected'];d=load(SOURCE/'development.pt');x=d['x'][:1].cuda()
    audit=load(OUT/'development/audit_0000.pt');rows=[];checks={}
    names=list(dict.fromkeys(['ecc32','smooth',selection['secc'],selection['necc'],selection['necc_rank'],selected,'pca__global','F_last__schaake']))
    for method in names:
        pred=Predictor(library='__' in method,method=method)
        z=pred.predict(x,method,seed=420000)
        error=float((z.cpu()-audit['samples'][method]).abs().max());assert error<2e-5,(method,error);checks[method]=error
        for _ in range(3):pred.predict(x,method)
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();times=[]
        resident=torch.cuda.memory_allocated()/2**20
        for _ in range(15):
            start=time.perf_counter();pred.predict(x,method);torch.cuda.synchronize();times.append(1000*(time.perf_counter()-start))
        rows.append(dict(method=method,median_ms=float(np.median(times)),p10_ms=float(np.quantile(times,.1)),p90_ms=float(np.quantile(times,.9)),resident_mb=resident,peak_allocated_mb=torch.cuda.max_memory_allocated()/2**20,repeats=15,extra_stochastic_backbone_paths=0 if '__' in method else 32))
        del pred,z;gc.collect();torch.cuda.empty_cache()
    pd.DataFrame(rows).to_csv(OUT/'timing.csv',index=False)
    dump(OUT/'inference_audit.json',dict(passed=True,replay_max_error=checks,scope='observed history only -> four-step P/F ->49 XPF heads ->sampling/coupling; preloaded models; excludes IO and scoring',gpu=torch.cuda.get_device_name(),tf32_override=os.environ.get('TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'),benchmarks=rows))
    print(pd.DataFrame(rows).to_string(index=False),flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--conditional-scores',action='store_true');a.add_argument('--ar',action='store_true');a.add_argument('--scorefit',action='store_true');args=a.parse_args()
    if args.conditional_scores or args.ar or args.scorefit:
        import study
        OUT=OUT/'conditional_scores';study.OUT=OUT
        if args.ar:OUT=OUT/'ar';study.OUT=OUT;study.AR_STAGE=True
        if args.scorefit:OUT=OUT/'scorefit';study.OUT=OUT;study.SCOREFIT_STAGE=True
    main()
