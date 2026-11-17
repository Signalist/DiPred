from controls import *

@torch.no_grad()
def run(initial=False):
    setup();src=load(SOURCE/'development.pt');n=st.norm('cpu');inputs=[src['x'][i:i+1]*n['std']+n['mean'] for i in [0,4,18,36,63,99]];names=REFERENCES+KINDS;apis={name:Predictor(name,initial=initial) for name in names};rows=[];packs=[]
    # Same frozen P/F, created from each actual input, for correction-only timing.
    for h in inputs:
        x=(h.cuda()-st.norm()['mean'])/st.norm()['std'];p,f=st.capture(apis['base'].net,x);packs.append((x,p,f))
    for api in apis.values():
        for h in inputs[:2]:api.predict(h)
    rng=np.random.default_rng(42)
    for repeat in range(5):
        for i,h in enumerate(inputs):
            for name in rng.permutation(names):
                api=apis[name];torch.cuda.synchronize();begin=time.perf_counter();out=api.predict(h);out['physical'].cpu();torch.cuda.synchronize();full=(time.perf_counter()-begin)*1000
                x,p,f=packs[i];torch.cuda.synchronize();begin=time.perf_counter();api.correct(x,p,f,force_gain=1.);torch.cuda.synchronize();post=(time.perf_counter()-begin)*1000
                rows.append(dict(method=name,repeat=repeat,window=i,full_ms=full,active_post_ms=post,parameters=0 if api.head is None else sum(p.numel() for p in api.head.parameters())))
    label='initial_' if initial else '';df=pd.DataFrame(rows);df.to_csv(OUT/f'{label}timing_samples.csv',index=False);sm=df.groupby('method')[['full_ms','active_post_ms','parameters']].median();sm.to_csv(OUT/f'{label}timing.csv')
    # Isolated live tensor allocation per operational predictor, including its weights/input/output.
    del apis,packs,src,out;torch.cuda.empty_cache();memory=[]
    for name in names:
        api=Predictor(name,initial=initial);api.predict(inputs[0]);torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();api.predict(inputs[0]);torch.cuda.synchronize();memory.append(dict(method=name,peak_allocated_mb=torch.cuda.max_memory_allocated()/2**20));del api;torch.cuda.empty_cache()
    pd.DataFrame(memory).to_csv(OUT/f'{label}memory.csv',index=False)
    dump(OUT/f'{label}timing_protocol.json',dict(device=torch.cuda.get_device_name(),windows=6,repeats=5,input='CPU float32 physical history',output='CPU physical future20',full='selected gain; original backbone plus all correction operations',post='active gain1, pooling/PCA included, original point and F already issued',phase='before fit' if initial else 'after frozen calibration',seed=42))
    if not initial:
        sel=json.loads((OUT/'selection.json').read_text());ref=sm.loc['warp_F_last'];valid=[k for k in KINDS if sel['models'][k]['gain']>0 and sm.loc[k,'full_ms']<=1.1*ref.full_ms and sm.loc[k,'active_post_ms']<=1.1*ref.active_post_ms];chosen=min(valid,key=lambda k:sel['models'][k]['calibration_mse']) if valid else None
        dump(OUT/'budget_selection.json',dict(eligible=valid,selected=chosen,rule='positive selected gain; full and active-post median<=110% frozen warp; choose calibration MSE',warp_full_ms=float(ref.full_ms),warp_post_ms=float(ref.active_post_ms)))
    print(sm.to_string(),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--initial',action='store_true');args=ap.parse_args();run(args.initial)
