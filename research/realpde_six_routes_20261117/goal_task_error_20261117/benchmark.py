from task_stream import *

@torch.no_grad()
def run():
    setup();ck=load(OUT/'models.pt');names=list(dict.fromkeys(['base','task_persistence','task_affine','direct/task_stats/8','direct/X+P/16','direct/F_last/16',ck['chosen'],'direct/kinematic/16','direct/task_weighted/16','pod_q/X+P+F_last/16']))
    apis={k:TaskErrorPredictor(k) for k in names};d=load(SOURCE/'development.pt');n=st.norm('cpu');inputs=[d['x'][i:i+1]*n['std']+n['mean'] for i in [0,4,18,36,63,99]];rows=[]
    for api in apis.values():
        for h in inputs[:2]:api.predict(h)
    rng=np.random.default_rng(42)
    for repeat in range(5):
        for i,h in enumerate(inputs):
            for name in rng.permutation(names):
                torch.cuda.synchronize();start=time.perf_counter();out=apis[name].predict(h);out['task_estimates'].cpu();torch.cuda.synchronize();elapsed=(time.perf_counter()-start)*1000
                rows.append(dict(method=name,repeat=repeat,window=i,ms=elapsed))
    df=pd.DataFrame(rows);df.to_csv(OUT/'timing_samples.csv',index=False);df.groupby('method').ms.agg(['mean','median','min','max']).to_csv(OUT/'timing.csv')
    dump(OUT/'timing_protocol.json',dict(device=torch.cuda.get_device_name(),torch_version=torch.__version__,windows=6,repeats=5,seed=42,input='CPU float32 physical history',output='CPU task estimates',initialization_excluded=True,heads_and_joint_sampling=False,extra_four_call_backbone_for_deterministic_feature=True,selected=ck['chosen']))
    print(df.groupby('method').ms.median().to_string(),flush=True)

if __name__=='__main__':run()
