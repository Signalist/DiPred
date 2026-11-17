"""Target-free inference, timing, and explicitly separate free-running task."""
from local_learn import *
from common import records,read

class Predictor:
    def __init__(self,name='base'):
        self.name=name;self.model=model();self.cfg=None;self.det=None
        if name!='base':
            ck=load(FOLDER/f'{name}.pt','cuda');self.cfg=ck['summary'];self.proj=ck['projection']
            self.head=Head(self.cfg['style']).cuda().eval();self.head.load_state_dict(ck['model']);self.head.requires_grad_(False)
            if 'F_det4' in self.cfg['kind']:
                self.det=model();self.det.load_state_dict(load(OUT/'deterministic4/model6000.pt','cuda')['model'])
    @torch.no_grad()
    def predict(self,x):
        """Input and output standardized [B,2,20,32,64]. No target accepted."""
        if self.name=='base':return self.model.rollout(x,500000,sign=0)
        p,f=capture(self.model,x)
        if self.det is not None:_,df=capture(self.det,x);f['F_det4_6000']=df['F_last']
        d=dict(x=x,p=p,features=f)
        z=apply_projection(raw_features(d,self.cfg['kind']),self.proj)
        return self.head(z,p,self.cfg['shrink'])

def selected_names():
    sel=json.loads((FOLDER/'selection.json').read_text())['selected']
    return ['base',sel['warp'],sel['additive']]

@torch.no_grad()
def audit_timing():
    d=load(OUT/'development.pt');x=d['x'][:1].cuda();rows=[];checks={}
    # No simultaneous training is permitted during this benchmark.
    for name in selected_names():
        pred=Predictor(name);q=pred.predict(x)
        if name=='base':expected=d['p'][:1].cuda()
        else:
            z=apply_projection(raw_features(d,pred.cfg['kind'])[:1],pred.proj)
            expected=pred.head(z,d['p'][:1].cuda(),pred.cfg['shrink'])
        error=float((q-expected).abs().max());assert error<2e-5;checks[name]=error
        for _ in range(3):pred.predict(x)
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();times=[]
        for _ in range(30):
            begin=time.perf_counter();pred.predict(x);torch.cuda.synchronize();times.append((time.perf_counter()-begin)*1000)
        rows.append(dict(method=name,median_ms=float(np.median(times)),p10_ms=float(np.percentile(times,10)),p90_ms=float(np.percentile(times,90)),peak_allocated_mb=torch.cuda.max_memory_allocated()/2**20,repeats=30))
        del pred;torch.cuda.empty_cache()
    pd.DataFrame(rows).to_csv(OUT/'timing.csv',index=False)
    dump(OUT/'inference_audit.json',dict(passed=True,cached_vs_direct_max_error=checks,input='only observed standardized history',timing_scope='A100 batch1 preloaded models; includes full four-step backbone, hooks, pooling, PCA, correction; excludes disk IO and scoring. Peak allocation includes loaded audit tensors.'))
    print(rows,flush=True)

@torch.no_grad()
def rollout():
    rows=[]
    # All forecasts from frame20 to220; only first20 are observed in free mode.
    n=norm();recs=records('development')
    raw={}
    for r in recs:
        raw[r['file']]=torch.cat([read(r,s)[:,:20] for s in range(0,201,20)],1).cuda()[None]
    for name in selected_names():
        pred=Predictor(name)
        for r in recs:
            a=(raw[r['file']]-n['mean'])/n['std'];h=a[:,:,:20]
            for block in range(10):
                free=pred.predict(h)
                observed=pred.predict(a[:,:,block*20:(block+1)*20])
                # Targets are read for scores only after both forecasts exist.
                y=a[:,:,(block+1)*20:(block+2)*20]
                for mode,q in [('free',free),('observed',observed)]:
                    rows.append(dict(method=name,file=r['file'],group=r['group'],block=block+1,mode=mode,mse=float(mse(q,y)[0]),field_energy=float((q*n['std']+n['mean']).square().mean())))
                h=free
        del pred
    df=pd.DataFrame(rows);df.to_csv(OUT/'rollout_windows.csv',index=False)
    df.groupby(['method','mode','block']).mse.mean().reset_index().to_csv(OUT/'rollout_summary.csv',index=False)
    print(df.groupby(['method','mode']).mse.mean().to_string(),flush=True)

if __name__=='__main__':
    setup();a=argparse.ArgumentParser();a.add_argument('action',choices=['audit','rollout']);args=a.parse_args()
    audit_timing() if args.action=='audit' else rollout()
