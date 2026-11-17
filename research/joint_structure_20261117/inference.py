"""Standalone frozen joint inference: accepts observed history, no future labels."""
from physics import *
from followcommon import base_features
from helmholtz_filter import transform as component_transform

class JointPredictor:
    def __init__(self):
        setup();self.n=norm();self.pred=model();self.heads,self.batches=models_for('XPF')
        self.stats=load(SHARED,'cuda')['statistics'];self.hodge=Hodge()

    def _features(self,x,p,f):
        raw=base_features(x*self.n['std']+self.n['mean'],p*self.n['std']+self.n['mean'],self.n['mean'],self.n['std'],torch.ones(len(x),3,device=x.device))
        base=((raw-self.stats['base_mean'])/self.stats['base_std']).clamp(-10,10)
        last=((f.half().float()-self.stats['last_mean'])/self.stats['last_std']).clamp(-10,10)
        return base,last

    @torch.no_grad()
    def predict_normalized(self,x,method='smooth',sampling_seed=500000):
        """x [1,2,20,32,64]; output [32,2,20,32,64], standardized units."""
        if x.shape!=(1,2,20,32,64):raise ValueError('Expected one complete [1,2,20,32,64] observed history')
        if method not in ('ecc32','smooth','sol'):raise ValueError(method)
        if not torch.isfinite(x).all():raise ValueError('Non-finite history')
        p,f=feature(self.pred,x,'diffusion');base,last=self._features(x,p,f)
        values=quantile_values(all_bounds(self.batches,base,last),p[0])
        native=torch.stack([self.pred.rollout(x,sampling_seed+k,sign=1)[0] for k in range(N)])
        z=couple(values,native)
        if method!='ecc32':z=component_transform(values,z,self.n,method,.5,1,self.hodge,p[0])
        return z

    @torch.no_grad()
    def predict(self,history,method='smooth',sampling_seed=500000):
        """Physical u/v history [1,2,20,32,64] -> 32 physical future fields."""
        x=(history-self.n['mean'])/self.n['std']
        return self.predict_normalized(x,method,sampling_seed)*self.n['std']+self.n['mean']

@torch.no_grad()
def benchmark():
    # Loading cached observations here is only a replay/benchmark harness.
    # JointPredictor itself never reads any dataset or evaluation cache.
    joint=JointPredictor();d=cache('development');x=d['x'][:1];audit=load(BASE/'physics/development/audit_0000.pt','cuda')
    rows=[];checks={}
    for method,key in [('ecc32','ecc32'),('smooth','smooth_ecc_b50_i1'),('sol','sol_ecc_b50_i1')]:
        result=joint.predict_normalized(x,method)
        reference=audit['samples'][key] if key in audit['samples'] else load(BASE/'helmholtz_filter/development/audit_0000.pt','cuda')['samples'][key]
        error=float((result-reference).abs().max());assert error<2e-5,(method,error);checks[method]=error
        for _ in range(2):joint.predict_normalized(x,method)
        times=[]
        for _ in range(10):
            torch.cuda.synchronize();start=time.perf_counter();joint.predict_normalized(x,method);torch.cuda.synchronize();times.append((time.perf_counter()-start)*1000)
        rows.append(dict(method=method,median_ms=float(np.median(times)),p10_ms=float(np.quantile(times,.1)),p90_ms=float(np.quantile(times,.9)),repeats=10,paths=32,training_seed=42))
    dump(BASE/'inference_audit.json',dict(replay_max_error=checks,scope='input -> four-step P -> 49 frozen XPF heads -> 32 four-step paths -> joint transform; all weights preloaded, no disk I/O or scores in timed region',gpu=torch.cuda.get_device_name(),tf32_override=os.environ.get('TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'),benchmarks=rows))
    print(json.dumps(rows,indent=2),flush=True)

if __name__=='__main__':benchmark()
