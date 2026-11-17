"""History-only task estimates with frozen empirical error readouts."""
from task_error import *

class TaskErrorPredictor:
    def __init__(self, method=None):
        ck=load(OUT/'models.pt','cuda');self.method=method or ck['chosen'];self.ck=ck['models'][self.method]
        self.net=st.model();self.norm=st.norm();self.tasks=Tasks();self.det=None
        kind=self.ck.get('kind','')
        if kind.startswith('F_det4_'):
            self.det=st.model();step=int(kind.split('_')[-1]);self.det.load_state_dict(load(SOURCE/'deterministic4'/f'model{step}.pt','cuda')['model'])
    @torch.no_grad()
    def predict(self, history, normalized=False):
        history=torch.as_tensor(history)
        if history.shape!=(1,2,20,32,64) or not torch.isfinite(history).all():raise ValueError('Expected finite history [1,2,20,32,64]')
        x=history.to(device='cuda',dtype=torch.float32)
        if not normalized:x=(x-self.norm['mean'])/self.norm['std']
        p,taps=st.capture(self.net,x);pp=(p*self.norm['std']+self.norm['mean']).double()
        desc=dict(X=st.pooled(x),P=st.pooled(p),F_last=taps['F_last'])
        if self.det is not None:
            _,dtaps=st.capture(self.det,x);desc[self.ck['kind']]=dtaps['F_last']
        if self.ck.get('kind') in ['kinematic','task_weighted']:
            xp=(x*self.norm['std']+self.norm['mean']).double();desc['kinematic'],desc['task_weighted']=self.tasks.descriptors(xp,pp)
        elif self.ck.get('kind')=='task_stats' or self.ck['style']=='task_persistence':
            xp=(x*self.norm['std']+self.norm['mean']).double();desc['kinematic']=self.tasks.values(xp)
        d=dict(jp=self.tasks.values(pp),features=desc)
        estimate=prediction(d,self.ck,self.tasks,pp)
        return dict(method=self.method,point=pp[0],baseline_tasks=d['jp'][0],task_estimates=estimate[0],signed_error_estimate=(estimate-d['jp'])[0])
