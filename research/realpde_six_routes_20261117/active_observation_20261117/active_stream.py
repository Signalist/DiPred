"""Actual masked-sensor interface; no future field is accepted by update()."""
from observations import *
DATA_OUT=W/'runs/active_observation_20261117'

class SparseUpdatePredictor:
    def __init__(self,method='voi',budget=8,assimilator=None,planner=None):
        if method not in ('uniform','spacefill','random','variance','risk','voi','voi_average_scale','critic') or budget not in BUDGETS:raise ValueError('Unsupported policy or site budget')
        if method=='critic' and planner is None:raise ValueError('A fitted benefit planner is required')
        self.planner=planner
        setup();self.method=method;self.budget=budget;self.core=joint.Predictor(library=False);self.n=self.core.n;self.floor=load(DATA_OUT/'scale_floor.pt','cuda').float()[:,None,None,None];self.assim=Assimilator(json.loads((DATA_OUT/'selection.json').read_text())['config']) if assimilator is None else assimilator;self.pending=None;self.fixed_order=self.assim.greedy(self.assim.pack(self.assim.model['average_scale']))[0];self.spacefill=spacefill()
    @torch.no_grad()
    def predict(self,observed_history,start=0,trajectory='live',normalized=False):
        if self.pending is not None:raise RuntimeError('Consume the issued sparse observation request first')
        if observed_history.shape!=(1,2,20,32,64) or not torch.isfinite(observed_history).all():raise ValueError('Expected full finite20-frame history')
        x=observed_history.cuda();x=x if normalized else (x-self.n['mean'])/self.n['std'];p,q,desc=self.core.issue(x);scale=torch.maximum(q.std(0,unbiased=False)*self.n['std'][0],self.floor);risk=observation_rows((q.var(0,unbiased=False)+(q.mean(0)-p).square())*self.n['std'][0].square());pack=self.assim.pack(scale)
        if self.method=='critic':indices=self.planner(self.assim,pack,scale,risk,desc,self.budget)
        elif self.method=='uniform':indices=uniform(self.budget)
        elif self.method=='spacefill':indices=self.spacefill[:self.budget]
        elif self.method=='random':indices=random_orders(dict(file=trajectory,start=start))[0,:self.budget]
        elif self.method=='voi':indices=self.assim.greedy(pack,self.budget)[0]
        elif self.method=='voi_average_scale':indices=self.fixed_order[:self.budget]
        else:
            score=(observation_rows(scale.square()) if self.method=='variance' else risk).sum(-1);indices=torch.argsort(score,descending=True,stable=True)[:self.budget].cpu().numpy()
        self.assim.pending=dict(p=p.detach().clone(),pack=pack,indices=indices.copy(),start=start);self.pending=dict(p=p,scale=scale,indices=indices.copy(),start=start)
        return dict(prediction=p*self.n['std'][0]+self.n['mean'][0],site_indices=indices.copy(),sites_yx=SITES[indices].copy(),observation_time=start+25,reissued_target_array_indices=list(range(5,20)))
    @torch.no_grad()
    def update(self,observed_uv,at):
        result=self.assim.observe(observed_uv,at);self.pending=None;return result

if __name__=='__main__':
    print('SparseUpdatePredictor.predict(full_history) -> requested sites; update(selected_uv, start+25) -> physical future15 field. No unselected future values accepted.')
