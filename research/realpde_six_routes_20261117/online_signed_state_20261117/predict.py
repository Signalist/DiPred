"""Observed-input online predictor with explicit, delayed full-block reveal."""
from state import *

class OnlinePredictor:
    def __init__(self,method=None):
        setup();self.model=st.model();self.fit=load(OUT/'basis.pt');self.models=load(OUT/'models.pt');selection=json.loads((OUT/'selection.json').read_text());self.method=selection['chosen'] if method is None else method;self.state=State(selection['models'][self.method]['config'],self.models);self.basis=self.fit['basis'].cuda();self.n=norm();self.pending_raw=None
    @torch.no_grad()
    def predict(self,observed,start,normalized=False):
        if observed.shape!=(1,2,20,32,64) or not torch.isfinite(observed).all():raise ValueError('Expected finite observed [1,2,20,32,64]')
        x=observed.cuda();x=x if normalized else (x-self.n['mean'])/self.n['std'];p,f=st.capture(self.model,x);f.update(X=st.pooled(x),P=st.pooled(p));raw=torch.cat([f[k] for k in ('X','P','F_last')],1).cpu().double();desc=project(raw,self.fit['projection'])[0].numpy();c=self.state.issue(desc,int(start));self.pending_raw=p.detach().clone();delta=(torch.tensor(c,device='cuda')@self.basis).reshape_as(p).float();out=p+delta/self.n['std']
        return out if normalized else out*self.n['std']+self.n['mean']
    @torch.no_grad()
    def reveal(self,observed_future,reveal_time,normalized=False):
        if self.pending_raw is None:raise RuntimeError('No pending raw forecast')
        if observed_future.shape!=self.pending_raw.shape or not torch.isfinite(observed_future).all():raise ValueError('Expected full20 feedback')
        y=observed_future.cuda();y=y if normalized else (y-self.n['mean'])/self.n['std'];r=((y-self.pending_raw)*self.n['std']).flatten().double();c=(r@self.basis.T/D).cpu().numpy();self.state.reveal(c,int(reveal_time));self.pending_raw=None
    def skip_feedback(self):self.state.skip_feedback();self.pending_raw=None

@torch.no_grad()
def rollout():
    from common import records,read
    rows=[];n=norm();recs=records('development')
    for method in ('base','rolling_bias','rw_kalman','conditional_kalman','linear_ar','rnn'):
        pred=OnlinePredictor(method)
        for rec in recs:
            a=torch.cat([read(rec,s)[:,:20] for s in range(0,281,20)],1).cuda()[None];a=(a-n['mean'])/n['std']
            for mode,warm in [('cold_free',0),('warm_free',3)]:
                selection=json.loads((OUT/'selection.json').read_text());pred.state=State(selection['models'][method]['config'],pred.models);pred.pending_raw=None
                for j in range(warm):
                    pred.predict(a[:,:,20*j:20*(j+1)],20*j,True);pred.reveal(a[:,:,20*(j+1):20*(j+2)],20*j+40,True)
                h=a[:,:,20*warm:20*(warm+1)]
                for k in range(10):
                    start=20*(warm+k);q=pred.predict(h,start,True);pred.skip_feedback();y=a[:,:,start+20:start+40]
                    rows.append(dict(method=method,file=rec['file'],group=rec['group'],mode=mode,block=k+1,mse=float(st.mse(q,y)[0]),field_energy=float((q*n['std']+n['mean']).square().mean()),real_reveals=warm));h=q
        print('ROLLOUT',method,flush=True);del pred;torch.cuda.empty_cache()
    df=pd.DataFrame(rows);df.to_csv(OUT/'rollout_windows.csv',index=False);df.groupby(['method','mode','block']).mse.mean().reset_index().to_csv(OUT/'rollout_summary.csv',index=False)

if __name__=='__main__':rollout()
