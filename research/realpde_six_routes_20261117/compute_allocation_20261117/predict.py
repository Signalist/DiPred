"""Deployable synchronous routing: only available histories enter this API."""
from allocation import *

class BudgetedPredictor:
    def __init__(self):
        s.setup();s.OUT=FINAL;self.core=s.Predictor(method='ar__mixed')
        models=s.load(OUT/'models.pt');selection=json.loads((OUT/'selection.json').read_text())
        self.models={role:models[selection['chosen'][role]] for role in ('benefit','risk')}

    @torch.no_grad()
    def predict_batch(self,observed,quota=.5,policy='benefit',normalized=False,request_ids=None,sampling_seeds=None,expensive_action='smooth'):
        if observed.ndim!=5 or tuple(observed.shape[1:])!=(2,20,32,64):raise ValueError('Expected [requests,2,20,32,64]')
        if not torch.isfinite(observed).all():raise ValueError('Non-finite observations')
        if policy not in ('benefit','risk','width','random'):raise ValueError('Unknown policy')
        if expensive_action not in ('smooth','secc3'):raise ValueError('Unknown action')
        size=len(observed);ids=[str(i) for i in range(size)] if request_ids is None else list(request_ids)
        seeds=list(range(42,42+size)) if sampling_seeds is None else list(sampling_seeds)
        if len(ids)!=size or len(set(ids))!=size or len(seeds)!=size:raise ValueError('Invalid request ids/seeds')
        n=self.core.n;x=observed.cuda();x=x if normalized else (x-n['mean'])/n['std'];issued=[];feats={};widths=[]
        for i in range(size):
            p,q,f=self.core.issue(x[i:i+1]);issued.append((p,q,f));widths.append(width(q,n))
            for k,v in f.items():feats.setdefault(k,[]).append(v.cpu())
        observable=dict(features={k:torch.cat(v) for k,v in feats.items()})
        if policy in self.models:priority=predict(observable,self.models[policy])
        elif policy=='width':priority=np.asarray(widths)
        else:priority=random_priority(ids,seeds)
        mask=choose([dict(file=k,start=0) for k in ids],priority,quota);samples=[]
        for i,(p,q,f) in enumerate(issued):
            if mask[i]:
                native=torch.stack([self.core.pred.rollout(x[i:i+1],500000+k,sign=1)[0] for k in range(32)])
                result=s.baselines(q,native,expensive_action)
            else:result=s.couple(q,self.core.generator.sample('ar__mixed',f,seeds[i]))
            samples.append(result if normalized else result*n['std']+n['mean'])
        return samples,dict(expensive=mask,priority=priority,quota=float(mask.mean()),request_ids=ids,extra_paths=int(mask.sum()*32))
