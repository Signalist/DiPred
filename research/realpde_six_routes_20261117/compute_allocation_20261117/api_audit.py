"""CPU physical/normalized input audit of the two public inference APIs."""
from predict import *
import importlib.util

@torch.no_grad()
def main():
    d=s.load(s.SOURCE/'development.pt');pred=BudgetedPredictor();n={k:v.cpu() for k,v in pred.core.n.items()};physical=d['x'][:2]*n['std']+n['mean'];normalized=(physical-n['mean'])/n['std'];names=['a','b'];seeds=[42,43]
    a,ia=pred.predict_batch(physical,.5,request_ids=names,sampling_seeds=seeds)
    b,ib=pred.predict_batch(normalized,.5,normalized=True,request_ids=names,sampling_seeds=seeds)
    budget_error=max(float((u-(v*pred.core.n['std']+pred.core.n['mean'])).abs().max()) for u,v in zip(a,b));assert budget_error<1e-6 and np.array_equal(ia['expensive'],ib['expensive'])
    spec=importlib.util.spec_from_file_location('lowcost_public_api',W/'lowcost_joint_20261117/predict.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);cheap=m.LowCostJointPredictor()
    a=cheap.predict(physical[:1]);b=cheap.predict(normalized[:1],normalized=True);cheap_error=float((a-(b*cheap.core.n['std']+cheap.core.n['mean'])).abs().max());assert cheap_error<1e-6
    cached=cheap.predict(d['x'][:1],normalized=True,sampling_seed=420000);expected=s.load(FINAL/'development/audit_0000.pt')['samples']['ar__mixed'].cuda();replay=float((cached-expected).abs().max());assert replay<2e-5
    s.dump(OUT/'public_api_audit.json',dict(passed=True,cpu_physical_and_normalized_inputs=True,budgeted_physical_conversion_max_error=budget_error,cheap_physical_conversion_max_error=cheap_error,compact_cached_field_replay_max_error=replay,quotas_match=True))
    print('PUBLIC_API_AUDIT',budget_error,cheap_error,replay,flush=True)

if __name__=='__main__':main()
