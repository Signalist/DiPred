"""Full 32-path numerical audit for fixed-seed batched sampling."""
import json,sys,torch
from clean_reliability_common import ROOT,configure,manifest
from experiment_reliability_transport_e12 import geometry,history,issued
from batched_noise import sample_batched
sys.path.insert(0,str(ROOT/'github/DiPred'))
from reliability.selected16_predict import RequestedThreePredictor
from train49 import OUT

configure();rec=manifest()['files']['development']
p=RequestedThreePredictor(geometry(rec),'selected16','cuda')
h=(history('development',0)-p.norm['mean'])/p.norm['std']
with torch.no_grad():
    seeds=list(range(500000,500032))
    serial=torch.stack([p.model.rollout(h,s,sign=1) for s in seeds])
    batch=torch.cat([sample_batched(p.model,h,seeds[i:i+2]) for i in range(0,32,2)])
    diff=(serial-batch).abs()
    old=issued('diffusion','development',0)
    physical=batch*p.norm['std']+p.norm['mean']
    meanerr=float((physical.mean(0)-old['mean']).abs().max())
    stderr=float((physical.std(0)-old['std32']).abs().max())
    print('max sample diff',float(diff.max()),'meanerr',meanerr,'stderr',stderr)
    assert float(diff.max())<2e-5 and meanerr<2e-6 and stderr<2e-6
    (OUT/'batched_noise_audit.json').write_text(json.dumps(dict(passed=True,sample_max_difference=float(diff.max()),
        mean_max_difference=meanerr,std_max_difference=stderr,seeds=[500000,500031],
        note='Same random initial tensors, CUDA convolution batching causes small rounding differences'),indent=2)+'\n')
