"""Independently replay DiPred on known observations and check its four calls."""
import json
import sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parent.parent
for sub in ('', 'fair_x_xp_xpf', 'xpf16_fullpercent_study', 'github/DiPred'):
    sys.path.insert(0,str(ROOT/sub))
from clean_reliability_common import configure,manifest
from experiment_reliability_transport_e12 import geometry,history
from reliability.selected16_predict import RequestedThreePredictor
from evaluate49 import prepare
from fair_x_xp_xpf.evaluate import OUT

@torch.no_grad()
def main():
    configure();records=manifest()['files']['calibration']
    raw=history('calibration',0);p=RequestedThreePredictor(geometry(records),'selected16','cuda')
    calls=[];handle=p.model.net.register_forward_pre_hook(lambda _,a:calls.append(int(a[1][0])))
    try:x,point,base,f=prepare(p,raw)
    finally:handle.remove()
    cached=torch.load(OUT/'inputs/calibration/0000.pt',map_location='cuda',weights_only=False)
    errors=dict(x=float((x-cached['x']).abs().max()),p=float((base[:,:2]-cached['p']).abs().max()),f=float((f-cached['f']).abs().max()))
    checks=dict(four_denoising_steps=calls==[99,66,33,0], frozen=not any(v.requires_grad for v in p.model.parameters()),
        cache_input_only=set(cached)=={'x','p','f','records','start'},
        issued_features_match=all(v<1e-6 for v in errors.values()),
        feature_shape=tuple(f.shape[1:])==(48,32,64))
    result=dict(checks=checks,passed=all(checks.values()),call_indices=calls,max_errors=errors,
                input='calibration known history at start 0; no future target read')
    (OUT/'feature_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2));assert result['passed']

if __name__=='__main__':main()
