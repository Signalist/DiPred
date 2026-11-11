"""Per-head latency on frozen development inputs, with GPU synchronization."""
import json,sys,time
from pathlib import Path
import torch
from clean_reliability_common import ROOT,configure,manifest
from experiment_reliability_transport_e12 import geometry,history
from experiment_selected16_quantile import new_head
from train49 import OUT,configs
from evaluate49 import prepare,make_head_batches,all_bounds
sys.path.insert(0,str(ROOT/'github/DiPred'))
from reliability.selected16_predict import RequestedThreePredictor

@torch.no_grad()
def main():
    configure();records=manifest()['files']['development']
    p=RequestedThreePredictor(geometry(records),'selected16','cuda')
    observed=history('development',0)
    for _ in range(3):
        h,point,base,last=prepare(p,observed)
    torch.cuda.synchronize();t=time.perf_counter()
    for _ in range(10):
        h,point,base,last=prepare(p,observed)
    torch.cuda.synchronize();backbone=(time.perf_counter()-t)/10
    heads=[];rows=[]
    for cfg in configs():
        head=new_head('compact16',None)
        checkpoint=torch.load(OUT/'models'/cfg['id']/'model.pt',map_location='cuda',weights_only=False)
        head.load_state_dict(checkpoint['model']);head.eval().requires_grad_(False)
        for _ in range(3):head.bounds(base,last,None,None,False)
        torch.cuda.synchronize();begin=time.perf_counter()
        for _ in range(10):head.bounds(base,last,None,None,False)
        torch.cuda.synchronize();elapsed=(time.perf_counter()-begin)/10
        rows.append(dict(id=cfg['id'],head_seconds_per_12_trajectory_batch=elapsed,parameters=34004))
        heads.append(head)
    batches=make_head_batches(heads)
    for _ in range(3):all_bounds(batches,base,last)
    torch.cuda.synchronize();begin=time.perf_counter()
    for _ in range(10):all_bounds(batches,base,last)
    torch.cuda.synchronize();combined=(time.perf_counter()-begin)/10
    data=dict(device=torch.cuda.get_device_name(0),batch_size=len(records),repetitions=10,
              point_plus_feature_seconds=backbone,head49_batched_seconds=combined,
              interpretation='latency excludes IO, interval calibration, output transfer, and 256 stochastic paths; wall seconds per batch',heads=rows)
    (OUT/'latency.json').write_text(json.dumps(data,indent=2)+'\n')
    print('LATENCY49_COMPLETE',flush=True)

if __name__=='__main__':main()
