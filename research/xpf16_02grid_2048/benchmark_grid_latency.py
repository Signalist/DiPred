"""Record warmed single-head and all-head inference time on dev batch."""
import json,sys,time,statistics
import torch
from clean_reliability_common import ROOT,configure,manifest
from experiment_reliability_transport_e12 import geometry,history
from experiment_selected16_quantile import new_head
from train_grid import OUT,configs
sys.path.insert(0,str(ROOT/'xpf16_fullpercent_study'))
from evaluate49 import prepare,make_head_batches,all_bounds
sys.path.insert(0,str(ROOT/'github/DiPred'))
from reliability.selected16_predict import RequestedThreePredictor

@torch.no_grad()
def main():
    configure();rec=manifest()['files']['development'];p=RequestedThreePredictor(geometry(rec),'selected16','cuda')
    observed=history('development',0)
    for _ in range(3):h,point,base,last=prepare(p,observed)
    torch.cuda.synchronize();begin=time.perf_counter()
    for _ in range(10):h,point,base,last=prepare(p,observed)
    torch.cuda.synchronize();backbone=(time.perf_counter()-begin)/10
    heads=[];rows=[]
    for cfg in configs():
        model=new_head('compact16',None)
        model.load_state_dict(torch.load(OUT/'models'/cfg['id']/'model.pt',map_location='cuda',weights_only=False)['model'])
        model.eval().requires_grad_(False)
        for _ in range(3):model.bounds(base,last,None,None,False)
        torch.cuda.synchronize();begin=time.perf_counter()
        for _ in range(10):model.bounds(base,last,None,None,False)
        torch.cuda.synchronize();t=(time.perf_counter()-begin)/10
        rows.append(dict(id=cfg['id'],seconds_per_batch=t,parameters=34004))
        heads.append(model)
    batches=make_head_batches(heads)
    for _ in range(3):all_bounds(batches,base,last)
    torch.cuda.synchronize();begin=time.perf_counter()
    for _ in range(10):all_bounds(batches,base,last)
    torch.cuda.synchronize();combined=(time.perf_counter()-begin)/10
    info=dict(device=torch.cuda.get_device_name(0),batch_size=len(rec),repetitions=10,
       point_and_feature_seconds=backbone,all_250_heads_batched_seconds=combined,
       median_single_head_seconds=statistics.median(r['seconds_per_batch'] for r in rows),heads=rows,
       scope='warm per-batch inference; excludes IO, calibration, result transfer and sample generation')
    (OUT/'latency.json').write_text(json.dumps(info,indent=2)+'\n')
    print('GRID_LATENCY_COMPLETE',flush=True)

if __name__=='__main__':main()
