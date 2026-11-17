"""Interleaved batch1 complete forecasts and selected-sensor updates."""
from critic import *
from active_stream import SparseUpdatePredictor

@torch.no_grad()
def main():
    setup();d=load(SOURCE/'development.pt');n=norm();inputs=[(d['x'][i:i+1].cuda()*n['std']+n['mean']).cpu() for i in range(6)];truth=[(d['y'][i].cuda()*n['std'][0]+n['mean'][0]).cpu() for i in range(6)];rows=[]
    for family in ('global','predictive'):
        predictor=SparseUpdatePredictor('uniform',8) if family=='global' else make_predictor('uniform',8);planner=Planner() if family=='predictive' else None;methods=['uniform','random','variance','risk','voi','voi_average_scale']+(['critic'] if planner is not None else [])
        for _ in range(2):
            req=predictor.predict(inputs[0],0,d['meta'][0]['file']);predictor.update(observation_rows(truth[0])[req['site_indices']],25)
        for repeat in range(5):
            order=methods[repeat%len(methods):]+methods[:repeat%len(methods)]
            for method in order:
                predictor.method=method;predictor.planner=planner
                for i,x in enumerate(inputs):
                    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter();req=predictor.predict(x,0,d['meta'][i]['file']);torch.cuda.synchronize();forecast_ms=(time.perf_counter()-start)*1000;values=observation_rows(truth[i])[req['site_indices']];start=time.perf_counter();q=predictor.update(values,25);torch.cuda.synchronize();update_ms=(time.perf_counter()-start)*1000;peak=torch.cuda.max_memory_allocated()/2**20;rows.append(dict(family=family,method=method,repeat=repeat,index=i,forecast_and_planning_ms=forecast_ms,update_ms=update_ms,total_compute_ms=forecast_ms+update_ms,peak_mib=peak,budget=8))
        del predictor,planner;torch.cuda.empty_cache();print('BENCHMARK',family,flush=True)
    df=pd.DataFrame(rows);df.to_csv(BASE/'timing_samples.csv',index=False);df.groupby(['family','method'])[['forecast_and_planning_ms','update_ms','total_compute_ms','peak_mib']].median().to_csv(BASE/'timing.csv');dump(BASE/'timing_protocol.json',dict(batch=1,history_input='CPU physical full20',observation_input='CPU selected8x2',repeats=5,windows_per_repeat=6,interleaved_policy_order=True,excluded='sensor acquisition/wait5frames; constructor/model loading; offline training',meaning='Compute ms per forecast+one sparse update, not wallclock forecast horizon latency',source_sha256=digest(Path(__file__))));print(df.groupby(['family','method']).total_compute_ms.median().to_string(),flush=True)

if __name__=='__main__':main()
