from followcommon import *
from evaluate49 import prepare
from experiment_reliability_transport_e12 import geometry,history
from reliability.selected16_predict import RequestedThreePredictor
import pandas as pd
@torch.no_grad()
def main():
 setup();result={};d=cache('development');pred=RequestedThreePredictor(geometry(records('development')),'selected16','cuda');x,p,b,last=prepare(pred,history('development',0));p2,f=feature(model(),d['x'][:12],'diffusion');base,last2=features(d['x'][:12],p2,f)
 result['original_x_max_error']=float((x-d['x'][:12]).abs().max());result['original_p_max_error']=float((p-p2).abs().max());result['original_selected_base_max_error']=float((b[:,list(BASE_INDICES)]-base[:,list(BASE_INDICES)]).abs().max());result['original_last_feature_max_error']=float((last-last2).abs().max())
 assert result['original_x_max_error']==result['original_p_max_error']==result['original_last_feature_max_error']==0 and result['original_selected_base_max_error']<1e-5
 record=records('development');a=pd.read_csv(OUT/'audit/per_group.csv');errors={}
 for arm,folder,keybase in [('XPF',OLD/'evaluation/development',''),('XPF_grid250',W/'runs/xpf16_02grid_2048_20261112/evaluation/development','grid')]:
  subset=a if not keybase else pd.read_csv(OUT/'audit/grid250_per_group.csv')
  for mode in ('raw','online'):
   if keybase and mode=='raw':continue # archived raw is unrearranged; current raw is rearranged
   differences=[]
   for start in (0,20,80,160):
    z=np.load(folder/f'{start:04d}.npz')
    key=(mode+'_rearranged_crps') if not keybase else 'xpf_online_rearranged_crps'
    ref=z[key];ref=ref.mean(-1) if ref.ndim==3 else ref
    for i,r in enumerate(record):
     for c in range(2):
      rows=subset[(subset.arm==arm)&(subset['mode']==mode)&(subset.center=='original')&(subset.region=='all')&(subset.start==start)&(subset.file==r['file'])&(subset.channel==c)]
      differences.append(abs(float(rows.crps.iloc[0])-float(ref[i,c])))
   errors[arm+'_'+mode]=max(differences);assert max(differences)<2e-5
 result['historical_replay_max_standardized_crps_difference']=errors
 for kind in ('XPF','XP','F'):
  for k in range(1,50):
   path=OLD/'models'/f'XPF16-Q{k:02d}'/'model.pt' if kind=='XPF' else ABL/'models'/kind/f'Q{k:02d}.pt';v=load(path);assert v['parameters']==34004 and v['config']['seed']==42
 grid=json.loads((W/'runs/xpf16_02grid_2048_20261112/training_complete.json').read_text());gcfg=json.loads((OUT/'audit/grid250_config.json').read_text());assert gcfg['weights']=={f'P{k:03d}':grid['weights'][f'XPF16-P{k:03d}'] for k in range(1,251)}
 result['all_original_head_checkpoints_and_250_hashes']=True
 q=load(OUT/'query/predictions.pt');table=pd.read_csv(OUT/'query/per_query.csv');std=norm('cpu')['std'].flatten().numpy();err=0.
 for method,v in q['predictions'].items():
  Q=v['q'].numpy().astype('float64');y=q['targets'].numpy();t=q['levels'].numpy().astype('float64');e=y[None]-Q;T=t.reshape(-1,1,1,1);pin=np.maximum(T*e,(T-1)*e);calc=2*(np.trapz(pin,t,axis=0)+t[0]*(pin[0]+np.maximum(Q[0]-y,0))/2+(1-t[-1])*(pin[-1]+np.maximum(y-Q[-1],0))/2)*np.tile(std,2)
  for stat in range(2):
   for c in range(2):
    vals=table[(table.method==method)&(table.statistic==('mean','max')[stat])&(table.channel==c)].crps.to_numpy().reshape(108,160);err=max(err,float(np.abs(calc[:,:,c+stat*2]-vals).max()))
 assert err<2e-6;result['independent_numpy_query_crps_max_error']=err
 ep=load(OUT/'query/ecc_predictions.pt');et=pd.read_csv(OUT/'query/ecc_per_query.csv');maxerr=0.;qs=q['queries'];aa,bb=zip(*contain_pairs(qs))
 for method,v in ep.items():
  samples=v['samples'][:,:2,:3].numpy().astype('float64');y=q['targets'][:2,:3].numpy();s=(np.abs(samples-y).mean(0)-.5*np.abs(samples[:,None]-samples[None,:]).mean((0,1)))*np.tile(std,2)
  for i in range(2):
   for j in range(3):
    for stat in range(2):
     for c in range(2):
      row=et[(et.method==method)&(et.file==q['meta'][i]['file'])&(et.start==q['meta'][i]['start'])&(et['query']==j)&(et.statistic==('mean','max')[stat])&(et.channel==c)]
      maxerr=max(maxerr,abs(float(row.crps.iloc[0])-s[i,j,c+stat*2]))
  assert (v['event'][:,aa]<=v['event'][:,bb]+1e-6).all()
 assert maxerr<2e-6;result['independent_ecc_crps_max_error']=maxerr;result['ecc_nested_events_consistent']=True
 for p in (OUT/'query').glob('*.pt'):
  if p.stem in ('XP','XPF','XPQ','XPFQ','Q'):ck=load(p);assert ck['seed']==42 and ck['steps']==4000
 events=json.loads((OUT/'audit/events.json').read_text())
 for i in range(0,len(events),2):assert events[i]['event']=='issue_all_49' and events[i+1]['event']=='reveal_full20' and events[i]['start']==events[i+1]['start']
 result['causal_49_issue_then_reveal']=True;result['single_seed_42']=True;result['passed']=True;dump(OUT/'verification.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
