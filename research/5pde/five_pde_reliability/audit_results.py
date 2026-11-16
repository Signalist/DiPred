"""Full-test result, causal update and score audit, without selective deletion."""
import argparse,json,sys
import numpy as np
import torch
from common import OUT,Scenario,records
from evaluate import ARMS,NOISE_ARMS,MODES,LEVELS


def main():
    parser=argparse.ArgumentParser();parser.add_argument('scenario');args=parser.parse_args();name=args.scenario
    sc=Scenario(name,device='cpu');rs=records(name);test=[r for r in rs if r['split']=='test'];n=len(test)
    expected=(min(r['frames'] for r in test)-40)//20
    checks={'all_models_complete':True,'all_windows_complete':True,'causal_feedback':True,
            'test_ids_ordered_and_disjoint':True,'raw_crossings_zero':True,'scores_finite':True,
            'quantile_coverage_ordered':True,'cdf_valid':True,'crps_snapshot_recomputed':True,
            'calibration_separate':True,'preflight_passed':True,'field_audit_passed':True,'noise_prefix_passed':True}
    checks['preflight_passed']=json.loads((OUT/'preflight'/f'{name}.json').read_text())['passed']
    checks['field_audit_passed']=json.loads((OUT/'field_audit'/f'{name}.json').read_text())['passed']
    checks['noise_prefix_passed']=json.loads((OUT/'noise_audit'/f'{name}.json').read_text())['passed']
    cal_ids={r['id'] for r in rs if r['split']=='calibration'}
    for arm,seed in [(arm,seed) for seed in (42,123,2026) for arm in ARMS]+[(arm,500000) for arm in NOISE_ARMS]:
            path=OUT/'evaluated'/name/f'{arm}_{seed}.pt'
            if not path.exists():checks['all_models_complete']=False;continue
            p=torch.load(path,map_location='cpu',weights_only=False)
            checks['test_ids_ordered_and_disjoint'] &= [r['id'] for r in p['records']]==[r['id'] for r in test]
            checks['all_windows_complete'] &= len(p['windows'])==3*expected and len(p['frames'])==3*expected
            events=p['events'];checks['causal_feedback'] &= len(events)==2*(expected+1)
            for i in range(expected+1):
                a,b=events[2*i:2*i+2]
                checks['causal_feedback'] &= a['event']=='issued' and b['event']=='revealed' and a['start']==b['start']==i*20 and a['time']<=b['time']
            for row in p['windows']:
                a=row['values'][:,:sc.channels]
                checks['scores_finite'] &= bool(np.isfinite(a).all())
                if row['mode']==0:checks['raw_crossings_zero'] &= bool((a[:,:,4]==0).all())
            coverage=p['qcoverage'].numpy();cdf=p['cdf'].numpy()
            checks['quantile_coverage_ordered'] &= bool((np.diff(coverage,axis=1)>=-1e-7).all())
            checks['cdf_valid'] &= bool(np.isfinite(cdf).all() and (np.diff(cdf,axis=2)>=-1e-6).all() and cdf.min()>=-1e-5 and cdf.max()<=1+1e-5)
            calibration=torch.load(OUT/'calibration'/name/f'{arm}_{seed}.pt',map_location='cpu',weights_only=False)
            checks['calibration_separate'] &= set(calibration['ids'])==cal_ids and cal_ids.isdisjoint({r['id'] for r in test})
            for sample in p['snapshots']:
                q=sample['q'].numpy();y=sample['y'].numpy();level=LEVELS.numpy().reshape(-1,1,1,1,1)
                err=y[None]-q;pin=np.maximum(level*err,(level-1)*err)
                result=.004*(.5*(np.maximum(q[0]-y,0)+np.maximum(y-q[-1],0))+pin.sum(0))
                result*=np.array(sc.info['std']).reshape(-1,1,1,1)
                checks['crps_snapshot_recomputed'] &= bool(np.allclose(result,sample['crps'].numpy(),rtol=3e-5,atol=2e-6))
    checks={k:bool(v) for k,v in checks.items()};result=dict(scenario=name,checks=checks,passed=all(checks.values()),
        test_trajectories=n,evaluated_windows_per_trajectory=expected,observations_per_channel=n*expected*20*32*64,
        caveat='The prior project may have used these trajectories; no fresh-independent-test claim.')
    path=OUT/'audit'/f'{name}.json';path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True);assert result['passed']

if __name__=='__main__':main()
