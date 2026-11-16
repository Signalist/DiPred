"""Audit data isolation and genuine frozen denoising features before testing."""
import argparse,json,sys
from pathlib import Path
import torch
from common import ROOT,OUT,Scenario,configure,predict,records
from evaluate import ARMS,load_head

@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('scenario');args=parser.parse_args()
    configure();name=args.scenario;rs=records(name);sc=Scenario(name);checks={}
    splits={s:{r['id'] for r in rs if r['split']==s} for s in ('train','development','calibration','test')}
    checks['trajectory_splits_disjoint']=all(splits[a].isdisjoint(splits[b]) for i,a in enumerate(splits) for b in list(splits)[i+1:])
    checks['official_test_never_in_training']=all(r['split']=='test' for r in rs if r['official_group'] in ('in_dist_test','out_dist_test','unlisted_test'))
    cache=json.loads((OUT/'risk_data'/name/'complete.json').read_text())
    checks['head_fit_uses_only_training_trajectories']=set(cache['ids']).issubset(splits['train'])
    backbone_state=torch.load(OUT/'backbones'/name/'model.pt',map_location='cuda',weights_only=False)
    checks['backbone_fit_uses_only_training_trajectories']=set(backbone_state['train_ids'])==splits['train']
    checks['normalization_uses_only_training_trajectories']='training trajectories only' in sc.info['stats']
    checks['three_equal_heads_per_seed']=True
    for seed in (42,123,2026):
        weights=[load_head(name,arm,seed)[1] for arm in ARMS]
        checks['three_equal_heads_per_seed'] &= len({w['head_parameters'] for w in weights})==1 and len({w['init_sha256'] for w in weights})==1
    backbone=sc.network().eval().requires_grad_(False);backbone.load_state_dict(backbone_state['model'])
    labels=[];hook=backbone.net.register_forward_pre_hook(lambda module,args:labels.append(int(args[1][0])))
    x=sc.batch([next(r for r in rs if r['split']=='development')],[0])
    try:p,f=predict(backbone,x,sc.mask,with_features=True)
    finally:hook.remove()
    checks['actual_four_denoising_calls']=labels==[99,66,33,0] and tuple(f.shape[1:])==(48,32,64)
    checks['frozen_backbone']=not any(q.requires_grad for q in backbone.parameters())
    head,_,mean,std=load_head(name,'X',42);feature=((f-mean)/std).clamp(-10,10);tau=torch.tensor([.1,.5,.9],device='cuda')
    zero=torch.zeros(1,2,device='cuda');xp=torch.tensor([[1.,0.]],device='cuda')
    a=head.quantiles(x,p,feature,zero,tau);b=head.quantiles(x,p*20+3,feature*20-3,zero,tau)
    c=head.quantiles(x,p,feature,xp,tau);d=head.quantiles(x,p,feature*20-3,xp,tau)
    checks['missing_P_F_slots_zeroed']=bool(torch.equal(a,b) and torch.equal(c,d))
    checks['raw_quantiles_monotone']=bool((a[1:]>=a[:-1]).all())
    checks['dummy_channel_excluded']=sc.channels==len(sc.info['channels']) and (sc.channels==2 or float(sc.mask[0,1])==0.)
    result={'scenario':name,'checks':checks,'passed':all(checks.values()),'notes':'Four calls depend only on observed history. Test targets are not read by this audit.'}
    (OUT/'preflight').mkdir(exist_ok=True,parents=True)
    (OUT/'preflight'/f'{name}.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2));assert result['passed']

if __name__=='__main__':main()
