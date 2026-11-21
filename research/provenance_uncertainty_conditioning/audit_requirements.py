"""Final task-scope audit against the user's fixed requirements, not just our protocol."""
from core import *


def main():
    assert not (OUT/'DELIVERY_AUDIT.json').exists()
    c=CFG;assert c['reference']=='50c25fdeb797de21703c039aeb13e6167b26bdbd'
    assert c['train_k']==[2,5,10] and c['k']==[2,5,10] and c['eval_k']==[0,2,5,10]
    assert c['M']==32 and c['seeds']==[42,43,44] and c['arms']==['C_aug','C_mask','C_unc']
    assert c['steps']==1000 and c['candidates']==[0,250,500,750,1000]
    assert {k:len(v) for k,v in c['records'].items()}=={'train':46,'development':12,'test':12}
    schedule=json.loads((OUT/'batch_schedule.json').read_text());assert len(schedule)==1000 and all(r['k'] in [2,5,10] for r in schedule)
    training=read_csv(OUT/'training.csv');assert len(training)==9000 and all(int(r['k']) in [2,5,10] for r in training)
    assert {(r['arm'],int(r['seed'])) for r in training}=={(a,s) for a in c['arms'] for s in c['seeds']}
    assert all({int(r['step']) for r in training if r['arm']==a and int(r['seed'])==s}==set(range(1,1001)) for a in c['arms'] for s in c['seeds'])
    pre=json.loads((OUT/'PREFLIGHT.json').read_text());verification=json.loads((OUT/'VERIFICATION.json').read_text());assert pre['passed'] and verification['passed']
    assert pre['checks']['pair_term_adapter_gradient_norm']>0 and min(pre['checks']['four_step_adapter_gradient_rms'])>0
    selected=json.loads((OUT/'selection.json').read_text());assert len(selected)==18 and {r['policy'] for r in selected}=={'scientific','guarded'}
    assert len(json.loads((OUT/'checkpoints.json').read_text()))==45
    complete=json.loads((OUT/'EVALUATION_COMPLETE.json').read_text());assert complete['completed']==528
    assert verification['groups']==528 and verification['extension_sources']==480 and verification['all_lead_rows_checked']>0
    analysis=json.loads((OUT/'ANALYSIS.json').read_text());assert analysis['score_rows']==220176
    assert (OUT/'DEVELOPMENT_ANALYSIS.json').exists()
    report=(OUT/'DECISION_REPORT.md').read_text()
    assert all(f'{i}.' in report for i in range(1,9))
    required=['PROTOCOL.md','config.json','TASK.md','PROTOCOL_CORRECTION.md','STATUS.md','EXECUTED_COMMANDS.json','DECISION_REPORT.md','selection_candidates.csv','paired_intervals.csv','scores_by_trajectory_seed.csv','scores_by_Re.csv','lead_summary.csv','EVALUATION_COST.json','TRAINING_COMPLETE.json','REVIEW_FILES.md']
    assert all((OUT/name).exists() for name in required)
    check()
    evidence={
      '1_scope_frozen_backbone':'reference HEAD, config dependencies, PREFLIGHT and VERIFICATION hashes; only new directories changed',
      '2_equal_capacity_three_arms':'1760 parameters per checkpoint; seed-aligned zero-effective initialization; mask/std slots masked by arm',
      '3_training_domain_and_truth_boundary':'all9000 valid update rows k2/5/10 only;528 issued-before-truth groups; trajectory/Re isolation; all-real direct evaluated separately',
      '4_nine_real_full_ensemble_fits':'nine1000-update logs; exact full32 CRPS with pair term and tested gradients;45 checkpoint hashes',
      '5_pretraining_checks':'PREFLIGHT nine requirements plus EVALUATION_TESTS and PLAN_CHECKS, before valid TRAINING_STARTED',
      '6_two_selection_policies':'18 choices independently recomputed; all45 candidate development metrics retained; frozen before formal scoring',
      '7_required_comparisons':'tail/joint/direct, S/B and incremental arm contrasts; covariance/ACF/seam/distribution baselines; all seeds/Re/channels/leads/windows',
      '8_scientific_answers_and_stop':'DECISION_REPORT answers1..8, fixed DECISIONS criteria, no automatic follow-on model/loss/search',
      '9_delivery':'code/tests/protocol/logs/weights/hashes/trajectory statistics/cost/Chinese report; final commit and push verified separately',
      'invalid_attempt_segregation':'invalid_direct_training retains9000 noncompliant updates,45 weights and stopped evaluation; excluded from every official selection/statistic; costs disclosed'}
    dump(OUT/'DELIVERY_AUDIT.json',dict(passed=True,reference=c['reference'],valid_updates=9000,invalid_updates_separately_preserved=9000,training_k=[2,5,10],checkpoints=45,choices=18,groups=528,requirements=evidence,scientific_success_not_required_for_execution_completion=True,utc=time.time()))
    print('PASS task-scope audit; valid nine fits use only requested k2/5/10; prior invalid attempt excluded')

if __name__=='__main__':main()
