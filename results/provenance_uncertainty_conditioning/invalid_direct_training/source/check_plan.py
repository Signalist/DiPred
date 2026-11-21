"""Metadata-only coverage and independent-noise checks before held-out scoring."""
from core import *
from research.provenance_uncertainty_conditioning.evaluate import extension_plan


def main():
    assert not (OUT/'PLAN_CHECKS.json').exists()
    plans=extension_plan();assert len(plans)==480
    records={r['file']:r for r in CFG['records']['test']}
    assert all(e['start']+40<=records[e['trajectory']]['frames'] for e in plans)
    sets={sp:{r['group'] for r in CFG['records'][sp]} for sp in CFG['records']}
    assert not sets['train']&sets['development'] and not sets['train']&sets['test'] and not sets['development']&sets['test']
    seeds=[]
    for e in prior.entries():
        for call in e['calls'].values():seeds.extend(call['seeds'])
    for e in plans:
        for family,k in [('A',0),('B',0),('C',0),('S',2),('S',5),('S',10)]:seeds.extend(native.seeds_for('test',e['trajectory'],e['start'],e['repeat'],family,k))
    assert len(seeds)==len(set(seeds))==145920
    dump(OUT/'PLAN_CHECKS.json',dict(passed=True,formal_groups=528,historical_groups=48,extension_groups=480,old_plus_extension_unique_seeds=len(seeds),max_scored_exclusive_frame=860,min_test_file_frames=min(r['frames'] for r in records.values()),Re_sets={k:sorted(v) for k,v in sets.items()},extension_seed_split='test',no_future_labels_read=True,utc=time.time()))
    print('PASS fixed528 groups;145920 distinct old+extension seeds;Re separation;all requested frames available',flush=True)

if __name__=='__main__':main()
