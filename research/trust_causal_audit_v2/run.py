"""One audit entry; deliberately has no training or scoring mode."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from research.trust_causal_audit_v2.common import OUT,freeze_originals,assert_protected,dump


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['freeze','tests','data','all'])
    args=parser.parse_args()
    if args.stage=='freeze':
        freeze_originals();return
    if not (OUT/'PROTECTED_BEFORE.json').exists():
        freeze_originals()
    assert_protected()
    # Enforce zero optimizer construction/updates even if future edits add a
    # hidden optimizer in any reachable imported module. Tests use autograd only.
    import torch
    def forbidden_optimizer(*args,**kwargs):
        raise AssertionError('Audit-only protocol forbids optimizer construction')
    torch.optim.Optimizer.__init__=forbidden_optimizer
    try:
        if args.stage in ('tests','all'):
            from research.trust_causal_audit_v2.tests import run
            result=run();print('A-G tests PASS',result['parameters'],flush=True)
        if args.stage in ('data','all'):
            from research.trust_causal_audit_v2.audit import run
            result=run()
        count=assert_protected()
        dump(OUT/'EXECUTION.json',dict(stage=args.stage,protected_files_unchanged=count,
            optimizer_construction_guard_enabled=True,optimizer_updates=0,training_runs=0,
            formal_test_comparisons=0,new_native_cache_generation=0))
    except Exception as exc:
        dump(OUT/'FAILURE.json',dict(stage=args.stage,error=repr(exc),optimizer_updates=0))
        raise


if __name__=='__main__':main()
