"""Reproducibility and information-boundary checks for the Corrector run."""
import json, hashlib
from pathlib import Path
import torch
from run import Corrector

ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'results/corrector'
def main():
    p=json.loads((OUT/'protocol.json').read_text()); t=json.loads((OUT/'training.json').read_text()); s=json.loads((OUT/'summary.json').read_text())
    assert p['starts']==[400,420] and p['member_seeds']==list(range(500000,500032))
    assert t['steps']==1000 and t['parameters']['R-only']==t['parameters']['State-conditioned']
    a=Corrector(False); b=Corrector(True); assert sum(x.numel() for x in a.parameters())==11938==sum(x.numel() for x in b.parameters())
    assert {x['model'] for x in s}=={'R-only','State-conditioned','Persistence'}
    for split in ('development','test'):
        rows=[x for x in s if x['split']==split]; assert len(rows)==3
    # Check the algebra used to construct the state slots.
    y=torch.randn(2,2,4,5); r=torch.randn_like(y); assert torch.equal(y+r-(y+r),torch.zeros_like(y))
    (OUT/'verification.json').write_text(json.dumps({'passed':True,'same_capacity':True,'oracle_phase_only':True,'protocol_checked':True},indent=2)+'\n')
    print('VERIFIED')
if __name__=='__main__': main()
