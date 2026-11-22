"""Read-only delivery audit for the fixed experiment."""
import csv, hashlib, json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/error_guided_spatial_routing'

def digest(p):
    h=hashlib.sha256();
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()

def main():
    pre=json.loads((OUT/'PREFLIGHT.json').read_text()); assert pre['passed']
    sel=json.loads((OUT/'development_selection.json').read_text())
    assert sel['G_context']==0 and sel['G_trust']==0
    for arm in ('G_context','G_trust'):
        rows=list(csv.DictReader((OUT/f'training_{arm}.csv').open()))
        assert len(rows)==1000 and int(rows[-1]['step'])==1000
        dev=list(csv.DictReader((OUT/f'development_{arm}.csv').open()))
        assert len(dev)==144 and {int(float(r['step'])) for r in dev}=={0,250,500,1000}
        for step in (0,250,500,1000):assert (OUT/'weights'/f'{arm}_{step}.pt').exists()
    test=list(csv.DictReader((OUT/'test_scores.csv').open()))
    assert len(test)==12 and {r['arm'] for r in test}=={'native'}
    assert json.loads((OUT/'data_audit.json').read_text())['effective_train_windows']==135
    assert digest(ROOT/'reliability/checkpoints/current.pt')==json.loads((OUT/'manifest.json').read_text())['checkpoint_sha256']
    result={'passed':True,'training_logs_checked':2,'updates_per_model':1000,
            'candidate_checkpoints_per_model':4,'selection_native_for_both':True,
            'formal_test_rows':len(test),'future_input_boundary_checked':pre['future_truth_input_independent']}
    (OUT/'VERIFICATION.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(result,indent=2,ensure_ascii=False))

if __name__=='__main__':main()
