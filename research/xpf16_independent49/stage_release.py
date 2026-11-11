"""Stage audited results for publishing; no network operations."""
import gzip
import hashlib
import json
import shutil
from pathlib import Path
from train49 import OUT, configs

ROOT=Path(__file__).resolve().parent.parent
REPO=ROOT/'github/DiPred'
DEST=REPO/'results/reliability/xpf16_independent49'
SRC=REPO/'research/xpf16_independent49'


def split_file(src,destination):
    destination.mkdir(parents=True,exist_ok=True)
    sha=hashlib.sha256();size=0;index=0
    with src.open('rb') as f:
        while True:
            block=f.read(600_000)
            if not block:break
            sha.update(block);size+=len(block)
            (destination/f'part{index:04d}').write_bytes(block);index+=1
    return dict(filename=src.name,sha256=sha.hexdigest(),size=size,parts=index)


def main():
    assert json.loads((OUT/'independent_audit.json').read_text())['passed']
    assert (OUT/'summary/REPORT.md').exists()
    DEST.mkdir(parents=True,exist_ok=True)
    SRC.mkdir(parents=True,exist_ok=True)
    weights=DEST/'weights';weights.mkdir(exist_ok=True)
    histories={}
    for c in configs():
        source=OUT/'models'/c['id']
        shutil.copy2(source/'model.pt',weights/(c['id']+'.pt'))
        histories[c['id']]=json.loads((source/'training.json').read_text())
    (DEST/'training_histories.json').write_text(json.dumps(histories,indent=2)+'\n')
    for path in ('protocol.json','q05_recheck.json','batch_training_audit.json','batched_noise_audit.json','portable49_audit.json','latency.json','training_complete.json','evaluation_complete.json','independent_audit.json'):
        shutil.copy2(OUT/path,DEST/path)
    for path in ('REPORT.md','macro49.csv','source_hashes.json','development_distribution_windows.json','test_distribution_windows.json'):
        shutil.copy2(OUT/'summary'/path,DEST/path)
    shutil.copy2(OUT/'initial_offsets.pt',DEST/'initial_offsets.pt')
    shutil.copy2(OUT/'noise256_calibration.pt',DEST/'noise256_calibration.pt')
    shutil.copy2(OUT/'evaluation/test/snapshot.npz',DEST/'test_snapshot.npz')
    shutil.copytree(OUT/'summary/figures',DEST/'figures',dirs_exist_ok=True)
    for split in ('development','test'):
        shutil.copy2(OUT/'evaluation'/split/'run.json',DEST/(split+'_run.json'))
    parts={}
    for split in ('development','test'):
        per=OUT/'summary'/f'{split}_per_window.csv.gz'
        parts[per.name]=split_file(per,DEST/'split_data'/per.name)
        lead=OUT/'summary'/f'{split}_group_lead.csv'
        compressed=OUT/'summary'/f'{split}_group_lead.csv.gz'
        with lead.open('rb') as a,gzip.open(compressed,'wb',compresslevel=9) as b:
            shutil.copyfileobj(a,b)
        parts[compressed.name]=split_file(compressed,DEST/'split_data'/compressed.name)
    (DEST/'split_data_manifest.json').write_text(json.dumps(parts,indent=2)+'\n')
    for path in Path(__file__).parent.glob('*.py'):
        if not path.name.startswith('benchmark_vectorized') and path.name!='benchmark_batched_noise.py':
            shutil.copy2(path,SRC/path.name)
    shutil.copy2(Path(__file__).parent/'README.md',SRC/'README.md')
    shutil.copy2(Path(__file__).parent/'portable49.py',REPO/'reliability/independent49_predict.py')
    with (DEST/'REPORT.md').open('a') as f:
        f.write('\n逐窗口及逐帧分组数据位于 `split_data/`；运行 `python research/xpf16_independent49/restore_result_parts.py results/reliability/xpf16_independent49 --output-directory reconstructed` 可拼接并核验 SHA-256。\n')
        f.write('\n可复用接口：`from reliability.independent49_predict import XPFIndependentPredictor`。构造函数传入网格几何、`weights/`、`initial_offsets.pt` 和所需 levels；依次调用 `predict(history)` 与 `reveal(full20_truth)`。levels 默认为全部 49 个，单组可指定如 `[5]`。\n')
    print('STAGED',DEST, 'parts',sum(p['parts'] for p in parts.values()))

if __name__=='__main__':main()
