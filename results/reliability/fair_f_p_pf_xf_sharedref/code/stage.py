"""Stage validated result files for lossless GitHub transfer."""
import gzip
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
SRC=ROOT/'github/DiPred/results/reliability/fair_f_p_pf_xf_sharedref'
STAGE=ROOT/'runs/fair_f_p_pf_xf_sharedref_20261116/upload_stage'
DEST=STAGE/'results/reliability/fair_f_p_pf_xf_sharedref'

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def main():
    audit=json.loads((SRC/'audit.json').read_text());assert audit['passed']
    original=json.loads((SRC/'manifest.json').read_text());compressed=[]
    for p in sorted(SRC.rglob('*')):
        if not p.is_file() or p.name=='manifest.json':continue
        rel=p.relative_to(SRC);assert original['files_sha256'][str(rel)]==sha(p)
        q=DEST/rel;q.parent.mkdir(parents=True,exist_ok=True)
        if p.suffix=='.csv' and p.stat().st_size>1000000:
            q=q.with_suffix('.csv.gz')
            with p.open('rb') as f,q.open('wb') as raw:
                with gzip.GzipFile(fileobj=raw,filename='',mode='wb',mtime=0) as z:shutil.copyfileobj(f,z,1048576)
            h=hashlib.sha256()
            with gzip.open(q,'rb') as f:
                for block in iter(lambda:f.read(1048576),b''):h.update(block)
            assert h.hexdigest()==original['files_sha256'][str(rel)]
            compressed.append(str(rel))
        else:shutil.copy2(p,q)
    with (DEST/'README.md').open('a') as f:f.write('\nLarge CSV files are losslessly gzip-compressed; pandas.read_csv reads them directly.\n')
    original['csv_gzip_files']=compressed
    original['files_sha256']={str(p.relative_to(DEST)):sha(p) for p in sorted(DEST.rglob('*')) if p.is_file() and p.name!='manifest.json'}
    (DEST/'manifest.json').write_text(json.dumps(original,indent=2)+'\n')
    notice=STAGE/'results/reliability/fair_f_p_pf_xf/NOTICE.md'
    notice.parent.mkdir(parents=True,exist_ok=True)
    notice.write_text('# Reference-state limitation\n\nThis earlier run used normalized-zero output reference for F/P/PF and last-X-frame reference for XF. It is not a controlled input ablation against X/XP/XPF. For the corrected shared-reference experiment, see `../fair_f_p_pf_xf_sharedref/`. Existing data are preserved for traceability.\n')
    entries=[]
    for p in sorted(STAGE.rglob('*')):
        if p.is_file():
            relative=p.relative_to(STAGE).as_posix()
            assert all(c.isalnum() or c in '/._-' for c in relative)
            entries.append(dict(path=relative,bytes=p.stat().st_size,sha=subprocess.check_output(['git','hash-object',str(p)]).decode().strip()))
    (ROOT/'runs/fair_f_p_pf_xf_sharedref_20261116/upload_items.json').write_text(json.dumps(entries,indent=2)+'\n')
    print('STAGED',len(entries),'FILES',sum(v['bytes'] for v in entries),'BYTES',flush=True)
if __name__=='__main__':main()
