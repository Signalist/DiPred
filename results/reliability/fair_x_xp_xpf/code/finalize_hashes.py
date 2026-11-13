"""Record and verify SHA-256 for each delivered file except the manifest itself."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
DEST=ROOT/'github/DiPred/results/reliability/fair_x_xp_xpf'
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
path=DEST/'manifest.json';data=json.loads(path.read_text())
data['files_sha256']={str(p.relative_to(DEST)):digest(p) for p in sorted(DEST.rglob('*')) if p.is_file() and p!=path}
path.write_text(json.dumps(data,indent=2)+'\n')
assert all(digest(DEST/name)==value for name,value in data['files_sha256'].items())
print('Verified output file hashes:',len(data['files_sha256']))
