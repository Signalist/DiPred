"""Restore the losslessly split checkpoint archive and verify every part."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
manifest = json.loads((root / 'weights_parts.json').read_text())
h = hashlib.sha256()
with (root / manifest['archive']).open('wb') as target:
    for item in manifest['parts']:
        block = (root / item['path']).read_bytes()
        if hashlib.sha256(block).hexdigest() != item['sha256']:
            raise ValueError(f'Checksum mismatch: {item["path"]}')
        h.update(block)
        target.write(block)
if h.hexdigest() != manifest['sha256']:
    raise ValueError('Reconstructed archive checksum mismatch')
print('Archive reconstructed and verified:', root / manifest['archive'])
