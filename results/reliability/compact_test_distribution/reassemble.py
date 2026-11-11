"""Reassemble the archived test snapshots and verify all SHA-256 hashes."""
import hashlib
import json
from pathlib import Path

folder = Path(__file__).resolve().parent
manifest = json.loads((folder / 'parts_manifest.json').read_text())
output = Path.cwd() / manifest['archive']
if output.exists():
    raise FileExistsError('Output already exists: {}'.format(output))
temporary = output.with_name(output.name + '.partial')
if temporary.exists():
    raise FileExistsError('Temporary file already exists: {}'.format(temporary))
hash_all = hashlib.sha256()
try:
    with temporary.open('xb') as stream:
        for item in manifest['parts']:
            data = (folder / 'parts' / item['file']).read_bytes()
            if len(data) != item['bytes'] or hashlib.sha256(data).hexdigest() != item['sha256']:
                raise ValueError('Corrupt archive part: {}'.format(item['file']))
            stream.write(data)
            hash_all.update(data)
    if temporary.stat().st_size != manifest['archive_bytes'] or hash_all.hexdigest() != manifest['archive_sha256']:
        raise ValueError('Reassembled archive SHA-256 mismatch')
    temporary.replace(output)
except Exception:
    temporary.unlink(missing_ok=True)
    raise
print('Verified archive:', output)
