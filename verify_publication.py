"""Verify the published files and source identities using the standard library."""
from pathlib import Path
import hashlib
import json


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / 'publication_manifest.json').read_text(encoding='utf-8'))
    for name, info in manifest['files'].items():
        path = (root / name).resolve()
        if root not in path.parents or not path.is_file():
            raise ValueError('Invalid or missing file: ' + name)
        if path.stat().st_size != info['bytes'] or sha(path) != info['sha256']:
            raise ValueError('Published content changed: ' + name)
    for line in (root / 'source/sources.sha256').read_text().splitlines():
        expected, name = line.split('  ', 1)
        if sha(root / 'source' / name) != expected:
            raise ValueError('Source fingerprint changed: ' + name)
    preparation = json.loads((root / 'provenance/share_preparation.json').read_text(encoding='utf-8'))
    copied = {row['path']: row for row in preparation['copied_files']}
    for name in preparation['core_source_byte_identical']:
        if sha(root / 'source' / name) != copied['source/' + name]['original_sha256']:
            raise ValueError('Scientific source differs from server: ' + name)
    print(f"Verified {len(manifest['files'])} published files and unchanged core scientific sources.")


if __name__ == '__main__':
    main()
