"""Check the published snapshot using only Python's standard library."""
from pathlib import Path
import json,hashlib


def main():
    root=Path(__file__).resolve().parent
    manifest=json.loads((root/'release_manifest.json').read_text(encoding='utf-8'))
    for name,info in manifest['files'].items():
        path=(root/name).resolve()
        if root not in path.parents or not path.is_file():raise ValueError('Invalid/missing release file: '+name)
        if path.stat().st_size!=info['bytes'] or hashlib.sha256(path.read_bytes()).hexdigest()!=info['sha256']:
            raise ValueError('Release differs: '+name)
    print(f"Verified {len(manifest['files'])} release files. GPU/data acceptance is explicitly pending.")


if __name__=='__main__':main()
