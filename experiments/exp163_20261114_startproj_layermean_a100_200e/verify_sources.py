"""Verify the delivered code, configuration and annotations with the Python standard library."""

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def verify():
    manifest = json.loads((ROOT/'provenance/delivery_sha256.json').read_text(encoding='utf-8'))
    for name, expected in manifest.items():
        path = ROOT/name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('Delivered file missing or changed: '+name)
    print(f'Delivery verified, {len(manifest)} files.', flush=True)
    return manifest


if __name__ == '__main__':
    verify()
