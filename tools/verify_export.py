from pathlib import Path
import argparse,json,hashlib


def main():
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path);args=p.parse_args();root=args.folder.resolve()
    receipt=json.loads((root/'complete.json').read_text())
    if receipt['patients']!=50:raise ValueError('Formal test is incomplete.')
    for name,expected in receipt['files'].items():
        f=(root/name).resolve()
        if root not in f.parents or hashlib.sha256(f.read_bytes()).hexdigest()!=expected:raise ValueError('Result checksum differs: '+name)
    print('Verified frozen-best 50-patient analysis delivery; 3D arrays are listed separately on the server.')


if __name__=='__main__':main()
