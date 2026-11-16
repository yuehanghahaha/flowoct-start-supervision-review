"""Verify the frozen public source/results snapshot without running a model."""
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np

ROOT=Path(__file__).resolve().parent


def main():
    manifest=json.loads((ROOT/'SNAPSHOT_MANIFEST.json').read_text(encoding='utf-8'))
    failures=[];total=0;arrays=0;maximum=0
    sensitive=[('GitHub token',re.compile(r'\b(?:gh[pousr]_[A-Za-z0-9]{25,}|github_pat_[A-Za-z0-9_]{60,})\b')),
               ('private key',re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----')),
               ('credential-bearing URL',re.compile(r'https?://[^\s/:@]+:[^\s/@]+@'))]
    for row in manifest['files']:
        p=(ROOT/row['path']).resolve()
        if not p.is_relative_to(ROOT.resolve()) or not p.is_file():failures.append((row['path'],'missing or outside snapshot'));continue
        raw=p.read_bytes();total+=len(raw);maximum=max(maximum,len(raw))
        if len(raw)!=row['bytes'] or hashlib.sha256(raw).hexdigest()!=row['sha256']:failures.append((row['path'],'hash mismatch'))
        if len(raw)>100*1024**2:failures.append((row['path'],'exceeds normal GitHub file limit'))
        if p.suffix.lower() in {'.pt','.pth','.ckpt','.npy','.mat'}:failures.append((row['path'],'unexpected model/original array'))
        if p.name.startswith('.env'):failures.append((row['path'],'environment file'))
        if p.suffix.lower() in {'.py','.json','.jsonl','.md','.sh','.ps1','.txt','.log','.yaml','.yml','.toml'}:
            text=raw.decode('utf-8-sig',errors='replace')
            for kind,pattern in sensitive:
                if pattern.search(text):failures.append((row['path'],kind))
        if p.suffix.lower()=='.npz':
            arrays+=1
            with np.load(p,allow_pickle=False) as z:
                for k in z.files:
                    x=z[k]
                    if x.ndim>2 or not np.isfinite(x).all():failures.append((row['path'],'unexpected dimensions/nonfinite array'))
    if arrays!=60:failures.append(('2D evidence count',str(arrays)))
    for n in manifest['scientific_training_experiments']:
        folders=list((ROOT/'experiments').glob(f'exp{n:03d}_*'))
        if len(folders)!=1 or not (folders[0]/'experiment.json').is_file() or not (folders[0]/'evidence').is_dir():failures.append((str(n),'missing experiment code or evidence'))
    for n in [173,174]:
        folders=list((ROOT/'analyses').glob(f'exp{n:03d}_*'))
        if len(folders)!=1 or not (folders[0]/'analyze.py').is_file():failures.append((str(n),'missing runnable analysis'))
    if failures:
        print(json.dumps(dict(passed=False,issues=failures),ensure_ascii=False,indent=2));raise SystemExit(1)
    print(json.dumps(dict(passed=True,files=len(manifest['files'])+1,MiB=round(total/2**20,2),
        max_file_MiB=round(maximum/2**20,2),scientific_experiments=9,analyses=2,
        full_resolution_2D_projection_pairs=arrays,credentials_detected=False,new_training=0,new_inference=0),ensure_ascii=False))


if __name__=='__main__':main()
