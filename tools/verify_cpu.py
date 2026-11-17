"""Reproduce engineering evidence without patient data or CUDA execution."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import ast
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def main():
    folder=ROOT/'verification';folder.mkdir(exist_ok=True)
    files=sorted(p for parent in ('src','tests','tools') for p in (ROOT/parent).rglob('*.py'))
    files+=[ROOT/'run.py',ROOT/'verify_release.py']
    for path in files:ast.parse(path.read_text(encoding='utf-8'),filename=str(path))
    environment=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUTF8='1')
    command=[sys.executable,'-X','utf8','-B','-m','unittest','discover','-s','tests','-t','.','-v']
    result=subprocess.run(command,cwd=ROOT,env=environment,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8')
    log=result.stdout
    (folder/'cpu_tests.log').write_text(log,encoding='utf-8')
    info=subprocess.run([sys.executable,'-X','utf8','-B','run.py','info'],cwd=ROOT,env=environment,capture_output=True,text=True,encoding='utf-8',check=True)
    (folder/'model_parameters.json').write_text(json.dumps(json.loads(info.stdout),indent=2)+'\n',encoding='utf-8')
    match=re.search(r'Ran (\d+) tests',log)
    config=ROOT/'configs/main.json'
    hashes={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files+[config]}
    report={'timestamp':datetime.now(timezone(timedelta(hours=8))).isoformat(),
        'experiment':json.loads(config.read_text(encoding='utf-8'))['experiment'],
        'scope':'CPU synthetic engineering checks only; no real patient, CUDA, official Mamba parity or efficacy evidence',
        'status':'passed' if result.returncode==0 else 'failed','returncode':result.returncode,
        'tests':int(match.group(1)) if match else None,'syntax_checked_files':len(files),
        'python':platform.python_version(),'platform':platform.platform(),
        'packages':{name:importlib.metadata.version(name) for name in ('torch','numpy','scipy','scikit-image','Pillow')},
        'command':['python',*command[1:]],'source_sha256':hashes}
    (folder/'cpu_verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(log,end='');print(json.dumps({k:report[k] for k in ('status','tests','syntax_checked_files')}))
    return result.returncode


if __name__=='__main__':raise SystemExit(main())
