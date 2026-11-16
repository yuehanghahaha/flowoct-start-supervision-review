"""Direct directory synchronization and remote operations; no delivery archives."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shlex
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parent
HOST = 'eias-member.eitech.edu.cn'
REMOTE = '/docsystem/niuke/lvyuehang/Flow_oct2octa/experience/' + ROOT.name
BOOTSTRAP = '/docsystem/niuke/lvyuehang/conda_envs/lyh_flowoct/bin/python'


def command(args, timeout=120):
    p = subprocess.run(args, capture_output=True, text=True, encoding='utf-8', timeout=timeout)
    if p.returncode:
        raise RuntimeError(p.stdout + '\n' + p.stderr)
    return p.stdout


def ssh(code, timeout=120):
    return command(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', HOST,
                    shlex.quote(BOOTSTRAP) + ' -B -c ' + shlex.quote(code)], timeout)


def upload():
    ssh("from pathlib import Path;r=Path(" + repr(REMOTE) + ");"
        "assert not (r/'verification/training_launch.json').exists(), 'Preserve launched sources'")
    excluded = {'.git', '__pycache__', 'environment', 'data', 'checkpoints', 'runs', 'logs', 'reports', 'predictions', 'weights'}
    files = [p for p in ROOT.rglob('*') if p.is_file()
             and p.relative_to(ROOT).parts[0] not in excluded
             and not set(p.relative_to(ROOT).parts) & {'.git', '__pycache__'}]
    dirs = {p.parent.relative_to(ROOT).as_posix() for p in files} | {'logs', 'runs', 'verification', 'checkpoints', 'reports', 'predictions', 'weights'}
    ssh('from pathlib import Path;r=Path(' + repr(REMOTE) + ');'
        '[(r/d).mkdir(parents=True,exist_ok=True) for d in ' + repr(sorted(dirs)) + ']')
    with tempfile.TemporaryDirectory(prefix='xoct_sync_') as tmp:
        def worker(i):
            batch = Path(tmp)/f'{i}.txt'
            batch.write_text('\n'.join('put "'+p.as_posix()+'" "'+REMOTE+'/'+p.relative_to(ROOT).as_posix()+'"' for p in files[i::6])+'\n', 'utf-8')
            return command(['sftp','-q','-o','BatchMode=yes','-b',str(batch),HOST], 600)
        with ThreadPoolExecutor(max_workers=6) as pool:
            list(pool.map(worker, range(6)))
    return {'files_uploaded': len(files), 'destination': REMOTE, 'archive_created': False}


def fetch():
    relatives = json.loads(ssh('from pathlib import Path;import json;r=Path('+repr(REMOTE)+');'
        "print(json.dumps([str(p.relative_to(r)) for d in ['verification','runs','logs','reports'] for p in (r/d).rglob('*') if p.is_file() and p.suffix in ['.json','.jsonl','.log','.csv','.png']]))"))
    with tempfile.TemporaryDirectory(prefix='xoct_fetch_') as tmp:
        batch = Path(tmp)/'fetch.txt'; lines = []
        for relative in relatives:
            dest = ROOT/relative;dest.parent.mkdir(parents=True,exist_ok=True)
            lines.append('get "'+REMOTE+'/'+relative+'" "'+dest.as_posix()+'"')
        batch.write_text('\n'.join(lines)+'\n','utf-8')
        command(['sftp','-q','-o','BatchMode=yes','-b',str(batch),HOST],600)
    return {'files_downloaded':len(relatives),'checkpoints_remain_on_server':True}


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['upload','fetch','exec']);p.add_argument('--code-file',type=Path)
    a=p.parse_args()
    if a.action=='upload': print(json.dumps(upload()))
    elif a.action=='fetch': print(json.dumps(fetch()))
    else: print(ssh(a.code_file.read_text('utf-8'), timeout=600))
